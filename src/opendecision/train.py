"""Training stages on one model. Each stage is a plain function over an iterator of batches so it
runs on CPU for tests and under torchrun/DDP for real runs.

Batch dict: ids, mask, opt_ids, opt_mask, instr_ids, qtype, y [B,Q] (y == K means the unknown slot)."""
from __future__ import annotations

import torch

from . import losses as L
from .masking import doc_mask, rl_gumbel, rtd_gumbel
from .tokenizer import MASK

RULES = {"log": L.log_loss, "brier": L.brier, "spherical": L.spherical, "rps": L.rps}

RTD_LAMBDA = 50.0   # ELECTRA discriminator weight (arXiv 2003.10555)
LABEL_TAU = 0.1     # label_negatives InfoNCE temperature (improvement D)

def _attn_mask(ids, eos_id):
    """None (v1, packed rows attend everywhere) or the [B,N,N] document-boundary mask from masking.doc_mask."""
    return None if eos_id is None else torch.from_numpy(doc_mask(ids.cpu().numpy(), eos_id)).to(ids.device)

def _masked_logits(h, positions, weight, gather):
    B, N = h.shape[:2]
    rows = (torch.arange(B, device=h.device)[:, None] * N + positions).reshape(-1)
    flat = h.reshape(B * N, -1)
    hm = flat[rows] if gather is None else gather(flat, rows)
    return (hm @ weight.T).float()

def mlm_loss(model, ids, positions, *, gather=None, eos_id=None):
    """Fixed-shape MLM loss on packed rows ids [B,N] (mask None), positions [B,n_mask] (sorted, from masking.py).
    fp32 mean over B*n_mask. gather(h [B*N,d], rows [B*n_mask]) defaults to torch indexing (torch.gather semantics).
    eos_id (improvement C, default None = v1): attention restricted to EOS-delimited documents (doc_mask)."""
    inp = ids.scatter(1, positions, torch.full_like(positions, MASK))
    h = model.encode_tokens(inp, _attn_mask(ids, eos_id))
    logits = _masked_logits(h, positions, model.tok.weight, gather)
    tgt = ids.gather(1, positions).reshape(-1)
    return torch.nn.functional.cross_entropy(logits, tgt)

def rtd_loss(model, ids, positions, *, seed: int, eos_id=None, lam=RTD_LAMBDA, gather=None):
    """ELECTRA (arXiv 2003.10555) on packed rows: generator MLM CE at `positions` + lam * discriminator BCE over
    all B*N tokens (label = token differs from the original). Generator samples = argmax(logits + Gumbel) with
    Philox noise from masking.rtd_gumbel(ids, seed) (replay-safe; no torch sampler). Needs rtd_gen_ratio > 0."""
    N = ids.size(1)
    am = _attn_mask(ids, eos_id)
    inp = ids.scatter(1, positions, torch.full_like(positions, MASK))
    hg = model.gen(model._embed(inp), am, *model._rope(N))
    glog = _masked_logits(hg, positions, model.tok.weight, gather)
    gen = torch.nn.functional.cross_entropy(glog, ids.gather(1, positions).reshape(-1))
    g = torch.from_numpy(rtd_gumbel(ids.cpu().numpy(), seed, positions.size(1), glog.size(-1))).to(ids.device)
    sample = (glog.detach().double().reshape(g.shape) + g).argmax(-1)
    corrupt = ids.scatter(1, positions, sample)
    logit = model.rtd_disc(model.encode_tokens(corrupt, am)).squeeze(-1).float()
    disc = torch.nn.functional.binary_cross_entropy_with_logits(logit, (corrupt != ids).float())
    return gen + lam * disc

def decision_loss(logits, opt_mask_ext, y, qtype, rule="log", rps_w=1.0, brier_w=1.0):
    """Per-question losses, weighted (not compacted) so shapes stay constant. RPS on qtype==1, `rule` elsewhere.
    The mask is already -inf in logits; opt_mask_ext is kept for signature symmetry with distill_loss.
    rule "ce_brier" (improvement D) = log loss + brier_w * Brier."""
    logits = logits.float()
    sel = qtype == 1
    if rule == "ce_brier":
        base = L.ce_brier(logits, y, reduction="none", w=brier_w)
    else:
        base = RULES[rule](logits, y, reduction="none")
    ordl = L.rps(logits, y, reduction="none")
    z = torch.zeros_like(base)
    b = torch.where(~sel, base, z).sum() / (~sel).sum().clamp_min(1)
    o = torch.where(sel, ordl, z).sum() / sel.sum().clamp_min(1)
    return b + rps_w * o

def distill_loss(logits, opt_mask_ext, teacher, T=1.0, *, alpha=None, y=None, t2=False):
    """KD from cached teacher distributions teacher [B,Q,K+1] (unknown included, padded = 0).
    v1 (alpha None, t2 False): -sum teacher * log_softmax(logits/T), mean over B*Q.
    t2 (improvement D, fixes W9): the teacher is tempered consistently, teacher^(1/T) renormalised, and KD is
    scaled by T^2 (Hinton et al. 2015) so its gradient magnitude does not shrink with T.
    alpha (improvement D): per question with a teacher label, alpha*KD + (1-alpha)*CE(y) at T=1; questions with
    no teacher label (all zeros) take the CE term alone. y == K is the unknown slot."""
    lf = logits.float()
    tf = teacher.float()
    if t2:
        tp = tf.pow(1.0 / T)
        tf = tp / tp.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = torch.log_softmax(lf / T, -1).masked_fill(~opt_mask_ext, 0)
    kd = -(tf * logp).sum(-1) * (T * T if t2 else 1.0)
    if alpha is None:
        return kd.mean()
    ce = L.log_loss(lf, y, reduction="none")
    has_t = teacher.sum(-1) > 0
    return torch.where(has_t, alpha * kd + (1 - alpha) * ce, ce).mean()

def _pool(h, hmask):
    """Masked mean over state positions (hmask None = all real)."""
    if hmask is None:
        return h.mean(1)
    w = hmask.unsqueeze(-1).to(h.dtype)
    return (h * w).sum(1) / w.sum(1).clamp_min(1)

def label_negatives_loss(model, h, hmask, opt_ids, opt_mask, instr_ids, qtype, y, tau=LABEL_TAU):
    """In-batch label negatives (improvement D, for K>8): anchor = instruction embedding + pooled state [B,Q,d],
    candidates = every real option label embedding of the batch with the same qtype, positive = gold."""
    opt = model.embed_text(opt_ids)
    anchor = model.embed_text(instr_ids) + _pool(h, hmask).unsqueeze(1)
    return L.label_contrastive(opt, y, tau, query=anchor, qtype=qtype, opt_mask=opt_mask, opt_ids=opt_ids)

def _step(model, opt, loss):
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss.item()

def _fwd(model, b):
    return model(b["ids"], b["mask"], b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"])

def finetune_step(model, opt, b, rule="log", rps_w=1.0, coh=None, coh_w=0.1, *, brier_w=1.0, label_neg=0.0):
    """Supervised strictly-proper loss (stage 2). `coh` = optional callable(model,b)->loss for coherence.
    label_neg > 0 adds label_neg * label_negatives_loss (improvement D)."""
    model.train()
    if label_neg:
        h, hm = model.encode_state(b["ids"], b["mask"])
        logits, ext = model.decide(h, hm, b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"])
    else:
        logits, ext = _fwd(model, b)
    loss = decision_loss(logits, ext, b["y"], b["qtype"], rule, rps_w, brier_w)
    if label_neg:
        loss = loss + label_neg * label_negatives_loss(
            model, h, hm, b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"], b["y"])
    if coh is not None:
        loss = loss + coh_w * coh(model, b)
    return _step(model, opt, loss)

def mlm_step(model, opt, ids, positions, *, eos_id=None):
    """Stage 0: masked-token pretraining; positions from masking.mask_positions (or span_positions)."""
    model.train()
    return _step(model, opt, mlm_loss(model, ids, positions, eos_id=eos_id))

def rtd_step(model, opt, ids, positions, *, seed: int, eos_id=None):
    """Stage 0 alternative (objective "rtd"): ELECTRA generator + discriminator, see rtd_loss."""
    model.train()
    return _step(model, opt, rtd_loss(model, ids, positions, seed=seed, eos_id=eos_id))

def distill_step(model, opt, b, teacher_probs, T=1.0, *, alpha=None, t2=False):
    """Stage 1: KD from cached (offline) teacher distributions [B,Q,K+1]; alpha/t2 see distill_loss."""
    model.train()
    logits, ext = _fwd(model, b)
    return _step(model, opt, distill_loss(logits, ext, teacher_probs, T, alpha=alpha, y=b.get("y"), t2=t2))

def rl_step(model, opt, b, outcome_fn, samples=4, *, seed: int, step: int):
    """Stage 3, replay-safe: actions = Gumbel-max with Philox noise pure in (seed, step, b["ids"] row bytes, question).
    RL only for outcome feedback; with labels use decision_loss/distill_loss (rlcd-research.md: CE>=proper>PG,
    arXiv 2610.02486). Reward r=c-p_a, leave-one-out baseline (losses.bandit_rl)."""
    model.train()
    logits, _ = _fwd(model, b)
    B, Q, K = logits.shape
    g = torch.from_numpy(rl_gumbel(b["ids"].cpu().numpy(), seed, step, Q, samples, K)).to(logits.device)
    a = (logits.detach().double().unsqueeze(2) + g).argmax(-1).reshape(B * Q, samples)   # [BQ, S]
    lg = logits.reshape(-1, logits.size(-1)).unsqueeze(1).expand(-1, samples, -1)
    correct = outcome_fn(a.reshape(B, Q, samples))                 # [B,Q,S] bool
    loss = L.bandit_rl(lg.reshape(-1, lg.size(-1)), a.reshape(-1), correct.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
