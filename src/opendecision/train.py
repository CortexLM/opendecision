"""Training stages on one model. Each stage is a plain function over an iterator of batches so it
runs on CPU for tests and under torchrun/DDP for real runs.

Batch dict: ids, mask, opt_ids, opt_mask, instr_ids, qtype, y [B,Q] (y == K means the unknown slot)."""
from __future__ import annotations
import torch
from . import losses as L
from .tokenizer import MASK

RULES = {"log": L.log_loss, "brier": L.brier, "spherical": L.spherical, "rps": L.rps}

def mlm_loss(model, ids, positions, *, gather=None):
    """Fixed-shape MLM loss on packed rows ids [B,N] (mask None), positions [B,n_mask] (sorted, from masking.py).
    fp32 mean over B*n_mask. gather(h [B*N,d], rows [B*n_mask]) defaults to torch indexing (torch.gather semantics)."""
    B, N = ids.shape
    n = positions.size(1)
    inp = ids.scatter(1, positions, torch.full_like(positions, MASK))
    h = model.encode_tokens(inp)
    rows = (torch.arange(B, device=ids.device)[:, None] * N + positions).reshape(-1)
    flat = h.reshape(B * N, -1)
    hm = flat[rows] if gather is None else gather(flat, rows)
    logits = (hm @ model.tok.weight.T).float()
    tgt = ids.gather(1, positions).reshape(-1)
    return torch.nn.functional.cross_entropy(logits, tgt)

def decision_loss(logits, opt_mask_ext, y, qtype, rule="log", rps_w=1.0):
    """Per-question losses, weighted (not compacted) so shapes stay constant. RPS on qtype==1, `rule` elsewhere.
    The mask is already -inf in logits; opt_mask_ext is kept for signature symmetry with distill_loss."""
    logits = logits.float()
    sel = qtype == 1
    base = RULES[rule](logits, y, reduction="none")
    ordl = L.rps(logits, y, reduction="none")
    z = torch.zeros_like(base)
    b = torch.where(~sel, base, z).sum() / (~sel).sum().clamp_min(1)
    o = torch.where(sel, ordl, z).sum() / sel.sum().clamp_min(1)
    return b + rps_w * o

def distill_loss(logits, opt_mask_ext, teacher, T=1.0):
    """KD from cached teacher distributions teacher [B,Q,K+1] (unknown included, padded = 0)."""
    logp = torch.log_softmax(logits.float() / T, -1).masked_fill(~opt_mask_ext, 0)
    return -(teacher.float() * logp).sum(-1).mean()

def _step(model, opt, loss):
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss.item()

def _fwd(model, b):
    return model(b["ids"], b["mask"], b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"])

def finetune_step(model, opt, b, rule="log", rps_w=1.0, coh=None, coh_w=0.1):
    """Supervised strictly-proper loss (stage 2). `coh` = optional callable(model,b)->loss for coherence."""
    model.train()
    logits, ext = _fwd(model, b)
    loss = decision_loss(logits, ext, b["y"], b["qtype"], rule, rps_w)
    if coh is not None:
        loss = loss + coh_w * coh(model, b)
    return _step(model, opt, loss)

def mlm_step(model, opt, ids, positions):
    """Stage 0: masked-token pretraining; positions from opendecision.masking.mask_positions."""
    model.train()
    return _step(model, opt, mlm_loss(model, ids, positions))

def distill_step(model, opt, b, teacher_probs, T=1.0):
    """Stage 1: KD from cached (offline) teacher distributions [B,Q,K+1]."""
    model.train()
    logits, ext = _fwd(model, b)
    return _step(model, opt, distill_loss(logits, ext, teacher_probs, T))

def rl_step(model, opt, b, outcome_fn, samples=4):
    """Stage 3. NOT replay-safe (torch.multinomial). (ONLY for outcome/bandit feedback): sample actions, reward r=c-p_a, leave-one-out baseline."""
    model.train()
    logits, _ = _fwd(model, b)
    p = torch.softmax(logits, -1)
    flat = p.reshape(-1, p.size(-1))
    a = torch.multinomial(flat, samples, replacement=True)                 # [BQ, S]
    lg = logits.reshape(-1, logits.size(-1)).unsqueeze(1).expand(-1, samples, -1)
    correct = outcome_fn(a.reshape(*p.shape[:2], samples))                 # [B,Q,S] bool
    loss = L.bandit_rl(lg.reshape(-1, lg.size(-1)), a.reshape(-1), correct.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
