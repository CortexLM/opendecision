"""Training stages on one model. Each stage is a plain function over an iterator of batches so it
runs on CPU for tests and under torchrun/DDP for real runs.

Batch dict: ids, mask, opt_ids, opt_mask, instr_ids, qtype, y [B,Q] (y == K means the unknown slot)."""
from __future__ import annotations
import torch
from . import losses as L

RULES = {"log": L.log_loss, "brier": L.brier, "spherical": L.spherical, "rps": L.rps}


def _loss(logits, b, rule, rps_w):
    y = b["y"]
    sel = b["qtype"] == 1
    base = RULES[rule](logits[~sel], y[~sel]) if (~sel).any() else logits.sum() * 0
    ordl = L.rps(logits[sel], y[sel]) if sel.any() else logits.sum() * 0
    return base + rps_w * ordl


def finetune_step(model, opt, b, rule="log", rps_w=1.0, coh=None, coh_w=0.1):
    """Supervised strictly-proper loss (stage 2). `coh` = optional callable(model,b)->loss for coherence."""
    model.train()
    logits = model(b["ids"], b["mask"], b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"])
    loss = _loss(logits, b, rule, rps_w)
    if coh is not None:
        loss = loss + coh_w * coh(model, b)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss.item()


def mlm_step(model, opt, ids, mask, mask_ratio=0.3, mask_id=1):
    """Stage 0: masked-token pretraining of the fusion encoder (CLM->MLM warm start is a data-schedule concern)."""
    model.train()
    m = (torch.rand_like(ids, dtype=torch.float) < mask_ratio) & mask
    inp = ids.masked_fill(m, mask_id)
    h = model.encode_tokens(inp, mask)
    logits = h @ model.tok.weight.T
    loss = torch.nn.functional.cross_entropy(logits[m], ids[m])
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def distill_step(model, opt, b, teacher_probs, T=1.0):
    """Stage 1: KD from cached (offline) teacher distributions [B,Q,K] (padded K = 0)."""
    model.train()
    logits = model(b["ids"], b["mask"], b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"])
    logp = torch.log_softmax(logits / T, -1)
    loss = -(teacher_probs * logp.masked_fill(~b["opt_mask"], 0)).sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def rl_step(model, opt, b, outcome_fn, samples=4):
    """Stage 3 (ONLY for outcome/bandit feedback): sample actions, reward r=c-p_a, leave-one-out baseline."""
    model.train()
    logits = model(b["ids"], b["mask"], b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"])
    p = torch.softmax(logits, -1)
    flat = p.reshape(-1, p.size(-1))
    a = torch.multinomial(flat, samples, replacement=True)                 # [BQ, S]
    lg = logits.reshape(-1, logits.size(-1)).unsqueeze(1).expand(-1, samples, -1)
    correct = outcome_fn(a.reshape(*p.shape[:2], samples))                 # [B,Q,S] bool
    loss = L.bandit_rl(lg.reshape(-1, lg.size(-1)), a.reshape(-1), correct.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
