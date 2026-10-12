"""Strictly proper scoring rules + coherence. All take logits [..., K] with -inf padding and target index."""
import torch
import torch.nn.functional as F


def _p(logits):
    return torch.softmax(logits, -1)


def _red(l, reduction):
    return l.mean() if reduction == "mean" else l

def log_loss(logits, y, reduction="mean"):
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(y.shape)
    return _red(l, reduction)

def brier(logits, y, reduction="mean"):
    p = _p(logits)
    oh = F.one_hot(y, p.size(-1)).to(p.dtype)
    return _red(((p - oh) ** 2).sum(-1), reduction)

def ce_brier(logits, y, reduction="mean", w=1.0):
    """Log loss + w * Brier (improvement D): CE ranks, Brier bounds over-confidence; typed head CE+Brier+temperature
    reached AUROC .972 vs .950 for CE alone (arXiv 2609.32160). Both terms strictly proper, so the sum is too."""
    return _red(log_loss(logits, y, "none") + w * brier(logits, y, "none"), reduction)

def label_contrastive(option_emb, target, temperature=0.1, *, query, qtype, opt_mask, opt_ids=None):
    """In-batch label negatives for K>8 (improvement D): InfoNCE of the question anchor query [B,Q,d] against
    EVERY real option of the batch option_emb [B,Q,K,d] (opt_mask [B,Q,K]) of the same qtype; positive = the gold
    option (target [B,Q] < K; unknown/padded gold is skipped). opt_ids [B,Q,K,T] (optional) masks candidates whose
    text equals the gold label (no false negatives from repeated labels). Cosine / temperature. The caller builds
    the anchor (train.label_negatives_loss: instruction embedding + mean-pooled state). Deterministic:
    type match and positive pick are one-hot matmuls/products, no scatter or index backward."""
    B, Q, K, d = option_emb.shape
    n = B * Q
    c = torch.nn.functional.normalize(option_emb.reshape(n * K, d).float(), dim=-1)
    a = torch.nn.functional.normalize(query.reshape(n, d).float(), dim=-1)
    s = a @ c.T / temperature                                            # [BQ, BQK]
    tq = F.one_hot(qtype.reshape(-1), 3).float()
    valid = ((tq @ tq.repeat_interleave(K, 0).T) > 0) & opt_mask.reshape(1, -1)
    y = target.reshape(-1)
    keep = (y < K) & opt_mask.reshape(n, K).gather(1, y.clamp_max(K - 1)[:, None]).squeeze(1)
    gold = torch.arange(n, device=y.device) * K + y.clamp_max(K - 1)
    pos = F.one_hot(gold, n * K).bool()
    if opt_ids is not None:
        flat = opt_ids.reshape(n * K, -1)
        valid &= ~((flat[gold][:, None, :] == flat[None]).all(-1) & ~pos)
    valid |= ~keep[:, None]                         # skipped rows stay finite (their loss is multiplied by 0)
    logp = torch.log_softmax(s.masked_fill(~valid, float("-inf")), -1)
    nll = -logp.masked_fill(~(pos & valid), 0).sum(-1)
    return (nll * keep).sum() / keep.sum().clamp_min(1)

def spherical(logits, y, reduction="mean"):
    p = _p(logits)
    oh = F.one_hot(y, p.size(-1)).to(p.dtype)
    return _red(-((p * oh).sum(-1) / p.norm(dim=-1)), reduction)

def rps(logits, y, reduction="mean"):
    """Ranked probability score for ordinal `score` questions."""
    p = _p(logits)
    cp = p.cumsum(-1)
    co = F.one_hot(y, p.size(-1)).cumsum(-1).to(p.dtype)
    return _red(((cp - co) ** 2).sum(-1), reduction)

def coherence(p_x, p_not_x, p_pair_a=None, p_pair_ab=None):
    """Label-free: P(X)+P(not X)=1 and P(A and B)<=min(P(A),P(B)). Inputs are probabilities."""
    loss = ((p_x + p_not_x - 1) ** 2).mean()
    if p_pair_a is not None:
        loss = loss + torch.relu(p_pair_ab - p_pair_a).pow(2).mean()
    return loss


def bandit_rl(logits, a, correct):
    """RLCD-style: r = c - p_a is an unbiased estimator of (half) the Brier gradient from one sampled
    action's outcome; leave-one-out baseline over samples of the same prompt. Use ONLY when labels
    are unavailable (outcome feedback); with labels use the supervised rules above."""
    p = _p(logits)
    pa = p.gather(-1, a.unsqueeze(-1)).squeeze(-1)
    r = (correct.float() - pa).detach()
    n = r.numel()
    base = (r.sum() - r) / max(n - 1, 1)
    logp = torch.log(pa.clamp_min(1e-9))
    return -((r - base) * logp).mean()
