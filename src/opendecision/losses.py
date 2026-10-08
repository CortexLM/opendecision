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
