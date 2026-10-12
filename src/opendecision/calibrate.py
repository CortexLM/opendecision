import math

import torch
import numpy as np


def nll(logits, y):
    return torch.nn.functional.cross_entropy(logits, y).item()


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, lo=0.5, hi=5.0) -> float:
    """Fit on a DISJOINT labelled split. Clamped to [lo,hi]."""
    ls = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([ls], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        l = torch.nn.functional.cross_entropy(logits / ls.exp(), y)
        l.backward()
        return l
    opt.step(closure)
    return float(ls.detach().exp().clamp(lo, hi))


def fit_temperature_per_qtype(logits: torch.Tensor, y: torch.Tensor, qtype: torch.Tensor, lo=0.5, hi=5.0,
                              steps=200) -> dict[str, float]:
    """Per-question-type temperature (improvement D) on a DISJOINT labelled split: logits [N,K(+1)] (-inf padding ok),
    y [N], qtype [N] (QTYPES ids). Deterministic 1-D search: golden section on log T in [log lo, log hi] (NLL of
    a temperature-scaled softmax is unimodal in log T on this range). Types without rows keep T=1.
    Use: model.log_temp[k] = log(T_k), then train stage C with ModelConfig.freeze_log_temp=True."""
    from .model import QTYPES
    out = {}
    lg, y = logits.double(), y.long()
    for name, i in QTYPES.items():
        sel = qtype == i
        if not bool(sel.any()):
            out[name] = 1.0
            continue
        z, t = lg[sel], y[sel]
        f = lambda u, z=z, t=t: float(torch.nn.functional.cross_entropy(z / math.exp(u), t))
        a, b = math.log(lo), math.log(hi)
        g = (math.sqrt(5) - 1) / 2
        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(steps):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a); fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a); fd = f(d)
            if b - a < 1e-9:
                break
        out[name] = math.exp((a + b) / 2)
    return out

def ece(probs: np.ndarray, y: np.ndarray, bins=15, adaptive=True) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    order = np.argsort(conf)
    chunks = np.array_split(order, bins) if adaptive else [np.where((conf > a) & (conf <= b))[0] for a, b in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:])]
    return float(sum(len(c) / len(y) * abs(acc[c].mean() - conf[c].mean()) for c in chunks if len(c)))


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    oh = np.eye(probs.shape[1])[y]
    return float(((probs - oh) ** 2).sum(1).mean())


def conformal_threshold(probs: np.ndarray, y: np.ndarray, alpha=0.1) -> float:
    """Split conformal: prediction set = {k: p_k >= 1 - q}. Marginal coverage only; NOT calibrated probabilities."""
    s = 1 - probs[np.arange(len(y)), y]
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def energy(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Energy score -T*logsumexp(logits/T) over REAL options (exclude unknown slot); higher = more OOD
    (arXiv 2010.03759). Fit the gate threshold on held-out out-of-scope requests from other schemas."""
    z = logits / T
    m = z.max(-1, keepdims=True)
    return -T * (m.squeeze(-1) + np.log(np.exp(z - m).sum(-1)))
