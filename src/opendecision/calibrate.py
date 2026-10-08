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
