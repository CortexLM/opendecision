"""Label-free coherence battery (cf. arXiv 2609.33209, 2412.18544). Works on any callable
`predict(state, questions)->answers`."""
from __future__ import annotations
import numpy as np


def complement_gap(decider, states: list[str], statement: str, negation: str) -> float:
    """mean |P(X)+P(not X)-1| over states for two noul questions."""
    gaps = []
    for s in states:
        r = decider.predict({"state": s, "questions": {
            "x": {"type": "noul", "instructions": statement},
            "nx": {"type": "noul", "instructions": negation}}})
        gaps.append(abs(r["answers"]["x"]["noul"] + r["answers"]["nx"]["noul"] - 1))
    return float(np.mean(gaps))


def order_flip_rate(decider, state: str, instructions: str, options: dict[str, str], trials=8, seed=0) -> float:
    """Fraction of option-order permutations that change the argmax."""
    rng = np.random.default_rng(seed)
    base = None
    flips = 0
    keys = list(options)
    for t in range(trials):
        ks = keys if t == 0 else list(rng.permutation(keys))
        r = decider.predict({"state": state, "questions": {"q": {"type": "choice", "instructions": instructions,
                             "criteria": {k: options[k] for k in ks}}}})
        a = r["answers"]["q"]["choice"]
        if base is None:
            base = a
        flips += a != base
    return flips / max(trials - 1, 1)


def counterfactual_sensitivity(decider, state: str, q1: dict, q2: dict) -> bool:
    """True if changing the question changes the answer distribution (anti-shortcut check, 2609.33689)."""
    a = decider.predict({"state": state, "questions": {"q": q1}})["answers"]["q"]["probabilities"]
    b = decider.predict({"state": state, "questions": {"q": q2}})["answers"]["q"]["probabilities"]
    return a != b
