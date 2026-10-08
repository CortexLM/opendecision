"""Offline teacher labelling (LLM/VLM allowed at TRAINING time only; Apache teachers: Qwen3, Qwen3-VL,
Mistral-Small. Gemma outputs make the student a Gemma Model Derivative; Llama needs attribution).

A teacher is any callable (state, question_dict, option_order) -> list[float] of option probabilities
(e.g. softmax over the option-letter logits of an LLM, or self-consistency vote fractions)."""
from __future__ import annotations
import numpy as np


def cyclic_average(teacher, state, q: dict, options: list[str]) -> np.ndarray:
    """Average over all cyclic rotations of option order: cancels position bias (arXiv 2608.11947)."""
    K = len(options); acc = np.zeros(K)
    for r in range(K):
        order = options[r:] + options[:r]
        p = np.asarray(teacher(state, q, order), float)
        for pos, o in enumerate(order):
            acc[options.index(o)] += p[pos]
    return acc / K


def poll(teachers: list, state, q: dict, options: list[str], temps: list[float] | None = None, unknown_floor: float = 0.0):
    """Panel of diverse-family teachers (PoLL, 2404.18796): per-teacher calibrated (temperature on log p), averaged.
    Returns (probs over options, disagreement). Unknown mass P(U) = max(unknown_floor, mean total-variation
    disagreement), because hosted models hide it (2609.35342)."""
    temps = temps or [1.0] * len(teachers)
    ps = []
    for t, T in zip(teachers, temps):
        p = np.clip(cyclic_average(t, state, q, options), 1e-9, 1) ** (1 / T)
        ps.append(p / p.sum())
    ps = np.stack(ps)
    mean = ps.mean(0)
    dis = float(0.5 * np.abs(ps - mean).sum(1).mean())
    return mean, dis, max(unknown_floor, dis)


def counterfactual_pairs(state: str, q: dict, alt_q: dict):
    """Training pair for anti-shortcut: same state, edited question; the label MUST differ for >=1 option
    according to the teacher, otherwise drop the pair (label-flip filter)."""
    return [(state, q), (state, alt_q)]


def swap_noul(q: dict) -> dict:
    """Yes/no swap augmentation (hosted models flip 50.5/100 on swapped polarity, 2610.00346): negate the question text."""
    return {**q, "instructions": "NOT: " + q["instructions"], "_swapped": True}
