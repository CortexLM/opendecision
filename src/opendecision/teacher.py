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


def teacher_panel(teachers: list[tuple[str, object]], state, q: dict, options: list[str],
                  temps: list[float] | None = None, unknown_floor: float = 0.0, min_families: int = 2):
    """Improvement D: PoLL panel over (family, teacher) pairs, each cyclic-averaged (see poll). Raises ValueError
    with fewer than `min_families` distinct families: one family's shared biases do not average out (2404.18796).
    Returns (probs over options, disagreement, unknown mass) like poll."""
    fams = {f for f, _ in teachers}
    if len(fams) < min_families:
        raise ValueError(f"teacher panel needs >= {min_families} distinct families, got {sorted(fams)}")
    return poll([t for _, t in teachers], state, q, options, temps, unknown_floor)

UNKNOWN_TEXT = "cannot be determined from the text"

def label_flips(teacher, state_a, q_a: dict, state_b, q_b: dict, options: list[str]) -> bool:
    """Label-flip filter: True iff the teacher's cyclic-averaged argmax differs between the two inputs."""
    pa = cyclic_average(teacher, state_a, q_a, options)
    pb = cyclic_average(teacher, state_b, q_b, options)
    return int(pa.argmax()) != int(pb.argmax())

def unknown_counterfactual(state, q: dict, options: list[str], ablated_state, teacher=None,
                           unknown_text: str = UNKNOWN_TEXT):
    """Improvement D: unknown-mass counterfactual. The caller supplies `ablated_state` (the answer-bearing sentence
    removed); this only builds the pair and its labels: [(state, q, None), (ablated_state, q, target)] with target
    = one-hot on the unknown slot (index len(options), A11 layout). With a `teacher`, the original gets the
    teacher's K+1 distribution and the pair is dropped ([]) unless the label flips TO unknown (original argmax is a
    real option, ablated argmax is the unknown option)."""
    K = len(options)
    target = np.zeros(K + 1)
    target[K] = 1.0
    if teacher is None:
        return [(state, q, None), (ablated_state, q, target)]
    ext = [*options, unknown_text]
    pa, pb = cyclic_average(teacher, state, q, ext), cyclic_average(teacher, ablated_state, q, ext)
    if int(pa.argmax()) == K or int(pb.argmax()) != K:
        return []
    return [(state, q, pa), (ablated_state, q, target)]

def counterfactual_pairs(state: str, q: dict, alt_q: dict):
    """Training pair for anti-shortcut: same state, edited question; the label MUST differ for >=1 option
    according to the teacher, otherwise drop the pair (label-flip filter)."""
    return [(state, q), (state, alt_q)]


def swap_noul(q: dict) -> dict:
    """Yes/no swap augmentation (hosted models flip 50.5/100 on swapped polarity, 2610.00346): negate the question text."""
    return {**q, "instructions": "NOT: " + q["instructions"], "_swapped": True}
