"""Fixed-shape decision record (design A11). Flat u16 layout:
state | instr Q*Ti | opts Q*K*T | qtype Q | y Q | teacher_q Q*(K+1)
The unknown answer is ALWAYS index n_options (the slot `decide` appends), whatever the question's own K_q."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .tokenizer import PAD

@dataclass(frozen=True)
class RecordShape:
    state_len: int
    n_questions: int
    n_options: int
    opt_len: int
    instr_len: int

    @property
    def length(self) -> int:
        q, k = self.n_questions, self.n_options
        return self.state_len + q * self.instr_len + q * k * self.opt_len + q + q + q * (k + 1)

def qmax_for(vocab: int) -> int:
    return 2 ** ((vocab - 1).bit_length() - 1) - 1

def _put(dst: np.ndarray, ids, width: int, what: str) -> None:
    if len(ids) > width:
        raise ValueError(f"{what}: {len(ids)} tokens > {width}")
    if len(ids) and (min(ids) < 0 or max(ids) > 0xFFFF):
        raise ValueError(f"{what}: token id outside u16")
    dst[: len(ids)] = ids

def pack_record(ex: dict, shape: RecordShape, qmax: int) -> np.ndarray:
    """ex: state [ids]; per question lists: instr [[ids]], opts [[[ids]]], qtype [int], y [int|None] (index into the
    question's own options; None or K_q = unknown), teacher [[p_0..p_{K_q-1}, p_unknown]] (optional)."""
    s = shape
    Q, K = s.n_questions, s.n_options
    nq = len(ex["qtype"])
    if nq > Q:
        raise ValueError(f"{nq} questions > {Q}")
    state = np.zeros(s.state_len, np.int64)
    instr = np.zeros((Q, s.instr_len), np.int64)
    opts = np.zeros((Q, K, s.opt_len), np.int64)
    qtype = np.zeros(Q, np.int64)
    y = np.full(Q, K, np.int64)
    teacher = np.zeros((Q, K + 1), np.int64)
    _put(state, ex["state"], s.state_len, "state")
    for q in range(nq):
        _put(instr[q], ex["instr"][q], s.instr_len, "instr")
        kq = len(ex["opts"][q])
        if kq > K:
            raise ValueError(f"{kq} options > {K}")
        for k, o in enumerate(ex["opts"][q]):
            _put(opts[q, k], o, s.opt_len, "option")
        qtype[q] = ex["qtype"][q]
        g = ex["y"][q]
        if g is not None and g != kq:
            if not 0 <= g < kq:
                raise ValueError(f"gold {g} outside 0..{kq - 1}")
            y[q] = g
        t = ex.get("teacher")
        if t is not None and t[q] is not None:
            if len(t[q]) != kq + 1:
                raise ValueError("teacher must have K_q+1 entries (unknown last)")
            qv = np.rint(np.asarray(t[q], np.float64) * qmax).astype(np.int64)
            teacher[q, :kq], teacher[q, K] = qv[:kq], qv[kq]
    out = np.concatenate([state, instr.ravel(), opts.ravel(), qtype, y, teacher.ravel()])
    assert out.size == s.length
    return out.astype(np.uint16)

def unpack_records(x: torch.Tensor, shape: RecordShape, qmax: int) -> dict[str, torch.Tensor]:
    s = shape
    Q, K = s.n_questions, s.n_options
    x = x.long()
    B = x.size(0)
    sizes = [s.state_len, Q * s.instr_len, Q * K * s.opt_len, Q, Q, Q * (K + 1)]
    st, ins, op, qt, y, te = torch.split(x, sizes, dim=1)
    opt_ids = op.reshape(B, Q, K, s.opt_len)
    return {"ids": st, "mask": st != PAD, "opt_ids": opt_ids, "opt_mask": opt_ids[..., 0] != PAD,
            "instr_ids": ins.reshape(B, Q, s.instr_len), "qtype": qt, "y": y,
            "teacher": te.reshape(B, Q, K + 1).float() / qmax}
