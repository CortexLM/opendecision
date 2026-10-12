"""Deterministic MLM masking: positions are a pure function of (seed, row bytes). Static masking on purpose
(a sample gets the same mask at every exposure). ponytail: add the epoch to the key for multi-epoch runs."""
from __future__ import annotations
import hashlib
import numpy as np

def row_keys(rows: np.ndarray, seed: int) -> np.ndarray:
    """uint64 [B]: first 8 bytes (little endian) of sha256(b"od-mlm-v1" + u64le(seed) + row as u32le), masked to 63 bits."""
    rows = np.ascontiguousarray(rows).astype("<u4", copy=False)
    pre = b"od-mlm-v1" + int(seed).to_bytes(8, "little")
    out = np.empty(rows.shape[0], dtype=np.uint64)
    for i in range(rows.shape[0]):
        d = hashlib.sha256(pre + rows[i].tobytes()).digest()[:8]
        out[i] = int.from_bytes(d, "little") & (2**63 - 1)
    return out

def rl_gumbel(rows: np.ndarray, seed: int, step: int, Q: int, S: int, K: int) -> np.ndarray:
    """float64 Gumbel noise [B,Q,S,K], pure in (seed, step, row bytes, q). Key per (row,q) = first 8 bytes of
    sha256(b"od-rl-v1" + u64le(seed) + u64le(step) + row as u32le + u64le(q)) & (2**63-1); Philox(key).random_raw(S*K)
    -> u = ((raw >> 11) + 0.5) * 2**-53 in (0,1) -> -log(-log u)."""
    rows = np.ascontiguousarray(rows).astype("<u4", copy=False)
    pre = b"od-rl-v1" + int(seed).to_bytes(8, "little") + int(step).to_bytes(8, "little")
    u = np.empty((rows.shape[0], Q, S * K), dtype=np.float64)
    for i in range(rows.shape[0]):
        rb = pre + rows[i].tobytes()
        for q in range(Q):
            k = int.from_bytes(hashlib.sha256(rb + q.to_bytes(8, "little")).digest()[:8], "little") & (2**63 - 1)
            u[i, q] = ((np.random.Philox(key=k).random_raw(S * K) >> np.uint64(11)) + 0.5) * 2.0**-53
    return -np.log(-np.log(u)).reshape(rows.shape[0], Q, S, K)

def n_mask_for(n: int, ratio: float) -> int:
    return max(1, round(ratio * n))

def mask_positions(rows: np.ndarray, seed: int, n_mask: int) -> np.ndarray:
    """int64 [B, n_mask], sorted per row. Philox(key).random_raw(N) -> stable argsort -> first n_mask -> sort."""
    n = rows.shape[1]
    if not 1 <= n_mask <= n:
        raise ValueError(f"n_mask {n_mask} not in [1, {n}]")
    out = np.empty((rows.shape[0], n_mask), dtype=np.int64)
    for i, k in enumerate(row_keys(rows, seed)):
        raw = np.random.Philox(key=int(k)).random_raw(n)
        out[i] = np.sort(np.argsort(raw, kind="stable")[:n_mask])
    return out
