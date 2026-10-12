"""Deterministic MLM masking: positions are a pure function of (seed, row bytes). Static masking on purpose
(a sample gets the same mask at every exposure). ponytail: add the epoch to the key for multi-epoch runs."""
from __future__ import annotations
import hashlib
import numpy as np

def row_keys(rows: np.ndarray, seed: int, tag: bytes = b"od-mlm-v1") -> np.ndarray:
    """uint64 [B]: first 8 bytes (little endian) of sha256(tag + u64le(seed) + row as u32le), masked to 63 bits."""
    rows = np.ascontiguousarray(rows).astype("<u4", copy=False)
    pre = tag + int(seed).to_bytes(8, "little")
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

# EOS id in model id space per preset (improvement C doc mask). od-base/od-large: hypertrain packs ModernBERT shards
# (registry tokenizer modernbert-base, tokenizer.json sha256 9fd55248...b7b1) with tok.eos_id + OFFSET =
# <|endoftext|> 50279 + 3 = 50282 (hypertrain datasets/build.py). od-tiny (byte vocab, tests): byte 0xFF + 3 = 258,
# a byte that never occurs in valid UTF-8.
DOC_EOS = {"od-tiny": 258, "od-base": 50282, "od-large": 50282}

def doc_mask(rows: np.ndarray, eos_id: int) -> np.ndarray:
    """bool [B,N,N] block-diagonal attention mask: token i attends j iff both are in the same EOS-delimited
    segment; an EOS belongs to the segment it ends. Derived from the row at load time (packing unchanged)."""
    e = np.asarray(rows) == eos_id
    seg = np.cumsum(e, axis=1) - e
    return seg[:, :, None] == seg[:, None, :]

def _u01(raw: np.ndarray) -> np.ndarray:
    return ((raw >> np.uint64(11)) + 0.5) * 2.0**-53

def span_positions(rows: np.ndarray, seed: int, n_mask: int, mean_span: float = 3.0,
                   max_span: int = 10) -> np.ndarray:
    """Span masking (SpanBERT, arXiv 1907.10529): same contract as mask_positions (sorted int64 [B,n_mask],
    exactly n_mask unique positions per row). Per row, Philox(key of b"od-span-v1") draws pairs (length, start):
    length = min(max_span, Geometric(p=1/mean_span) on {1,2,..}), start uniform in [0, N-length]; not-yet-masked
    positions of the span are added left to right until n_mask is reached (the last span is truncated)."""
    n = rows.shape[1]
    if not 1 <= n_mask <= n:
        raise ValueError(f"n_mask {n_mask} not in [1, {n}]")
    lq = np.log1p(-1.0 / mean_span) if mean_span > 1 else -np.inf
    out = np.empty((rows.shape[0], n_mask), dtype=np.int64)
    for i, k in enumerate(row_keys(rows, seed, b"od-span-v1")):
        bg = np.random.Philox(key=int(k))
        got = np.zeros(n, dtype=bool)
        cnt = 0
        while cnt < n_mask:   # terminates a.s.: every draw has P(length 1 at any free position) > 0
            u = _u01(bg.random_raw(2 * n)).reshape(-1, 2)
            for ul, us in u:
                ln = min(max_span, n, 1 + int(np.floor(np.log(ul) / lq)) if lq > -np.inf else 1)
                s = min(int(us * (n - ln + 1)), n - ln)
                for j in range(s, s + ln):
                    if not got[j] and cnt < n_mask:
                        got[j] = True
                        cnt += 1
                if cnt == n_mask:
                    break
        out[i] = np.flatnonzero(got)
    return out

def rtd_gumbel(rows: np.ndarray, seed: int, n: int, V: int) -> np.ndarray:
    """float64 Gumbel noise [B,n,V] for ELECTRA generator sampling (Gumbel-max, no torch sampler), pure in
    (seed, row bytes): Philox(row_keys(rows, seed, b"od-rtd-v1")).random_raw(n*V) -> -log(-log u)."""
    g = np.empty((rows.shape[0], n * V), dtype=np.float64)
    for i, k in enumerate(row_keys(rows, seed, b"od-rtd-v1")):
        g[i] = -np.log(-np.log(_u01(np.random.Philox(key=int(k)).random_raw(n * V))))
    return g.reshape(rows.shape[0], n, V)

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
