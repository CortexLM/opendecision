"""Improvement C: doc-boundary mask, span masking, ELECTRA RTD (all flagged, v1 default unchanged)."""
import dataclasses

import numpy as np
import torch

from opendecision.masking import doc_mask, mask_positions, rtd_gumbel, span_positions
from opendecision.model import ModelConfig, OpenDecisionModel, param_names
from opendecision.train import mlm_loss, rtd_loss, rtd_step

CFG = ModelConfig(d=32, layers=2, heads=4, head_layers=1, max_len=64)
EOS = 258

def _rows(B=2, N=24, seed=0):
    r = np.random.default_rng(seed).integers(3, 258, (B, N)).astype(np.uint32)
    r[0, [5, 11]] = EOS
    r[1, 0] = EOS
    return r

def test_doc_mask_hand_built_row():
    m = doc_mask(np.array([[7, EOS, 8, 9, EOS, 4]]), EOS)[0]
    seg = [0, 0, 1, 1, 1, 2]
    want = np.array([[a == b for b in seg] for a in seg])
    assert m.dtype == bool and m.shape == (6, 6) and (m == want).all()
    assert doc_mask(np.array([[4, 5, 6]]), EOS).all()

def test_doc_mask_off_is_v1_bitwise_and_on_changes_loss():
    torch.manual_seed(0)
    m = OpenDecisionModel(CFG)
    ids = torch.from_numpy(_rows().astype(np.int64))
    pos = torch.from_numpy(mask_positions(ids.numpy(), 1, 4))
    with torch.no_grad():
        a, b = mlm_loss(m, ids, pos), mlm_loss(m, ids, pos, eos_id=None)
        full = torch.ones(2, 24, 24, dtype=torch.bool)
        h_v1 = m.encode_tokens(ids)
        h_full = m.encode_tokens(ids, full)
        c = mlm_loss(m, ids, pos, eos_id=EOS)
    assert torch.equal(a, b)
    assert torch.allclose(h_v1, h_full, atol=1e-6)
    assert not torch.equal(a, c) and torch.isfinite(c)

def test_doc_mask_isolates_documents():
    torch.manual_seed(0)
    m = OpenDecisionModel(CFG).eval()
    ids = torch.from_numpy(_rows().astype(np.int64))
    alt = ids.clone()
    alt[0, 12:] = 5
    with torch.no_grad():
        h1 = m.encode_tokens(ids, torch.from_numpy(doc_mask(ids.numpy(), EOS)))
        h2 = m.encode_tokens(alt, torch.from_numpy(doc_mask(alt.numpy(), EOS)))
    assert torch.allclose(h1[0, :12], h2[0, :12], atol=1e-6)

def test_span_positions_contract_determinism_and_mean_span():
    rows = np.random.default_rng(1).integers(3, 258, (64, 512)).astype(np.uint32)
    n = 77
    p = span_positions(rows, 5, n)
    assert p.shape == (64, n) and p.dtype == np.int64
    assert all(len(set(r)) == n and (np.diff(r) > 0).all() and r.min() >= 0 and r.max() < 512 for r in p)
    assert np.array_equal(p, span_positions(rows, 5, n))
    assert not np.array_equal(p, span_positions(rows, 6, n))
    runs = [len(s) for r in p for s in np.split(r, np.flatnonzero(np.diff(r) != 1) + 1)]
    # truncated geometric(1/3) at 10 has mean 2.948; merged neighbours push observed runs up slightly
    assert 2.6 < float(np.mean(runs)) < 3.6 and max(runs) <= 30
    assert np.array_equal(span_positions(rows[:2, :8], 0, 8), np.tile(np.arange(8), (2, 1)))

def test_rtd_param_names_and_module_agree():
    c = dataclasses.replace(CFG, rtd_gen_ratio=0.5)
    names = param_names(c)
    assert "gen.proj_in.weight" in names and names["gen.proj_in.weight"] == (16, 32)
    assert not any(k.startswith("gen.") for k in param_names(CFG))
    with torch.device("meta"):
        sd = OpenDecisionModel(c).state_dict()
    assert {k: tuple(v.shape) for k, v in sd.items()} == names

def _rtd_run(seed=0):
    c = dataclasses.replace(CFG, rtd_gen_ratio=0.5)
    torch.manual_seed(0)
    m = OpenDecisionModel(c)
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    ids = torch.from_numpy(_rows().astype(np.int64))
    pos = torch.from_numpy(mask_positions(ids.numpy(), 1, 6))
    losses = [rtd_step(m, opt, ids, pos, seed=seed, eos_id=EOS) for _ in range(2)]
    with torch.no_grad():
        final = rtd_loss(m, ids, pos, seed=seed, eos_id=EOS)
    return losses, final

def test_rtd_loss_finite_and_deterministic():
    (l1, f1), (l2, f2) = _rtd_run(), _rtd_run()
    assert all(np.isfinite(l1)) and l1 == l2 and torch.equal(f1, f2)
    rows = _rows()
    assert np.array_equal(rtd_gumbel(rows, 3, 6, 11), rtd_gumbel(rows, 3, 6, 11))
    assert not np.array_equal(rtd_gumbel(rows, 3, 6, 11), rtd_gumbel(rows, 4, 6, 11))
