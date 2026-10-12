"""Improvement H: alternating local/global attention (default off = v1 bit-identical)."""
import dataclasses
import hashlib

import torch

from opendecision import OpenDecisionModel
from opendecision import model as M

CFG = M.ModelConfig(d=32, layers=4, heads=4, head_layers=1, max_len=32, vocab=64)
# v1 encode_tokens digest captured before H was added (torch.manual_seed(0) init, ids seed 0)
V1_DIGEST = "c7b700415dbdba81eb590ef489205d5688c72d48a62c1432eba1314cfa5a5e6c"

def _ids(seed=0, B=2, N=16):
    return torch.randint(3, 64, (B, N), generator=torch.Generator().manual_seed(seed))

def _enc(cfg, ids):
    torch.manual_seed(0)
    with torch.no_grad():
        return OpenDecisionModel(cfg).encode_tokens(ids)

def test_local_window_off_is_v1_bitwise():
    assert M.ModelConfig().local_window == 0
    h = _enc(dataclasses.replace(CFG, global_every=2), _ids())
    assert hashlib.sha256(h.numpy().tobytes()).hexdigest() == V1_DIGEST
    assert M.param_names(CFG) == M.param_names(dataclasses.replace(CFG, local_window=4))

def test_band_mask_small_n():
    m = M.local_mask(None, 5, 2, "cpu")[0]
    want = torch.tensor([[abs(i - j) <= 1 for j in range(5)] for i in range(5)])
    assert torch.equal(m, want)
    pad = torch.tensor([[True, True, False, True, True]])
    mp = M.local_mask(pad, 5, 2, "cpu")[0]
    assert torch.equal(mp, (want & pad) | torch.eye(5, dtype=torch.bool))

def test_full_window_and_same_theta_equals_v1():
    ids = _ids()
    same = dataclasses.replace(CFG, local_window=64, global_theta=CFG.rope_theta)
    torch.testing.assert_close(_enc(same, ids), _enc(CFG, ids), rtol=0, atol=1e-6)
    narrow = dataclasses.replace(CFG, local_window=4)
    assert not torch.allclose(_enc(narrow, ids), _enc(CFG, ids))

def test_local_layer_only_sees_its_band():
    ids, ids2 = _ids(), _ids().clone()
    ids2[:, 0] = 3 + (ids2[:, 0] - 2) % 61
    torch.manual_seed(0)
    m = OpenDecisionModel(dataclasses.replace(CFG, layers=2, local_window=2, global_every=2))
    with torch.no_grad():
        g = m.encoder.blocks[0]
        g.o.weight.zero_(), g.ff[2].weight.zero_(), g.ff[2].bias.zero_()
        a, b = m.encode_tokens(ids), m.encode_tokens(ids2)
    assert not torch.equal(a[:, 1], b[:, 1])
    assert torch.equal(a[:, 2:], b[:, 2:])

def test_local_global_backward_deterministic_twice():
    cfg = dataclasses.replace(CFG, local_window=4, chunk=8, chunk_latents=2, global_layers=1)

    def grads():
        torch.manual_seed(0)
        m = OpenDecisionModel(cfg)
        h, _ = m.encode_state(_ids(N=20), torch.ones(2, 20, dtype=torch.bool))
        h.float().pow(2).mean().backward()
        return [p.grad.clone() for p in m.parameters() if p.grad is not None]

    a, b = grads(), grads()
    assert a and all(torch.equal(x, y) for x, y in zip(a, b))
