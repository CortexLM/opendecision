import dataclasses
import hashlib

import pytest
import torch

from opendecision import OpenDecisionModel
from opendecision import model as M
from opendecision.presets import PRESETS

TINY = PRESETS["od-tiny"]
# sha256 of the od-tiny logits / state_dict below, computed on the pre-B code (commit e8dc52c)
GOLD_LOGITS = "38fde5c1c71e2f6a7247f4e6b81011e83ad50e07d0f2bb57c70b12cf67922e41"
GOLD_SD = "cb1787799604155af71d5cadba8fa34d1c14cf5b7db0753c1a1e2790645f7a08"


def _batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 200, (2, 16), generator=g)
    opt = torch.randint(3, 200, (2, 2, 3, 4), generator=g)
    ins = torch.randint(3, 200, (2, 2, 5), generator=g)
    return ids, None, opt, torch.ones(2, 2, 3, dtype=torch.bool), ins, torch.tensor([[0, 1], [2, 0]])


def _sha(ts):
    return hashlib.sha256(b"".join(t.detach().numpy().tobytes() for t in ts)).hexdigest()


def test_v1_default_bitwise_unchanged():
    torch.manual_seed(0)
    m = OpenDecisionModel(TINY)
    logits, _ = m(*_batch())
    assert _sha([logits]) == GOLD_LOGITS and _sha(m.state_dict().values()) == GOLD_SD


def test_swiglu_size_and_v2_names():
    assert [M.swiglu_ff(d) for d in (64, 768, 1024)] == [192, 2048, 2752]
    for g in ("none", "headwise", "elementwise"):
        cfg = dataclasses.replace(TINY, block="v2", attn_gate=g)
        names = M.param_names(cfg)
        with torch.device("meta"):
            assert {k: tuple(v.shape) for k, v in OpenDecisionModel(cfg).state_dict().items()} == names
        enc = [k for k in names if k.startswith("encoder.")]
        assert not any(k.endswith(".bias") for k in enc) and "encoder.blocks.0.ff.gate.weight" in names
        assert ("encoder.blocks.0.gate.weight" in names) == (g != "none")


def test_rope_scaling_identity_and_linear():
    base = M.rope_table(32, 16, 1e4)
    for s in ("linear", "yarn"):
        assert all(torch.equal(a, b) for a, b in zip(base, M.rope_table(32, 16, 1e4, s, 1.0, 64)))
    lin = M.rope_table(32, 16, 1e4, "linear", 4.0, 64)
    assert torch.allclose(lin[0][8], base[0][2], atol=1e-6)        # position interpolation: p -> p/4
    y = M.rope_table(32, 16, 1e4, "yarn", 4.0, 64)
    assert not torch.equal(y[0], base[0]) and abs(y[0][0, 0].item() - (0.1 * torch.log(torch.tensor(4.0)).item() + 1)) < 1e-6


def test_attn_gate_none_equals_no_gate():
    torch.manual_seed(0)
    a = OpenDecisionModel(dataclasses.replace(TINY, block="v2"))
    torch.manual_seed(0)
    b = OpenDecisionModel(dataclasses.replace(TINY, block="v2", attn_gate="none", rope_scaling="yarn", rope_factor=1.0))
    assert torch.equal(a(*_batch())[0], b(*_batch())[0])


@pytest.mark.parametrize("gate", ["none", "headwise", "elementwise"])
@pytest.mark.parametrize("scaling", ["none", "linear", "yarn"])
def test_v2_backward_deterministic(gate, scaling):
    cfg = dataclasses.replace(TINY, block="v2", attn_gate=gate, rope_scaling=scaling, rope_factor=2.0)
    torch.use_deterministic_algorithms(True)
    try:
        grads = []
        for _ in range(2):
            torch.manual_seed(0)
            m = OpenDecisionModel(cfg)
            m(*_batch())[0].float().nan_to_num(0, 0, 0).sum().backward()
            grads.append([p.grad.clone() for p in m.parameters() if p.grad is not None])
    finally:
        torch.use_deterministic_algorithms(False)
    assert grads[0] and all(torch.equal(x, y) for x, y in zip(*grads))
    assert all(torch.isfinite(g).all() for g in grads[0])
