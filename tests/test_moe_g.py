"""Improvement G: gated MoE observation path (default dense v1 untouched)."""
import dataclasses
import math

import torch

from opendecision import OpenDecisionModel
from opendecision import model as M

CFG = M.ModelConfig(d=32, layers=4, heads=4, head_layers=1, max_len=32, vocab=64)
MOE = dataclasses.replace(CFG, n_experts=8, top_k=2)

def _ids(seed=0, B=2, N=16):
    return torch.randint(3, 64, (B, N), generator=torch.Generator().manual_seed(seed))

def test_dense_default_has_no_moe_and_v1_forward_unchanged():
    assert M.ModelConfig().n_experts == 1 and not list(M.moe_layers(CFG))
    torch.manual_seed(0)
    a = OpenDecisionModel(CFG)
    torch.manual_seed(0)
    b = OpenDecisionModel(dataclasses.replace(CFG, top_k=2, moe_every=1))  # n_experts == 1: inert
    assert M.param_names(CFG) == M.param_names(dataclasses.replace(CFG, top_k=2))
    ids = _ids()
    assert torch.equal(a.encode_tokens(ids), b.encode_tokens(ids))

def test_param_names_match_state_dict_and_count():
    for blk in ("v1", "v2"):
        cfg = dataclasses.replace(MOE, block=blk)
        names = M.param_names(cfg)
        sd = OpenDecisionModel(cfg).state_dict()
        assert {k: tuple(v.shape) for k, v in sd.items()} == names
        assert list(M.moe_layers(cfg)) == [2]
        assert names["encoder.blocks.2.router.weight"] == (8, 32)
        assert not any(n.startswith("encoder.blocks.2.ff.") for n in names)
        assert any(n.startswith("encoder.blocks.2.experts.7.") for n in names)
        ffn = sum(math.prod(s) for n, s in names.items() if n.startswith("encoder.blocks.0.ff."))
        dense = M.od_param_count(dataclasses.replace(cfg, n_experts=1))
        assert M.od_param_count(cfg) == dense + 8 * 32 + 7 * ffn

def test_moe_forward_backward_deterministic_twice():
    torch.use_deterministic_algorithms(True)
    try:
        outs = []
        for _ in range(2):
            torch.manual_seed(1)
            m = OpenDecisionModel(dataclasses.replace(MOE, init_scheme="trunc002"))
            h = m.encode_tokens(_ids())
            aux = sum(sum(b.aux) for b in m.encoder.blocks if b.moe is not None)
            (h.square().mean() + aux).backward()
            outs.append((h.detach(), [p.grad.clone() for p in m.parameters() if p.grad is not None]))
        assert torch.equal(outs[0][0], outs[1][0])
        assert all(torch.equal(x, y) for x, y in zip(outs[0][1], outs[1][1], strict=True))
        assert any(n.endswith("router.weight") and p.grad is not None for n, p in m.named_parameters())
    finally:
        torch.use_deterministic_algorithms(False)

def test_route_ties_go_to_lowest_index():
    experts, keep = M.route(torch.zeros(3, 8), 2, 100)
    assert experts.tolist() == [0, 0, 0, 1, 1, 1] and keep.all()

def test_route_capacity_drops_in_rank_token_order():
    logits = torch.tensor([[2.0, 1.0, 0.0]] * 3)  # every token: rank0 -> e0, rank1 -> e1
    experts, keep = M.route(logits, 2, 2)
    assert experts.tolist() == [0, 0, 0, 1, 1, 1]
    assert keep.tolist() == [True, True, False, True, True, False]  # token 2 dropped in both ranks
