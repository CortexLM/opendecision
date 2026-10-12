"""Improvement D: CE+Brier rule, per-qtype temperature, alpha KD+CE / T^2, teacher panel, unknown counterfactuals,
in-batch label negatives. Every option flagged; defaults reproduce v1 bit-for-bit."""
import dataclasses
import math

import numpy as np
import pytest
import torch

from opendecision import calibrate as C
from opendecision import losses as L
from opendecision import train as TR
from opendecision.model import ModelConfig, OpenDecisionModel
from opendecision.teacher import label_flips, teacher_panel, unknown_counterfactual

CFG = ModelConfig(d=32, layers=1, heads=4, head_layers=1, max_len=32)


def _logits(seed=0, B=2, Q=3, K=4):
    g = torch.Generator().manual_seed(seed)
    lg = torch.randn(B, Q, K + 1, generator=g)
    ext = torch.ones(B, Q, K + 1, dtype=torch.bool)
    ext[0, 1, 2:K] = False
    return lg.masked_fill(~ext, float("-inf")), ext


def _v1_decision(logits, y, qtype, rule, rps_w):          # verbatim pre-D body
    logits = logits.float()
    sel = qtype == 1
    base = TR.RULES[rule](logits, y, reduction="none")
    ordl = L.rps(logits, y, reduction="none")
    z = torch.zeros_like(base)
    return (torch.where(~sel, base, z).sum() / (~sel).sum().clamp_min(1)
            + rps_w * torch.where(sel, ordl, z).sum() / sel.sum().clamp_min(1))


def _v1_distill(logits, ext, teacher, T):                  # verbatim pre-D body
    logp = torch.log_softmax(logits.float() / T, -1).masked_fill(~ext, 0)
    return -(teacher.float() * logp).sum(-1).mean()


def test_v1_defaults_bitwise():
    lg, ext = _logits()
    y = torch.tensor([[0, 1, 4], [3, 2, 1]])
    qt = torch.tensor([[0, 1, 2], [1, 0, 0]])
    for rule in ("log", "brier"):
        assert torch.equal(TR.decision_loss(lg, ext, y, qt, rule, 0.7), _v1_decision(lg, y, qt, rule, 0.7))
    t = torch.softmax(torch.randn(2, 3, 5, generator=torch.Generator().manual_seed(1)), -1) * ext
    for T in (1.0, 2.0):
        assert torch.equal(TR.distill_loss(lg, ext, t, T), _v1_distill(lg, ext, t, T))
    torch.manual_seed(0)
    a = OpenDecisionModel(CFG)
    torch.manual_seed(0)
    b = OpenDecisionModel(dataclasses.replace(CFG, freeze_log_temp=True))
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_ce_brier_is_log_plus_weighted_brier():
    lg, ext = _logits(2)
    y = torch.tensor([[0, 1, 4], [3, 2, 1]])
    qt = torch.zeros(2, 3, dtype=torch.long)
    got = TR.decision_loss(lg, ext, y, qt, "ce_brier", 1.0, 0.5)
    ref = L.log_loss(lg, y) + 0.5 * L.brier(lg, y)
    assert torch.allclose(got, ref, atol=1e-6)


def test_fit_temperature_per_qtype_recovers_scale_and_freeze():
    g = torch.Generator().manual_seed(0)
    n = 4000
    qt = torch.arange(n) % 2                                  # choice=0 at T=2, score=1 at T=0.8
    z = torch.randn(n, 4, generator=g) * 2
    u = torch.rand(n, 1, generator=g)
    y = (torch.softmax(z, -1).cumsum(-1) < u).sum(-1).clamp_max(3)   # inverse-CDF draw
    true_t = torch.where(qt == 0, 2.0, 0.8)
    T = C.fit_temperature_per_qtype(z * true_t[:, None], y, qt)
    assert abs(T["choice"] - 2.0) < 0.15 and abs(T["score"] - 0.8) < 0.08 and T["noul"] == 1.0
    assert T == C.fit_temperature_per_qtype(z * true_t[:, None], y, qt)
    m = OpenDecisionModel(dataclasses.replace(CFG, freeze_log_temp=True))
    with torch.no_grad():
        for k, v in T.items():
            m.log_temp[k].fill_(math.log(v))
    b = _batch()
    logits, ext = m(b["ids"], b["mask"], b["opt_ids"], b["opt_mask"], b["instr_ids"], b["qtype"])
    TR.decision_loss(logits, ext, b["y"], b["qtype"], "ce_brier").backward()
    assert all(p.grad is None for p in m.log_temp.values())
    assert m.scorer[3].weight.grad is not None


def test_distill_alpha_and_t2():
    lg, ext = _logits(3)
    t = torch.softmax(torch.randn(2, 3, 5, generator=torch.Generator().manual_seed(4)), -1) * ext
    t = t / t.sum(-1, keepdim=True)
    t[1, 2] = 0                                               # no teacher label: CE only
    y = torch.tensor([[0, 1, 4], [3, 2, 1]])
    T = 2.0
    kd1 = -(t * torch.log_softmax(lg, -1).masked_fill(~ext, 0)).sum(-1)
    ce = L.log_loss(lg, y, reduction="none")
    want = torch.where(t.sum(-1) > 0, 0.3 * kd1 + 0.7 * ce, ce).mean()
    assert torch.allclose(TR.distill_loss(lg, ext, t, 1.0, alpha=0.3, y=y), want, atol=1e-6)
    tp = t.pow(1 / T)
    tp = tp / tp.sum(-1, keepdim=True).clamp_min(1e-12)
    kdT = -(tp * torch.log_softmax(lg / T, -1).masked_fill(~ext, 0)).sum(-1).mean() * T * T
    assert torch.allclose(TR.distill_loss(lg, ext, t, T, t2=True), kdT, atol=1e-6)
    assert torch.allclose(TR.distill_loss(lg, ext, t, 1.0, t2=True), TR.distill_loss(lg, ext, t, 1.0))


def _fake(prefer):
    """Content teacher: puts 0.7 on option `prefer(state)`, regardless of position."""
    def t(state, q, order):
        k = prefer(state)
        return [0.7 if o == k else 0.3 / (len(order) - 1) for o in order]
    return t


def test_teacher_panel_requires_two_families():
    a, b = _fake(lambda s: "x"), _fake(lambda s: "y")
    with pytest.raises(ValueError, match="families"):
        teacher_panel([("qwen", a), ("qwen", b)], "s", {}, ["x", "y"])
    mean, dis, unk = teacher_panel([("qwen", a), ("mistral", b)], "s", {}, ["x", "y"], unknown_floor=0.05)
    assert np.allclose(mean, [0.5, 0.5]) and abs(dis - 0.2) < 1e-9 and unk == dis


def test_unknown_counterfactual_and_flip_filter():
    opts = ["yes", "no"]
    unk = "cannot be determined from the text"
    good = _fake(lambda s: "yes" if "sky is blue" in s else unk)
    pairs = unknown_counterfactual("the sky is blue.", {"q": "blue?"}, opts, "the grass is green.", good)
    assert len(pairs) == 2 and pairs[1][0] == "the grass is green."
    assert np.array_equal(pairs[1][2], [0, 0, 1]) and int(pairs[0][2].argmax()) == 0
    stubborn = _fake(lambda s: "yes")
    assert unknown_counterfactual("the sky is blue.", {}, opts, "x", stubborn) == []
    raw = unknown_counterfactual("a", {}, opts, "b")
    assert raw[0][2] is None and np.array_equal(raw[1][2], [0, 0, 1])
    assert label_flips(good, "the sky is blue.", {}, "nothing", {}, [*opts, unk])
    assert not label_flips(stubborn, "a", {}, "b", {}, opts)


def _batch(B=2, Q=2, K=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt_ids = torch.randint(3, 250, (B, Q, K, 3), generator=g)
    opt_mask = torch.ones(B, Q, K, dtype=torch.bool)
    opt_mask[1, 1, 6:] = False
    opt_ids[1, 1, 6:] = 0
    return {"ids": torch.randint(3, 250, (B, 10), generator=g), "mask": torch.ones(B, 10, dtype=torch.bool),
            "opt_ids": opt_ids, "opt_mask": opt_mask, "instr_ids": torch.randint(3, 250, (B, Q, 4), generator=g),
            "qtype": torch.tensor([[0, 0], [0, 2]]), "y": torch.tensor([[1, 8], [K, 5]])}


def test_label_contrastive_reference_and_type_isolation():
    b = _batch()
    g = torch.Generator().manual_seed(5)
    emb = torch.randn(2, 2, 9, 8, generator=g)
    qry = torch.randn(2, 2, 8, generator=g)
    got = L.label_contrastive(emb, b["y"], 0.5, query=qry, qtype=b["qtype"], opt_mask=b["opt_mask"])
    # loop reference: rows (0,0),(0,1) type 0 share 18 candidates (+ row (1,0) has unknown gold: skipped);
    # row (1,1) type 2 sees only its own 6 real options
    e = torch.nn.functional.normalize(emb, dim=-1)
    a = torch.nn.functional.normalize(qry, dim=-1)
    pool0 = torch.cat([e[0, 0], e[0, 1], e[1, 0]])
    ls = []
    for (bi, qi, gold, pool, off) in [(0, 0, 1, pool0, 0), (0, 1, 8, pool0, 9), (1, 1, 5, e[1, 1, :6], 0)]:
        s = pool @ a[bi, qi] / 0.5
        ls.append(-(torch.log_softmax(s, 0)[off + gold]))
    assert torch.allclose(got, torch.stack(ls).mean(), atol=1e-5)


def test_label_negatives_finetune_trains_deterministically():
    def run():
        torch.manual_seed(0)
        m = OpenDecisionModel(CFG)
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        return [TR.finetune_step(m, opt, _batch(), "ce_brier", label_neg=0.5) for _ in range(3)]
    a, b = run(), run()
    assert a == b and all(np.isfinite(a))
    torch.manual_seed(0)
    m = OpenDecisionModel(CFG)
    torch.manual_seed(0)
    m2 = OpenDecisionModel(CFG)
    o1, o2 = torch.optim.SGD(m.parameters(), lr=0.05), torch.optim.SGD(m2.parameters(), lr=0.05)
    assert TR.finetune_step(m, o1, _batch()) == TR.finetune_step(m2, o2, _batch(), label_neg=0.0)
