import inspect, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from opendecision import OpenDecisionModel, model as M, train as TR, losses as L
from opendecision.masking import mask_positions, n_mask_for, row_keys
from opendecision.presets import PRESETS
from opendecision.records import RecordShape, pack_record, unpack_records, qmax_for

torch.manual_seed(0)
GOLD_POS = [[6, 9, 11, 12, 14], [4, 7, 11, 12, 15]]
GOLD_KEYS = [5743788249220662041, 5761332486301057853]
TINY = PRESETS["od-tiny"]

def test_mask_golden_and_fixed_count():
    rows = np.arange(32, dtype=np.uint32).reshape(2, 16) + 5
    p = mask_positions(rows, 7, 5)
    assert p.shape == (2, 5) and (np.diff(p, axis=1) > 0).all()
    assert np.array_equal(p, mask_positions(rows.copy(), 7, 5))            # pure function of (seed, row)
    assert not np.array_equal(p, mask_positions(rows, 8, 5))
    assert p.tolist() == GOLD_POS and row_keys(rows, 7).tolist() == GOLD_KEYS
    assert n_mask_for(64, 0.3) == 19 and n_mask_for(1, 0.3) == 1

def test_param_names_match_state_dict_and_a3_counts():
    for name, full, stage_a, n in (("od-tiny", 187_140, 116_288, 58), ("od-base", 143_241_220, 123_753_984, 214),
                                   ("od-large", 422_072_324, 353_861_632, 414)):
        cfg = PRESETS[name]
        with torch.device("meta"):
            sd = OpenDecisionModel(cfg).state_dict()
        names = M.param_names(cfg)
        assert {k: tuple(v.shape) for k, v in sd.items()} == names
        assert M.od_param_count(cfg) == full and len(names) == n
        assert sum(math.prod(s) for k, s in names.items() if k.startswith(("tok.", "encoder."))) == stage_a

def test_no_multihead_attention_and_head_attention_plain():
    m = OpenDecisionModel(TINY)
    assert not any(isinstance(x, nn.MultiheadAttention) for x in m.modules())
    a = m.head[0].opt_attn
    assert isinstance(a, M.Attention) and [n for n, _ in a.named_children()] == ["q", "kv", "o"]
    assert all(l.bias is None for l in a.children())

def _mk(K=3, Kq=(2, 3), B=1):
    Q, T = len(Kq), 4
    opt_ids = torch.randint(3, 20, (B, Q, K, T))
    om = torch.zeros(B, Q, K, dtype=torch.bool)
    for q, k in enumerate(Kq): om[:, q, :k] = True
    return opt_ids, om, torch.randint(3, 20, (B, Q, 5)), torch.tensor([[0, 1]] * B)

def test_distill_loss_with_kmax_greater_than_k():
    logits = torch.randn(1, 2, 4); ext = torch.tensor([[[1, 1, 0, 1], [1, 1, 1, 1]]], dtype=torch.bool)
    logits = logits.masked_fill(~ext, float("-inf"))
    teacher = torch.tensor([[[.5, .3, 0, .2], [.1, .2, .3, .4]]])
    ref = -(teacher * torch.log_softmax(logits, -1).nan_to_num(0).masked_fill(~ext, 0)).sum(-1).mean()
    got = TR.distill_loss(logits, ext, teacher)
    assert torch.isfinite(got) and torch.allclose(got, ref)

def test_pack_record_unknown_gold_and_decision_loss():
    s = RecordShape(state_len=6, n_questions=2, n_options=4, opt_len=3, instr_len=3)
    ex = {"state": [3, 4], "instr": [[5], [6]], "opts": [[[7], [8]], [[9], [10], [11]]], "qtype": [0, 0],
          "y": [None, 2], "teacher": [[.5, .25, .25], [.1, .2, .3, .4]]}
    qm = qmax_for(50432); assert qm == 32767
    r = pack_record(ex, s, qm); assert r.dtype == np.uint16 and r.size == s.length
    u = unpack_records(torch.from_numpy(r.astype(np.int64))[None], s, qm)
    assert u["y"].tolist() == [[4, 2]]                       # unknown == n_options, whatever K_q
    assert u["opt_mask"].tolist() == [[[True, True, False, False], [True, True, True, False]]]
    assert abs(u["teacher"][0, 0, 4].item() - .25) < 1e-4 and u["teacher"][0, 0, 2].item() == 0
    m = OpenDecisionModel(TINY).eval()
    with torch.no_grad():
        logits, ext = m(u["ids"], u["mask"], u["opt_ids"], u["opt_mask"], u["instr_ids"], u["qtype"])
        loss = TR.decision_loss(logits, ext, u["y"], u["qtype"])
    assert ext.shape == (1, 2, 5) and ext[0, 0].tolist() == [True, True, False, False, True]
    ref = -torch.log_softmax(logits, -1)[0, 0, 4]
    ref2 = -torch.log_softmax(logits, -1)[0, 1, 2]
    assert torch.allclose(loss, (ref + ref2) / 2, atol=1e-6)

def test_decision_loss_equals_compaction():
    torch.manual_seed(1)
    logits = torch.randn(3, 4, 6); y = torch.randint(0, 6, (3, 4)); qt = torch.randint(0, 3, (3, 4))
    for rule in ("log", "brier", "spherical", "rps"):
        sel = qt == 1
        base = TR.RULES[rule](logits[~sel], y[~sel]) if (~sel).any() else 0
        ordl = L.rps(logits[sel], y[sel]) if sel.any() else 0
        assert torch.allclose(TR.decision_loss(logits, None, y, qt, rule, 0.7), base + 0.7 * ordl, atol=1e-6)

def test_mask_none_equals_all_true_mask():
    m = OpenDecisionModel(TINY).eval()
    ids = torch.randint(3, 200, (2, 16))
    with torch.no_grad():
        assert torch.allclose(m.encode_tokens(ids), m.encode_tokens(ids, torch.ones_like(ids, dtype=torch.bool)), atol=1e-6)
        a = m.head[0].cross
        x, kv = torch.randn(2, 3, 64), torch.randn(2, 5, 64)
        assert torch.allclose(a(x, kv, None), a(x, kv, torch.ones(2, 5, dtype=torch.bool)), atol=1e-6)

def test_onehot_type_emb_and_temperature_match_lookup():
    m = OpenDecisionModel(TINY)
    with torch.no_grad():
        for k in M.QTYPES: m.log_temp[k].copy_(torch.randn(()))
    qt = torch.tensor([[0, 1, 2, 1]])
    oh = F.one_hot(qt, 3)
    e1 = oh.float() @ m.type_emb.weight; t1 = torch.exp(oh.float() @ torch.stack([m.log_temp[k] for k in M.QTYPES]))
    e1.pow(2).sum().add(t1.sum()).backward(); g1 = [m.type_emb.weight.grad.clone()] + [m.log_temp[k].grad.clone() for k in M.QTYPES]
    m.zero_grad()
    e2 = m.type_emb.weight[qt]; t2 = torch.stack([torch.exp(m.log_temp[k]) for k in M.QTYPES])[qt]
    e2.pow(2).sum().add(t2.sum()).backward(); g2 = [m.type_emb.weight.grad] + [m.log_temp[k].grad for k in M.QTYPES]
    assert torch.allclose(e1, e2) and torch.allclose(t1, t2)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g1, g2))
    # decide itself uses the dense path: no index-gather on type_emb
    assert "type_emb(" not in inspect.getsource(M.OpenDecisionModel.decide)

def test_mlm_loss_default_gather_and_custom_gather():
    m = OpenDecisionModel(TINY)
    ids = torch.randint(3, 200, (2, 16)); pos = torch.from_numpy(mask_positions(ids.numpy(), 1, 5))
    a = TR.mlm_loss(m, ids, pos)
    b = TR.mlm_loss(m, ids, pos, gather=lambda h, rows: torch.index_select(h, 0, rows))
    assert torch.allclose(a, b) and a.dtype == torch.float32

def test_rope_table_and_embed_fn():
    c, s = M.rope_table(8, 16, 10000.0)
    assert c.dtype == torch.float32 and c.shape == (8, 8) and torch.allclose(c[0], torch.ones(8))
    calls = []
    m = OpenDecisionModel(TINY, embed_fn=lambda w, i: (calls.append(1), F.embedding(i, w))[1])
    m.encode_tokens(torch.randint(3, 20, (1, 4)))
    assert calls and "tok.weight" in m.state_dict()
