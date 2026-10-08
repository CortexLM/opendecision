import numpy as np, torch
from opendecision import Decider, OpenDecisionModel, ModelConfig, DecisionRequest
from opendecision import losses as L, calibrate as C, train as TR, tokenizer as T
from opendecision.coherence import complement_gap, order_flip_rate

torch.manual_seed(0)
CFG = ModelConfig(d=64, layers=2, heads=2, head_layers=1, max_len=64)
REQ = {"state": "Billed twice, refund please", "questions": {
    "dept": {"type": "choice", "instructions": "dept?", "criteria": {"billing": "money", "tech": "bugs", "other": "else"}},
    "urg": {"type": "score", "instructions": "urgent?", "criteria": ["low", "mid", "high"]},
    "churn": {"type": "noul", "instructions": "cancel?"}}}


def mk(cfg=CFG):
    return Decider(OpenDecisionModel(cfg))


def batch_for(d, req, y):
    names, oid, om, ins, qt = d._pack(DecisionRequest.model_validate(req).questions)
    x, m = T.batch([req["state"]], 4096)
    return {"ids": x, "mask": m, "opt_ids": oid, "opt_mask": om, "instr_ids": ins, "qtype": qt, "y": torch.tensor([y])}


def test_json_probabilities_sum_to_one():
    r = mk().predict(REQ)
    for a in r["answers"].values():
        assert abs(sum(a["probabilities"].values()) + a["unknown"] - 1) < 1e-5
    assert r["usage"]["output_tokens"] == 0 and "choice" in r["answers"]["dept"] and "noul" in r["answers"]["churn"]


def test_empty_questions_validation_and_text_only_schema():
    assert mk().predict({"state": "x", "questions": {}})["answers"] == {}
    for bad in ({"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "x"}}}},
                {"state": "x", "questions": {}, "media": [{"kind": "image", "uri": "u"}]}):   # media rejected in v1
        try:
            mk().predict(bad); assert False
        except ValueError:
            pass


def test_option_order_equivariance():
    d = mk()
    q = lambda c: {"state": "s", "questions": {"q": {"type": "choice", "instructions": "i", "criteria": c}}}
    a = d.predict(q({"x": "1", "y": "2", "z": "3"}))["answers"]["q"]["probabilities"]
    b = d.predict(q({"z": "3", "x": "1", "y": "2"}))["answers"]["q"]["probabilities"]
    assert all(abs(a[k] - b[k]) < 1e-5 for k in a)


def test_extra_questions_do_not_change_other_answers():
    d = mk()
    one = {"state": REQ["state"], "questions": {"dept": REQ["questions"]["dept"]}}
    a = d.predict(one)["answers"]["dept"]["probabilities"]
    b = d.predict(REQ)["answers"]["dept"]["probabilities"]
    assert all(abs(a[k] - b[k]) < 1e-5 for k in a)           # same state, more questions, padding K differs


def test_scoring_rules_proper_minimum_at_truth():
    y = torch.tensor([1]); good = torch.tensor([[0., 10., 0.]]); bad = torch.tensor([[10., 0., 0.]])
    for f in (L.log_loss, L.brier, L.spherical, L.rps):
        assert f(good, y) < f(bad, y)


def test_finetune_learns_and_temperature_calibrates():
    d = mk(); m = d.model; opt = torch.optim.Adam(m.parameters(), 3e-3)
    b = batch_for(d, REQ, [0, 2, 1])
    first = TR.finetune_step(m, opt, b)
    for _ in range(150): last = TR.finetune_step(m, opt, b)
    assert last < first * 0.3
    logits = torch.randn(200, 4) * 4; y = torch.randint(0, 4, (200,))
    T_ = C.fit_temperature(logits, y)
    assert C.nll(logits / T_, y) <= C.nll(logits, y) + 1e-6 and 0.5 <= T_ <= 5.0


def test_bandit_estimator_unbiased_for_half_brier_grad():
    torch.manual_seed(1); K = 5; z = torch.randn(K, requires_grad=True); y = 2
    p = torch.softmax(z, 0)
    (-(1 - ((torch.eye(K)[y] - p) ** 2).sum()) / 2).backward()
    ref = -z.grad.clone()
    n = 40000; a = torch.multinomial(p.detach(), n, replacement=True)
    r = (a == y).float() - p.detach()[a]
    g = torch.zeros(K)
    for ai, ri in zip(a.tolist(), r.tolist()):
        gl = -p.detach().clone(); gl[ai] += 1; g += gl * ri
    assert torch.allclose(g / n, ref, atol=0.01)


def test_metrics_and_conformal():
    p = np.array([[.9, .1], [.2, .8], [.6, .4]]); y = np.array([0, 1, 1])
    assert 0 <= C.ece(p, y, bins=2) <= 1 and C.brier_score(p, y) > 0
    q = C.conformal_threshold(np.tile([.7, .3], (50, 1)), np.zeros(50, int), 0.1)
    assert abs(q - 0.3) < 1e-6


def test_coherence_battery_runs():
    d = mk()
    assert 0 <= complement_gap(d, ["a", "b"], "refund?", "no refund?") <= 1
    assert 0 <= order_flip_rate(d, "s", "i", {"x": "1", "y": "2", "z": "3"}) <= 1


def test_hierarchical_long_context_shapes_and_latent_budget():
    cfg = ModelConfig(d=64, layers=2, heads=2, head_layers=1, chunk=64, chunk_latents=4, global_layers=1, keep_tokens=False)
    m = OpenDecisionModel(cfg).eval()
    x, mask = T.batch(["a" * 1000], 4096)
    with torch.no_grad():
        h, hm = m.encode_state(x, mask)
    assert h.shape[1] == 16 * 4 and hm.all()              # 1001 tokens -> 64 latents
    r = Decider(m, max_tokens=4096).predict({"state": "b" * 3000, "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert r["usage"]["input_tokens"] == 3001 and not r["usage"]["truncated"]


def test_keep_tokens_adds_full_memory_and_batch_padding_is_masked():
    cfg = ModelConfig(d=64, layers=2, heads=2, head_layers=1, chunk=64, chunk_latents=4, global_layers=1, keep_tokens=True)
    m = OpenDecisionModel(cfg).eval()
    x, mask = T.batch(["a" * 300, "b" * 100], 4096)
    with torch.no_grad():
        h, hm = m.encode_state(x, mask)
        h1, hm1 = m.encode_state(*T.batch(["b" * 100], 4096))
    assert hm.shape[1] > 6 * 4 and hm[1].sum() < hm[0].sum()
    # the short doc alone must give the same latent block as inside a padded batch
    nl = hm1.shape[1] // 2 if False else 2 * 4          # 101 tokens -> 2 chunks -> 8 latents
    assert torch.allclose(h[1, :nl], h1[0, :nl], atol=1e-4)


def test_rope_generalises_beyond_trained_length_without_error():
    m = OpenDecisionModel(ModelConfig(d=64, layers=2, heads=2, head_layers=1, max_len=64)).eval()
    x, mask = T.batch(["z" * 2000], 4096)                 # flat path, 30x the nominal max_len: no position table to overflow
    with torch.no_grad():
        h, _ = m.encode_state(x, mask)
    assert h.shape[1] == 2001 and torch.isfinite(h).all()


def test_teacher_poll_and_cascade():
    from opendecision.teacher import cyclic_average, poll
    from opendecision.cascade import Cascade
    biased = lambda s, q, order: [0.6 if i == 0 else 0.4 / (len(order) - 1) for i in range(len(order))]
    p = cyclic_average(biased, "s", {}, ["a", "b", "c"])
    assert np.allclose(p, 1 / 3, atol=1e-9)
    mean, dis, unk = poll([biased, biased], "s", {}, ["a", "b", "c"])
    assert abs(mean.sum() - 1) < 1e-9 and dis < 1e-9
    d = mk(); c = Cascade(d, d, threshold=1.01)
    c.predict(REQ); assert c.escalation_rate == 1.0


def test_long_instruction_options_stay_distinct():
    d = mk()
    q = DecisionRequest.model_validate({"state": "s", "questions": {"q": {"type": "choice", "instructions": "Which department should handle this customer message right now?", "criteria": {"billing": "money", "technical": "bugs"}}}}).questions
    _, ids, om, ins, _ = d._pack(q)
    assert not torch.equal(ids[0, 0, 0], ids[0, 0, 1])


def test_truncation_reported():
    r = mk().predict({"state": "x" * 500, "questions": {"q": {"type": "noul", "instructions": "x"}}})
    assert r["usage"]["truncated"] is True and r["usage"]["input_tokens"] == 64


def test_unknown_mass_learnable_for_out_of_scope():
    d = mk(); m = d.model; opt = torch.optim.Adam(m.parameters(), 3e-3)
    q = {"state": "weather is nice", "questions": {"q": {"type": "choice", "instructions": "dept?", "criteria": {"billing": "money", "tech": "bugs"}}}}
    b = batch_for(d, q, [2])                              # index K == unknown slot
    for _ in range(120): TR.finetune_step(m, opt, b)
    assert d.predict(q)["answers"]["q"]["unknown"] > 0.9


def test_mlm_step_and_real_bpe_tokenizer():
    m = OpenDecisionModel(ModelConfig(d=64, layers=2, heads=2, head_layers=1)); opt = torch.optim.Adam(m.parameters(), 1e-3)
    x, mask = T.batch(["hello world " * 8] * 4, 64)
    assert TR.mlm_step(m, opt, x, mask) > 0
    import pytest
    tk = pytest.importorskip("tokenizers")
    try:
        t = T.HFTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    except Exception:
        pytest.skip("tokenizer.json not available offline")
    n_bpe, n_byte = len(t.encode("The customer asked for a refund after being charged twice.", 999)), len(T.ByteTokenizer().encode("The customer asked for a refund after being charged twice.", 999))
    assert n_bpe * 3 < n_byte * 1.2 and t.vocab > 50000
