import pathlib

import numpy as np
import torch
from test_core import CFG, REQ, batch_for, mk

from opendecision import OpenDecisionModel
from opendecision import train as TR
from opendecision.masking import rl_gumbel


def _run(seed, step):
    torch.manual_seed(0)
    m = OpenDecisionModel(CFG); d = mk(); d.model = m
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    b = batch_for(d, REQ, [0, 1, 2])
    seen = []
    def outcome(a):
        seen.append(a.clone()); return a == 0
    TR.rl_step(m, opt, b, outcome, samples=8, seed=seed, step=step)
    return seen[0], [p.detach().clone() for p in m.parameters()]

def test_rl_step_replay_identical():
    a1, p1 = _run(7, 3); a2, p2 = _run(7, 3)
    assert torch.equal(a1, a2) and all(torch.equal(x, y) for x, y in zip(p1, p2))

def test_rl_step_differs_by_step():
    assert not torch.equal(_run(7, 3)[0], _run(7, 4)[0])

def test_gumbel_max_matches_softmax():
    logits = np.array([0.5, -1.0, 1.5, 0.0]); p = np.exp(logits) / np.exp(logits).sum()
    n = 20000
    rows = np.arange(n, dtype=np.uint32)[:, None]
    a = (logits + rl_gumbel(rows, 1, 0, 1, 1, 4)[:, 0, 0]).argmax(-1)
    f = np.bincount(a, minlength=4) / n
    assert np.all(np.abs(f - p) <= 3 * np.sqrt(p * (1 - p) / n))

def test_no_multinomial_in_src():
    src = pathlib.Path(TR.__file__).parent
    assert not any("multinomial" in f.read_text() for f in src.glob("*.py"))
