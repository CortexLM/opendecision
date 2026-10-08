"""Two-stage cascade: cheap model on everything, escalate uncertain items (DeeBERT/FrugalGPT idea).
Escalation uses a held-out threshold on answer_confidence, NOT Laya's act_probability (AUROC 0.30, issue #185).
Intent-first-stage -> big model matched accuracy at 0.43 of the cost (2610.00346)."""
from __future__ import annotations


class Cascade:
    def __init__(self, small, large, threshold: float = 0.8):
        self.small, self.large, self.t = small, large, threshold
        self.escalated = self.total = 0

    def predict(self, req):
        r = self.small.predict(req); self.total += 1
        if any(a["answer_confidence"] < self.t for a in r["answers"].values()):
            self.escalated += 1
            r = self.large.predict(req); r["routing"] = {"stage": "large"}
        else:
            r["routing"] = {"stage": "small"}
        return r

    @property
    def escalation_rate(self):
        return self.escalated / max(self.total, 1)
