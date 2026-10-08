"""Request -> batched tensors -> one forward pass -> JSON with probabilities. No decoding, no generation."""
from __future__ import annotations
import json, math
import torch
from .model import OpenDecisionModel, ModelConfig, QTYPES
from .schema import DecisionRequest
from . import tokenizer as T


class Decider:
    def __init__(self, model: OpenDecisionModel | None = None, tokenizer=None, device="cpu", opt_len=32,
                 instr_len=64, version="opendecision-0.1", max_tokens: int = 0):
        self.model = (model or OpenDecisionModel(ModelConfig())).eval().to(device)
        self.tok = tokenizer or T.DEFAULT
        if self.tok.vocab > self.model.cfg.vocab:
            raise ValueError(f"tokenizer vocab {self.tok.vocab} > model vocab {self.model.cfg.vocab}")
        self.device, self.opt_len, self.instr_len, self.version = device, opt_len, instr_len, version
        self.max_tokens = max_tokens or self.model.cfg.max_len

    def _pack(self, qs):
        names = list(qs)
        Kmax = max(len(qs[n].options()) for n in names)
        ids = torch.zeros(1, len(names), Kmax, self.opt_len, dtype=torch.long)
        om = torch.zeros(1, len(names), Kmax, dtype=torch.bool)
        qt = torch.zeros(1, len(names), dtype=torch.long)
        ins = torch.zeros(1, len(names), self.instr_len, dtype=torch.long)
        for i, n in enumerate(names):
            q = qs[n]
            qt[0, i] = QTYPES[q.type]
            e = self.tok.encode(q.instructions, self.instr_len)
            ins[0, i, : len(e)] = torch.tensor(e)
            desc = q.criteria if q.type == "choice" else None
            for k, o in enumerate(q.options()):
                e = self.tok.encode(o + (f": {desc[o]}" if desc else ""), self.opt_len)
                ids[0, i, k, : len(e)] = torch.tensor(e)
                om[0, i, k] = True
        return names, ids.to(self.device), om.to(self.device), ins.to(self.device), qt.to(self.device)

    @torch.no_grad()
    def predict(self, request: dict | DecisionRequest) -> dict:
        req = request if isinstance(request, DecisionRequest) else DecisionRequest.model_validate(request)
        state = req.state if isinstance(req.state, str) else json.dumps(req.state, ensure_ascii=False)
        if not req.questions:
            return {"model": self.version, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0, "truncated": False}}
        ids = self.tok.encode(state, 10**9)
        truncated = len(ids) > self.max_tokens
        ids = ids[: self.max_tokens]
        x = torch.tensor([ids], device=self.device)
        names, oid, om, ins, qt = self._pack(req.questions)
        probs = torch.softmax(self.model(x, x != T.PAD, oid, om, ins, qt)[0][0], -1).cpu()
        out = {}
        for i, n in enumerate(names):
            q, p = req.questions[n], probs[i]
            opts = q.options()
            pv = p[: len(opts)].tolist()
            unk = float(p[-1])                       # learned 'none of the above' mass (last slot)
            top = max(range(len(pv)), key=pv.__getitem__)
            s = sum(pv) or 1.0
            ent = -sum((v / s) * math.log(max(v / s, 1e-12)) for v in pv)
            conf = 1 - ent / max(math.log(len(pv)), 1e-12)
            a = {"type": q.type, "probabilities": dict(zip(opts, pv)), "unknown": unk, "confidence": conf, "answer_confidence": pv[top]}
            if q.type == "choice":
                a["choice"] = opts[top]
            elif q.type == "score":
                a["score"] = sum(j * v for j, v in enumerate(pv)) / s
                a["legend"] = {str(j): o for j, o in enumerate(opts)}
            else:
                a["noul"] = pv[1]
            out[n] = a
        return {"model": self.version, "answers": out, "usage": {"input_tokens": len(ids), "output_tokens": 0, "truncated": truncated}}
