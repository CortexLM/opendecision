import torch
from opendecision import Decider, OpenDecisionModel, ModelConfig, DecisionRequest
from opendecision import train as TR, tokenizer as T
m = OpenDecisionModel(ModelConfig(d=64, layers=2, heads=2, head_layers=1, max_len=64)); opt = torch.optim.Adam(m.parameters(), 3e-3)
req = {"state": "Refund please", "questions": {"q": {"type": "choice", "instructions": "dept?", "criteria": {"billing": "money", "tech": "bugs"}}}}
d = Decider(m); names, oid, om, ins, qt = d._pack(DecisionRequest.model_validate(req).questions); x, mk = T.batch([req["state"]], 64)
b = {"ids": x, "mask": mk, "opt_ids": oid, "opt_mask": om, "instr_ids": ins, "qtype": qt, "y": torch.tensor([[0]])}
for i in range(60): l = TR.finetune_step(m, opt, b)
print("loss", l, d.predict(req)["answers"]["q"]["probabilities"])
