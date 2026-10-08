"""Measure the 300k tok/s/GPU target. Usage: python benchmarks/bench_throughput.py --device cuda --d 1024 --layers 24
Reports tokens/s for encode_state on packed-length batches + p50/p99 end-to-end Decider.predict latency.
Targets are CLAIMS TO MEASURE, not results. Run on an H200 and commit the JSON."""
import argparse, json, time, torch
from opendecision import ModelConfig, OpenDecisionModel, Decider

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu"); ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--seq", type=int, default=512); ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()
cfg = ModelConfig(d=a.d, layers=a.layers, heads=max(a.d // 64, 1), max_len=a.seq)
m = OpenDecisionModel(cfg).to(a.device).eval()
dt = torch.bfloat16 if a.device == "cuda" else torch.float32
ids = torch.randint(3, 259, (a.batch, a.seq), device=a.device); mask = torch.ones_like(ids, dtype=torch.bool)
with torch.no_grad(), torch.autocast(a.device, dtype=dt, enabled=a.device == "cuda"):
    for _ in range(3): m.encode_state(ids, mask)
    if a.device == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(a.iters): m.encode_state(ids, mask)
    if a.device == "cuda": torch.cuda.synchronize()
    tps = a.iters * a.batch * a.seq / (time.perf_counter() - t)
d = Decider(m, a.device); lat = []
req = {"state": "x" * 400, "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": "a", "b": "b"}}}}
for _ in range(50):
    t = time.perf_counter(); d.predict(req); lat.append((time.perf_counter() - t) * 1e3)
lat.sort()
print(json.dumps({"device": a.device, "params_M": sum(p.numel() for p in m.parameters()) / 1e6, "tok_per_s": tps,
                  "p50_ms": lat[25], "p99_ms": lat[-1], "target_tok_per_s": 300000, "target_p99_ms": 200}, indent=1))
