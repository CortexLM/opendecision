# Unit economics: profitable at $0.030 per 1M input tokens

300k tok/s was dropped as a target. The target is **margin at $0.030/M** with JSON in <200 ms.

Formula: `$/1M = $/h / (tok/s x utilization x 3600 / 1e6)`, `tok/s = MFU x peak / (2 x active_params x 1.15)`. Run `python benchmarks/cost_model.py --active-m 300 --usd-h 3.59 --util 0.5`.

| Active params | H200 @3.59/h, MFU35%, util 50% | gross margin at $0.030 |
|---|---|---|
| 100M | $0.0013 | 96% |
| 300M | $0.0040 | 87% |
| 1B | $0.0133 | 56% |
| 3B MoE-active | $0.0398 | negative |

**All DERIVED from assumptions** (GPU prices dated Sept 2026: H200 community $3.59 / secure $4.59 on RunPod, hyperscalers $6-12; peak TFLOPS from spec; MFU 35% optimistic for small ragged batches). Nothing is measured on a target GPU. Rules that follow:
- **Ceiling:** ~1.1B active params on H200 at 50% utilization keeps a 50% margin; ~1.5B on H100; FP8 may double that (unverified for encoders; 5090 training measured ~1.2x, not 2x).
- **H200 is not cheaper than H100** for a compute-bound encoder (same peak FLOPS, higher price). L40S/5090/H100 community are the cost leaders.
- **Utilization, not FLOPs, decides profit**: batch-1 latency serving idles the GPU. Dynamic token-budget batching with a 5-10 ms window is mandatory.
- **Cascade**: small model on all requests, escalate uncertain ones (`opendecision.cascade`). Intent-first-stage -> big model matched accuracy at 0.43 of the cost (arXiv 2610.00346). Escalate on held-out `answer_confidence`, never on Laya-style `act_probability` (AUROC 0.30).
- **Multimodal billing**: one token rate, per-modality token caps. Estimated GPU cost: image (SigLIP2-B, 196 tok) ~$9e-8, so400m 256 tok ~$5e-7, audio ~$2e-8/s (Whisper-small), 16-frame V-JEPA2-L clip ~$3.3e-6. Charge images >=256 token-equivalents, audio >=25/s, video 1 fps x 256. CPU decode (JPEG/video) is unpriced and likely the real cost.
- Competition: commodity embeddings cost $0.008-0.02/M, so $0.030 is defensible only against decision engines, not embeddings.
