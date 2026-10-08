# Long context (32k+) without dense attention

Why dense 32k looked impossible: attention FLOPs per token ~ `4 x d x L x layers`. For d=1024, 24 layers, L=32k that is 3.2 GFLOP/token on top of 0.6 GFLOP of matmuls, i.e. attention dominates (DERIVED).

OpenDecision is parallel along three axes already (tokens, questions Q, options K). For long inputs it adds a fourth: **chunk parallelism** (`ModelConfig(chunk=512, chunk_latents=16, global_layers=2)`):
1. The document is cut into chunks; **all chunks are encoded in one batched pass** (attention cost O(N x chunk), embarrassingly parallel, shardable across GPUs).
2. Each chunk is pooled to `chunk_latents` latents (32x compression at 512 -> 16).
3. A few global layers attend over latents only (2k latents for 32k tokens), then the question head cross-attends the latents.

DERIVED cost per 1M tokens (30% MFU, $3.5/h): short-context large $0.0026; 32k with global attention every 3rd layer $0.0065 but ~0.2 s for one request (latency problem); hierarchical 32k ~$0.003 and ~85 ms. Accuracy of compressed latents on decisions that need one specific span is **unproven** (BASED/linear-attention recall results warn about lossy state): evaluate with needle-in-document decisions before shipping. Alternatives to ablate: ModernBERT-style alternating local(128)/global layers, FlashAttention-3 (740 TFLOPs on H100; encoder varlen support unverified), sequence parallelism (Ring/Ulysses) for single huge requests.
