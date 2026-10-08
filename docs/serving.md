# Serving plan (targets to measure, not results)

- Python FastAPI (`serve.py`) is a reference surface only; independent H100 NVL numbers for the Laya stack (175 decisions/s at p99<=130 ms) imply ~10-30k tok/s, two orders below target.
- Production path: Rust/C++ router with Rust tokenizers, token-budget dynamic batching (TEI-style), padding-free varlen attention, bucketed CUDA graphs per (batch, seq), TensorRT/ORT-TRT engine, one replica per GPU (no MIG), 2-4 CUDA streams per GPU, load balance by tokens.
- Latency budget (p99 200 ms, server-side, media already uploaded): network+parse 2-5, tokenize 1-3, batch window <=10, forward 10-30, heads+JSON <1.
- Media: decode on GPU (nvJPEG/NVDEC/DALI); CPU decode is the likely bottleneck, unmeasured.
- Measure first: `python benchmarks/bench_throughput.py --device cuda --d 1024 --layers 24`, commit the JSON under `research/results/`.
