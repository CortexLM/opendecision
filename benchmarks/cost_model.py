"""$ per 1M tokens = $/h / (tok/s * util * 3600 / 1e6); tok/s = MFU*peak/(2*active_params*1.15).
ALL INPUTS ASSUMED until measured: run bench_throughput.py on the target GPU and pass --tok-s to override."""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--active-m", type=float, default=300); ap.add_argument("--usd-h", type=float, default=3.59)
ap.add_argument("--peak-tflops", type=float, default=989); ap.add_argument("--mfu", type=float, default=0.35)
ap.add_argument("--util", type=float, default=0.5); ap.add_argument("--tok-s", type=float, default=0)
ap.add_argument("--price", type=float, default=0.030)
a = ap.parse_args()
tps = a.tok_s or a.mfu * a.peak_tflops * 1e12 / (2 * a.active_m * 1e6 * 1.15)
cost = a.usd_h / (tps * a.util * 3600 / 1e6)
print(f"tok/s {tps:,.0f}  cost ${cost:.4f}/1M  price ${a.price}  gross margin {(1 - cost / a.price) * 100:.0f}%")
