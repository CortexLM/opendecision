"""v1 model configs (design A3). Dense only, no chunk path."""
from dataclasses import replace

from .model import ModelConfig
from .tokenizer import MODEL_VOCAB, VOCAB

PRESETS: dict[str, ModelConfig] = {
    "od-tiny": ModelConfig(d=64, layers=2, heads=4, head_layers=1, max_len=64, vocab=VOCAB, chunk=0),
    "od-base": ModelConfig(d=768, layers=12, heads=12, head_layers=2, max_len=1024, vocab=MODEL_VOCAB, chunk=0),
    "od-large": ModelConfig(d=1024, layers=24, heads=16, head_layers=4, max_len=1024, vocab=MODEL_VOCAB, chunk=0),
}
# v2 = best measured on the R0-R5 proxy (d384x6, 3 seeds): trunc-normal init + emb LayerNorm (A) + block v2 (B).
# A: MLM eval loss 5.74 -> 3.87, Banking77 acc 0.40 -> 0.86. Gate / doc mask / mask rate: within noise, left off.
PRESETS |= {f"{k}-v2": replace(v, init_scheme="trunc002", emb_norm=True, block="v2")
            for k, v in list(PRESETS.items())}
STAGE_A_PREFIXES = ("tok.", "emb_norm.", "encoder.")
