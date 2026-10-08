"""v1 model configs (design A3). Dense only, no chunk path."""
from .model import ModelConfig
from .tokenizer import MODEL_VOCAB, VOCAB

PRESETS: dict[str, ModelConfig] = {
    "od-tiny": ModelConfig(d=64, layers=2, heads=4, head_layers=1, max_len=64, vocab=VOCAB, chunk=0),
    "od-base": ModelConfig(d=768, layers=12, heads=12, head_layers=2, max_len=1024, vocab=MODEL_VOCAB, chunk=0),
    "od-large": ModelConfig(d=1024, layers=24, heads=16, head_layers=4, max_len=1024, vocab=MODEL_VOCAB, chunk=0),
}
STAGE_A_PREFIXES = ("tok.", "encoder.")
