"""Tokenizers. ids 0/1/2 are reserved (PAD/MASK/CLS); real tokens are shifted by OFFSET.

`ByteTokenizer` is dependency-free (tests/demos only: ~4x more tokens than BPE, which breaks the cost
model, see docs/economics.md). `HFTokenizer` wraps any `tokenizers` tokenizer.json, e.g. ModernBERT's
50k BPE (`HFTokenizer.from_pretrained("answerdotai/ModernBERT-base")`, extra: opendecision[tok])."""
from __future__ import annotations
import torch

PAD, MASK, CLS = 0, 1, 2
OFFSET = 3


class ByteTokenizer:
    name = "byte"
    vocab = 256 + OFFSET

    def encode(self, text: str, max_len: int) -> list[int]:
        return [CLS] + [b + OFFSET for b in text.encode("utf-8")][: max_len - 1]


class HFTokenizer:
    def __init__(self, tk, name="hf"):
        self.tk, self.name = tk, name
        self.vocab = tk.get_vocab_size() + OFFSET

    @classmethod
    def from_file(cls, path: str):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path), name=path)

    @classmethod
    def from_pretrained(cls, repo: str):
        from huggingface_hub import hf_hub_download
        return cls.from_file(hf_hub_download(repo, "tokenizer.json"))

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = self.tk.encode(text, add_special_tokens=False).ids[: max_len - 1]
        return [CLS] + [i + OFFSET for i in ids]


DEFAULT = ByteTokenizer()
VOCAB = DEFAULT.vocab


def encode(text: str, max_len: int, tok=None) -> list[int]:
    return (tok or DEFAULT).encode(text, max_len)


def batch(texts: list[str], max_len: int, tok=None):
    ids = [encode(t, max_len, tok) for t in texts]
    L = max(len(i) for i in ids)
    x = torch.zeros(len(ids), L, dtype=torch.long)
    for i, s in enumerate(ids):
        x[i, : len(s)] = torch.tensor(s)
    return x, x != PAD
