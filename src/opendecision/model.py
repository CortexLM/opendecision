"""OpenDecisionModel (v1, text only): encoder-only, non-autoregressive.

state tokens -> RoPE fusion encoder (run ONCE per state; long inputs: chunk-parallel + global layers over
chunk latents, optional full-token memory) -> late-interaction head: every option/instruction query
cross-attends the cached state in one batched call (no per-question copy of the state) -> logits.

Options are order-equivariant (no positions on queries), see each other only through `opt_attn`, never
through the state, so adding questions never changes the state encoding. Every question has one extra
learned `unknown` slot (none-of-the-above) as the LAST logit. Option text, instruction text and state
share one encoder and one embedding table."""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import VOCAB, PAD

QTYPES = {"choice": 0, "score": 1, "noul": 2}


@dataclass
class ModelConfig:
    d: int = 256
    layers: int = 4
    heads: int = 4
    head_layers: int = 2
    max_len: int = 512           # default input limit for the flat path (RoPE has no hard cap)
    vocab: int = VOCAB
    rope_theta: float = 10_000.0
    chunk: int = 0               # >0: chunk-parallel long context (chunk size, e.g. 512)
    chunk_latents: int = 16      # latents kept per chunk (index for the global layers)
    global_layers: int = 2       # layers over the chunk latents
    global_theta: float = 160_000.0  # ModernBERT raises theta on global layers for long context
    keep_tokens: bool = False    # heads also see full token states (+chunk id). UNPROVEN: in the toy needle test it stayed at chance (0.5) where latents-only reached 1.0 in 500 steps


def rope(x, pos, theta):
    """x [B,H,N,hd]; pos [N]. Rotary embedding (rotate-half)."""
    half = x.size(-1) // 2
    inv = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = pos.to(torch.float32)[:, None] * inv
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def sincos(idx, d):
    f = 10_000.0 ** (-torch.arange(0, d, 2, device=idx.device, dtype=torch.float32) / d)
    a = idx.to(torch.float32)[:, None] * f
    return torch.cat([a.sin(), a.cos()], -1)[:, :d]


class Block(nn.Module):
    def __init__(self, d, heads, theta):
        super().__init__()
        self.h, self.theta = heads, theta
        hd = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.qn, self.kn = nn.LayerNorm(hd), nn.LayerNorm(hd)   # QK-norm: stability at scale
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask, pos):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(self.qn(q), pos, self.theta), rope(self.kn(k), pos, self.theta)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        x = x + self.o(a.transpose(1, 2).reshape(B, N, d))
        return x + self.ff(self.n2(x))


class Encoder(nn.Module):
    def __init__(self, d, heads, layers, theta):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, theta) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask, pos):
        for b in self.blocks:
            x = b(x, mask, pos)
        return self.norm(x)


class HeadLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.opt_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, qmask, h, hmask):
        # q [B,Q,K,d], qmask [B,Q,K]; h [B,N,d] (NOT repeated per question), hmask [B,N]
        B, Q, K, d = q.shape
        x = self.n1(q).reshape(B * Q, K, d)
        q = q + self.opt_attn(x, x, x, key_padding_mask=~qmask.reshape(B * Q, K))[0].view(B, Q, K, d)
        x = self.n2(q).reshape(B, Q * K, d)
        q = q + self.cross(x, h, h, key_padding_mask=~hmask)[0].view(B, Q, K, d)
        return q + self.ff(self.n3(q))


class OpenDecisionModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        c = self.cfg = cfg or ModelConfig()
        self.tok = nn.Embedding(c.vocab, c.d)
        self.encoder = Encoder(c.d, c.heads, c.layers, c.rope_theta)
        self.type_emb = nn.Embedding(3, c.d)
        self.unknown = nn.Parameter(torch.randn(c.d) * 0.02)
        self.head = nn.ModuleList([HeadLayer(c.d, c.heads) for _ in range(c.head_layers)])
        self.scorer = nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, c.d), nn.GELU(), nn.Linear(c.d, 1))
        self.log_temp = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in QTYPES})
        if c.chunk:
            self.chunk_q = nn.Parameter(torch.randn(c.chunk_latents, c.d) * 0.02)
            self.chunk_pool = nn.MultiheadAttention(c.d, c.heads, batch_first=True)
            self.global_enc = Encoder(c.d, c.heads, c.global_layers, c.global_theta)

    # ---- stage 1: encode state once ----------------------------------------------------
    def _chunks(self, ids, mask):
        c = self.cfg
        B, N = ids.shape
        pad = (-N) % c.chunk
        ids, mask = F.pad(ids, (0, pad)), F.pad(mask, (0, pad))
        nc = ids.size(1) // c.chunk
        x = self.tok(ids).reshape(B * nc, c.chunk, -1)
        m = mask.reshape(B * nc, c.chunk)
        m = m | (~m.any(1, keepdim=True) & (torch.arange(c.chunk, device=ids.device) == 0))  # empty chunk stays finite
        h = self.encoder(x, m, torch.arange(c.chunk, device=ids.device))
        return h, m, mask, B, nc

    def encode_tokens(self, ids, mask):
        """Per-token states [B,N,d] (flat or chunk-local); used by MLM pretraining."""
        if self.cfg.chunk and ids.size(1) > self.cfg.chunk:
            h, _, _, B, nc = self._chunks(ids, mask)
            return h.reshape(B, nc * self.cfg.chunk, -1)[:, : ids.size(1)]
        return self.encoder(self.tok(ids), mask, torch.arange(ids.size(1), device=ids.device))

    def encode_state(self, ids, mask):
        c = self.cfg
        if not (c.chunk and ids.size(1) > c.chunk):
            return self.encode_tokens(ids, mask), mask
        h, m, mask, B, nc = self._chunks(ids, mask)
        q = self.chunk_q.unsqueeze(0).expand(B * nc, -1, -1)
        lat, _ = self.chunk_pool(q, h, h, key_padding_mask=~m)
        lat = lat.reshape(B, nc * c.chunk_latents, -1)
        lm = mask.reshape(B, nc, c.chunk).any(2).unsqueeze(-1).expand(-1, -1, c.chunk_latents).reshape(B, -1)
        g = self.global_enc(lat, lm, torch.arange(nc * c.chunk_latents, device=ids.device))
        if not c.keep_tokens:
            return g, lm
        cid = sincos(torch.arange(nc, device=ids.device), h.size(-1)).repeat_interleave(c.chunk, 0)
        mem = h.reshape(B, nc * c.chunk, -1) + cid.unsqueeze(0)
        return torch.cat([g, mem], 1), torch.cat([lm, mask], 1)

    # ---- stage 2: cheap per-question head over the cached state ---------------------------
    def embed_text(self, ids):
        """ids [...,T] -> [...,d] masked mean of the SHARED encoder (labels, instructions)."""
        sh, T = ids.shape[:-1], ids.size(-1)
        flat = ids.reshape(-1, T)
        m = flat != PAD
        m[:, 0] = True
        h = self.encoder(self.tok(flat), m, torch.arange(T, device=ids.device))
        w = m.unsqueeze(-1).to(h.dtype)
        return ((h * w).sum(1) / w.sum(1)).view(*sh, -1)

    def decide(self, h, hmask, opt_ids, opt_mask, instr_ids, qtype):
        """opt_ids [B,Q,K,T]; opt_mask [B,Q,K]; instr_ids [B,Q,Ti]; qtype [B,Q].
        Returns logits [B,Q,K+1]; padded options = -inf; last slot = unknown."""
        B, Q, K, _ = opt_ids.shape
        ctx = self.embed_text(instr_ids) + self.type_emb(qtype)          # [B,Q,d] shared question context
        e = self.embed_text(opt_ids) + ctx.unsqueeze(2)                  # [B,Q,K,d]
        e = torch.cat([e, (self.unknown + ctx).unsqueeze(2)], 2)
        opt_mask = torch.cat([opt_mask, torch.ones(B, Q, 1, dtype=torch.bool, device=opt_mask.device)], 2)
        q = e
        for layer in self.head:
            q = layer(q, opt_mask, h, hmask)
        logits = self.scorer(q).squeeze(-1)
        t = torch.stack([torch.exp(self.log_temp[k]) for k in QTYPES])[qtype].unsqueeze(-1)
        return (logits / t).masked_fill(~opt_mask, float("-inf"))

    def forward(self, ids, mask, opt_ids, opt_mask, instr_ids, qtype):
        h, hm = self.encode_state(ids, mask)
        return self.decide(h, hm, opt_ids, opt_mask, instr_ids, qtype)
