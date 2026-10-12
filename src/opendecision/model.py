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
from typing import Callable, Literal
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
    init_scheme: Literal["v1", "trunc002"] = "v1"  # trunc002: weights trunc-normal std 0.02 (+-3 sigma), biases 0 (arch-review A/W1)
    emb_norm: bool = False       # LayerNorm after the token embedding (arch-review A/W2)
    keep_tokens: bool = False    # heads also see full token states (+chunk id). UNPROVEN: in the toy needle test it stayed at chance (0.5) where latents-only reached 1.0 in 500 steps
    # block v2 (arch-review B): pre-norm RMSNorm, bias-free linears, RMS QK-norm, SwiGLU d_ff=swiglu_ff(d)
    block: Literal["v1", "v2"] = "v1"
    rope_scaling: Literal["none", "linear", "yarn"] = "none"  # applied only when rope_factor > 1
    rope_factor: float = 1.0
    attn_gate: Literal["none", "headwise", "elementwise"] = "none"  # sigmoid gate on SDPA output (2505.06708 G1)
    # ELECTRA RTD (improvement C, arXiv 2003.10555): >0 adds the generator `gen.*` and the discriminator head
    # `rtd_disc.*`. Generator = same Encoder class, same head dim, heads = round(ratio*heads), tied tok embeddings.
    rtd_gen_ratio: float = 0.0
    rtd_gen_layers: int = 0      # 0 = same as layers

def gen_dims(cfg: ModelConfig) -> tuple[int, int, int]:
    """(d_gen, heads_gen, layers_gen): head dim kept (one RoPE table), heads = int(ratio*heads + 0.5) >= 1."""
    hg = max(1, int(cfg.rtd_gen_ratio * cfg.heads + 0.5))
    return hg * (cfg.d // cfg.heads), hg, cfg.rtd_gen_layers or cfg.layers


def swiglu_ff(d: int) -> int:
    """(8/3)*d rounded (half up) to a multiple of 64: SwiGLU 3*d*f ~ param-matched to the 4d GELU MLP."""
    return (d + 12) // 24 * 64


def rope_table(n: int, head_dim: int, theta: float, scaling: str = "none", factor: float = 1.0,
               orig_len: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [n, head_dim//2], built on CPU in float64 (host-independent) and cast to fp32.
    scaling (only when factor > 1): "linear" = position interpolation (all freqs / factor); "yarn" =
    arXiv 2309.00071 NTK-by-parts ramp over r = orig_len/wavelength (alpha=1, beta=32) plus the
    attention temperature sqrt(1/t) = 0.1 ln(factor) + 1 folded into cos/sin (so it scales q and k)."""
    half = head_dim // 2
    inv = theta ** (-torch.arange(half, device="cpu", dtype=torch.float64) / half)
    mscale = 1.0
    if scaling == "linear" and factor > 1:
        inv = inv / factor
    elif scaling == "yarn" and factor > 1:
        g = ((orig_len * inv / (2 * math.pi) - 1.0) / (32.0 - 1.0)).clamp(0.0, 1.0)
        inv = (1 - g) * inv / factor + g * inv
        mscale = 0.1 * math.log(factor) + 1.0
    ang = torch.arange(n, device="cpu", dtype=torch.float64)[:, None] * inv
    if mscale != 1.0:
        return (ang.cos() * mscale).to(torch.float32), (ang.sin() * mscale).to(torch.float32)
    return ang.cos().to(torch.float32), ang.sin().to(torch.float32)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Bias-free RMSNorm computed in fp32 with explicit ops (deterministic), cast back to x.dtype."""
    y = x.float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self, x):
        return rms_norm(x, self.weight)


class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.gate, self.up = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B,H,N,hd]; cos/sin [N,hd/2] (rotate-half)."""
    half = x.size(-1) // 2
    cos, sin = cos.to(device=x.device, dtype=x.dtype), sin.to(device=x.device, dtype=x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

def sincos(idx, d):
    f = 10_000.0 ** (-torch.arange(0, d, 2, device=idx.device, dtype=torch.float32) / d)
    a = idx.to(torch.float32)[:, None] * f
    return torch.cat([a.sin(), a.cos()], -1)[:, :d]


class Block(nn.Module):
    def __init__(self, d, heads, block="v1", attn_gate="none"):
        super().__init__()
        self.h = heads
        hd = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        if block == "v1":
            self.qn, self.kn = nn.LayerNorm(hd), nn.LayerNorm(hd)   # QK-norm: stability at scale
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        else:
            self.qn, self.kn = RMSNorm(hd), RMSNorm(hd)
            self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
            self.ff = SwiGLU(d, swiglu_ff(d))
        self.attn_gate = attn_gate
        if attn_gate != "none":   # G1: gate computed from the (normed) block input, per head or per element
            self.gate = nn.Linear(d, heads if attn_gate == "headwise" else d, bias=False)

    def forward(self, x, mask, cos, sin):
        B, N, d = x.shape
        y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.qn(q), cos, sin), apply_rope(self.kn(k), cos, sin)
        am = None if mask is None else mask[:, None] if mask.dim() == 3 else mask[:, None, None, :]  # [B,N,N]: doc mask
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        a = a.transpose(1, 2)
        if self.attn_gate != "none":
            a = a * torch.sigmoid(self.gate(y)).view(B, N, self.h, -1)
        x = x + self.o(a.reshape(B, N, d))
        return x + self.ff(self.n2(x))

class Encoder(nn.Module):
    def __init__(self, d, heads, layers, block="v1", attn_gate="none"):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads, block, attn_gate) for _ in range(layers)])
        self.norm = nn.LayerNorm(d) if block == "v1" else RMSNorm(d)

    def forward(self, x, mask, cos, sin):
        for b in self.blocks:
            x = b(x, mask, cos, sin)
        return self.norm(x)

class Generator(nn.Module):
    """ELECTRA generator over the shared (tied) token embedding: proj_in d->d_gen, Encoder, proj_out d_gen->d;
    MLM logits = proj_out(h) @ tok.weight.T."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        dg, hg, lg = gen_dims(cfg)
        self.proj_in = nn.Linear(cfg.d, dg, bias=False)
        self.encoder = Encoder(dg, hg, lg, cfg.block, cfg.attn_gate)
        self.proj_out = nn.Linear(dg, cfg.d, bias=False)

    def forward(self, x, mask, cos, sin):
        return self.proj_out(self.encoder(self.proj_in(x), mask, cos, sin))

class Attention(nn.Module):
    """Head attention (SDPA): no bias, no QK-norm, no RoPE (options are order-equivariant)."""
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, kv, key_mask):
        B, Lq, d = x.shape
        Lk = kv.size(1)
        q = self.q(x).view(B, Lq, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(kv).view(B, Lk, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None if key_mask is None else key_mask[:, None, None, :])
        return self.o(a.transpose(1, 2).reshape(B, Lq, d))

class HeadLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.opt_attn = Attention(d, heads)
        self.cross = Attention(d, heads)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, qmask, h, hmask):
        # q [B,Q,K,d], qmask [B,Q,K]; h [B,N,d] (NOT repeated per question), hmask [B,N]
        B, Q, K, d = q.shape
        x = self.n1(q).reshape(B * Q, K, d)
        q = q + self.opt_attn(x, x, qmask.reshape(B * Q, K)).view(B, Q, K, d)
        x = self.n2(q).reshape(B, Q * K, d)
        q = q + self.cross(x, h, hmask).view(B, Q, K, d)
        return q + self.ff(self.n3(q))

class TokEmbed(nn.Module):
    """Embedding table; the lookup (and so its backward) is pluggable: embed_fn(weight, ids)."""
    def __init__(self, vocab, d, embed_fn=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, d))
        nn.init.normal_(self.weight)
        self.embed_fn = embed_fn or (lambda w, ids: F.embedding(ids, w))

    def forward(self, ids):
        return self.embed_fn(self.weight, ids)

def _encoder_names(prefix, d, heads, layers, block="v1", attn_gate="none"):
    n = {}
    v1 = block == "v1"
    for i in range(layers):
        p = f"{prefix}.blocks.{i}"
        n.update({f"{p}.qkv.weight": (3 * d, d), f"{p}.o.weight": (d, d)})
        for k in ("qn", "kn"):
            n[f"{p}.{k}.weight"] = (d // heads,)
            if v1:
                n[f"{p}.{k}.bias"] = (d // heads,)
        for k in ("n1", "n2"):
            n[f"{p}.{k}.weight"] = (d,)
            if v1:
                n[f"{p}.{k}.bias"] = (d,)
        if v1:
            n.update({f"{p}.ff.0.weight": (4 * d, d), f"{p}.ff.0.bias": (4 * d,),
                      f"{p}.ff.2.weight": (d, 4 * d), f"{p}.ff.2.bias": (d,)})
        else:
            f = swiglu_ff(d)
            n.update({f"{p}.ff.gate.weight": (f, d), f"{p}.ff.up.weight": (f, d), f"{p}.ff.down.weight": (d, f)})
        if attn_gate != "none":
            n[f"{p}.gate.weight"] = (heads if attn_gate == "headwise" else d, d)
    n[f"{prefix}.norm.weight"] = (d,)
    if v1:
        n[f"{prefix}.norm.bias"] = (d,)
    return n

def param_names(cfg: "ModelConfig") -> dict[str, tuple[int, ...]]:
    """Pinned state-dict names and shapes (single source for counts, checkpoints, adapter)."""
    d = cfg.d
    n = _encoder_names("encoder", d, cfg.heads, cfg.layers, cfg.block, cfg.attn_gate)
    n["tok.weight"] = (cfg.vocab, d)
    if cfg.emb_norm:
        n.update({"emb_norm.weight": (d,), "emb_norm.bias": (d,)})
    n.update({"type_emb.weight": (3, d), "unknown": (d,)})
    for j in range(cfg.head_layers):
        p = f"head.{j}"
        for a in ("opt_attn", "cross"):
            n.update({f"{p}.{a}.q.weight": (d, d), f"{p}.{a}.kv.weight": (2 * d, d), f"{p}.{a}.o.weight": (d, d)})
        n.update({f"{p}.ff.0.weight": (4 * d, d), f"{p}.ff.0.bias": (4 * d,),
                  f"{p}.ff.2.weight": (d, 4 * d), f"{p}.ff.2.bias": (d,)})
        for k in ("n1", "n2", "n3"):
            n.update({f"{p}.{k}.weight": (d,), f"{p}.{k}.bias": (d,)})
    n.update({"scorer.0.weight": (d,), "scorer.0.bias": (d,), "scorer.1.weight": (d, d), "scorer.1.bias": (d,),
              "scorer.3.weight": (1, d), "scorer.3.bias": (1,)})
    n.update({f"log_temp.{k}": () for k in QTYPES})
    if cfg.rtd_gen_ratio > 0:
        dg, hg, lg = gen_dims(cfg)
        n.update({"gen.proj_in.weight": (dg, d), "gen.proj_out.weight": (d, dg)})
        n.update(_encoder_names("gen.encoder", dg, hg, lg, cfg.block, cfg.attn_gate))
        n.update({"rtd_disc.weight": (1, d), "rtd_disc.bias": (1,)})
    if cfg.chunk:
        n["chunk_q"] = (cfg.chunk_latents, d)
        n.update({f"chunk_pool.{k}.weight": s for k, s in (("q", (d, d)), ("kv", (2 * d, d)), ("o", (d, d)))})
        n.update(_encoder_names("global_enc", d, cfg.heads, cfg.global_layers, cfg.block, cfg.attn_gate))
    return n

def od_param_count(cfg: "ModelConfig") -> int:
    return sum(math.prod(s) for s in param_names(cfg).values())

class OpenDecisionModel(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None, *, embed_fn: Callable | None = None):
        super().__init__()
        c = self.cfg = cfg or ModelConfig()
        self.tok = TokEmbed(c.vocab, c.d, embed_fn)
        if c.emb_norm:
            self.emb_norm = nn.LayerNorm(c.d)
        self.encoder = Encoder(c.d, c.heads, c.layers, c.block, c.attn_gate)
        cos, sin = self._table(c.max_len, c.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.type_emb = nn.Embedding(3, c.d)
        self.unknown = nn.Parameter(torch.randn(c.d) * 0.02)
        self.head = nn.ModuleList([HeadLayer(c.d, c.heads) for _ in range(c.head_layers)])
        self.scorer = nn.Sequential(nn.LayerNorm(c.d), nn.Linear(c.d, c.d), nn.GELU(), nn.Linear(c.d, 1))
        self.log_temp = nn.ParameterDict({k: nn.Parameter(torch.zeros(())) for k in QTYPES})
        if c.rtd_gen_ratio > 0:
            self.gen = Generator(c)
            self.rtd_disc = nn.Linear(c.d, 1)
        if c.chunk:
            self.chunk_q = nn.Parameter(torch.randn(c.chunk_latents, c.d) * 0.02)
            self.chunk_pool = Attention(c.d, c.heads)
            self.global_enc = Encoder(c.d, c.heads, c.global_layers, c.block, c.attn_gate)
            gcos, gsin = self._table(c.max_len, c.global_theta)
            self.register_buffer("rope_gcos", gcos, persistent=False)
            self.register_buffer("rope_gsin", gsin, persistent=False)
        if c.init_scheme == "trunc002":
            for n, p in self.named_parameters():
                if p.dim() >= 2 or n == "unknown":
                    nn.init.trunc_normal_(p, std=0.02, a=-0.06, b=0.06)
                elif n.endswith(".bias"):
                    nn.init.zeros_(p)

    def _embed(self, ids):
        x = self.tok(ids)
        return self.emb_norm(x) if self.cfg.emb_norm else x

    def _table(self, n, theta):
        c = self.cfg
        return rope_table(n, c.d // c.heads, theta, c.rope_scaling, c.rope_factor, c.max_len)

    def _rope(self, n, glob=False):
        """Slice of the precomputed table; beyond max_len the table is rebuilt (RoPE has no hard cap)."""
        c = self.cfg
        cos, sin = (self.rope_gcos, self.rope_gsin) if glob else (self.rope_cos, self.rope_sin)
        if n <= cos.size(0):
            return cos[:n], sin[:n]
        return self._table(n, c.global_theta if glob else c.rope_theta)

    # ---- stage 1: encode state once ----------------------------------------------------
    def _chunks(self, ids, mask):
        c = self.cfg
        B, N = ids.shape
        pad = (-N) % c.chunk
        ids, mask = F.pad(ids, (0, pad)), F.pad(mask, (0, pad))
        nc = ids.size(1) // c.chunk
        x = self._embed(ids).reshape(B * nc, c.chunk, -1)
        m = mask.reshape(B * nc, c.chunk)
        m = m | (~m.any(1, keepdim=True) & (torch.arange(c.chunk, device=ids.device) == 0))  # empty chunk stays finite
        h = self.encoder(x, m, *self._rope(c.chunk))
        return h, m, mask, B, nc

    def encode_tokens(self, ids, mask=None):
        """Per-token states [B,N,d] (flat or chunk-local); used by MLM pretraining. mask=None: packed, all real."""
        if self.cfg.chunk and ids.size(1) > self.cfg.chunk:
            if mask is None:
                mask = torch.ones_like(ids, dtype=torch.bool)
            h, _, _, B, nc = self._chunks(ids, mask)
            return h.reshape(B, nc * self.cfg.chunk, -1)[:, : ids.size(1)]
        return self.encoder(self._embed(ids), mask, *self._rope(ids.size(1)))

    def encode_state(self, ids, mask=None):
        c = self.cfg
        if not (c.chunk and ids.size(1) > c.chunk):
            return self.encode_tokens(ids, mask), mask
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        h, m, mask, B, nc = self._chunks(ids, mask)
        q = self.chunk_q.unsqueeze(0).expand(B * nc, -1, -1)
        lat = self.chunk_pool(q, h, m)
        lat = lat.reshape(B, nc * c.chunk_latents, -1)
        lm = mask.reshape(B, nc, c.chunk).any(2).unsqueeze(-1).expand(-1, -1, c.chunk_latents).reshape(B, -1)
        g = self.global_enc(lat, lm, *self._rope(nc * c.chunk_latents, glob=True))
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
        h = self.encoder(self._embed(flat), m, *self._rope(T))
        w = m.unsqueeze(-1).to(h.dtype)
        return ((h * w).sum(1) / w.sum(1)).view(*sh, -1)

    def decide(self, h, hmask, opt_ids, opt_mask, instr_ids, qtype):
        """opt_ids [B,Q,K,T]; opt_mask [B,Q,K]; instr_ids [B,Q,Ti]; qtype [B,Q].
        Returns (logits [B,Q,K+1], opt_mask_ext [B,Q,K+1]); padded options = -inf; last slot = unknown.
        No index gathers: type embedding and temperature are one-hot matmuls (deterministic backward)."""
        B, Q, K, _ = opt_ids.shape
        oh = F.one_hot(qtype, 3)
        ctx = self.embed_text(instr_ids) + oh.to(h.dtype) @ self.type_emb.weight   # [B,Q,d] shared question context
        e = self.embed_text(opt_ids) + ctx.unsqueeze(2)                  # [B,Q,K,d]
        e = torch.cat([e, (self.unknown + ctx).unsqueeze(2)], 2)
        opt_mask = torch.cat([opt_mask, torch.ones(B, Q, 1, dtype=torch.bool, device=opt_mask.device)], 2)
        q = e
        for layer in self.head:
            q = layer(q, opt_mask, h, hmask)
        logits = self.scorer(q).squeeze(-1)
        t = torch.exp(oh.float() @ torch.stack([self.log_temp[k] for k in QTYPES])).unsqueeze(-1)
        return (logits / t).masked_fill(~opt_mask, float("-inf")), opt_mask

    def forward(self, ids, mask, opt_ids, opt_mask, instr_ids, qtype):
        h, hm = self.encode_state(ids, mask)
        return self.decide(h, hm, opt_ids, opt_mask, instr_ids, qtype)
