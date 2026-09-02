"""LLaVA projector + Flamingo perceiver / gated cross-attention (aq-owned connectors)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from neural.vit.blocks import Attention, MLP, init_vit_weights


class LLaVAProjector(nn.Module):
    """LLaVA-1.5 style: Linear → GELU → Linear (vision_dim → lm_hidden)."""

    def __init__(self, vision_dim: int, lm_hidden: int, mlp_depth: int = 2):
        super().__init__()
        if mlp_depth < 1:
            raise ValueError("mlp_depth >= 1")
        layers: list[nn.Module] = [nn.Linear(vision_dim, lm_hidden)]
        for _ in range(mlp_depth - 1):
            layers += [nn.GELU(), nn.Linear(lm_hidden, lm_hidden)]
        self.net = nn.Sequential(*layers)
        self.apply(init_vit_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverResampler(nn.Module):
    """Flamingo perceiver: learnable latents cross-attend to vision tokens."""

    def __init__(
        self,
        *,
        dim: int = 768,
        depth: int = 6,
        heads: int = 12,
        num_latents: int = 64,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm_lat": nn.LayerNorm(dim),
                        "norm_media": nn.LayerNorm(dim),
                        "attn": Attention(dim, heads=heads),
                        "norm_ff": nn.LayerNorm(dim),
                        "ff": MLP(dim, dim * ff_mult),
                    }
                )
            )
        self.norm = nn.LayerNorm(dim)
        self.num_latents = num_latents

    def forward(self, media: torch.Tensor) -> torch.Tensor:
        """media: B, T, C (vision tokens) → B, num_latents, C."""
        b = media.shape[0]
        x = self.latents.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            # cross-attn: Q from latents, KV from media (via concat trick on our Attention)
            q = layer["norm_lat"](x)
            kv = layer["norm_media"](media)
            # build cross-attn manually
            attn = layer["attn"]
            bsz, n_q, c = q.shape
            n_kv = kv.shape[1]
            # project q from q, k/v from kv
            q_lin, k_lin, v_lin = attn.qkv.weight.chunk(3, dim=0)
            q_b, k_b, v_b = attn.qkv.bias.chunk(3) if attn.qkv.bias is not None else (None, None, None)
            qq = F.linear(q, q_lin, q_b).reshape(bsz, n_q, attn.heads, attn.head_dim).transpose(1, 2)
            kk = F.linear(kv, k_lin, k_b).reshape(bsz, n_kv, attn.heads, attn.head_dim).transpose(1, 2)
            vv = F.linear(kv, v_lin, v_b).reshape(bsz, n_kv, attn.heads, attn.head_dim).transpose(1, 2)
            scores = (qq * attn.scale) @ kk.transpose(-2, -1)
            weights = attn.attn_drop(scores.softmax(dim=-1))
            out = (weights @ vv).transpose(1, 2).reshape(bsz, n_q, c)
            out = attn.proj_drop(attn.proj(out))
            x = x + out
            x = x + layer["ff"](layer["norm_ff"](x))
        return self.norm(x)


class GatedCrossAttention(nn.Module):
    """Flamingo gated xattn block: tanh(gate) * cross_attn(text ← media)."""

    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.norm_txt = nn.LayerNorm(dim)
        self.norm_media = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = MLP(dim, dim * ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, text: torch.Tensor, media: torch.Tensor) -> torch.Tensor:
        # text: B, L, C ; media: B, M, C
        q = self.norm_txt(text)
        kv = self.norm_media(media)
        attn = self.attn
        bsz, n_q, c = q.shape
        n_kv = kv.shape[1]
        q_lin, k_lin, v_lin = attn.qkv.weight.chunk(3, dim=0)
        q_b, k_b, v_b = attn.qkv.bias.chunk(3) if attn.qkv.bias is not None else (None, None, None)
        qq = F.linear(q, q_lin, q_b).reshape(bsz, n_q, attn.heads, attn.head_dim).transpose(1, 2)
        kk = F.linear(kv, k_lin, k_b).reshape(bsz, n_kv, attn.heads, attn.head_dim).transpose(1, 2)
        vv = F.linear(kv, v_lin, v_b).reshape(bsz, n_kv, attn.heads, attn.head_dim).transpose(1, 2)
        scores = (qq * attn.scale) @ kk.transpose(-2, -1)
        weights = attn.attn_drop(scores.softmax(dim=-1))
        out = (weights @ vv).transpose(1, 2).reshape(bsz, n_q, c)
        out = attn.proj_drop(attn.proj(out))
        text = text + out * self.attn_gate.tanh()
        text = text + self.ff(self.norm_ff(text)) * self.ff_gate.tanh()
        return text
