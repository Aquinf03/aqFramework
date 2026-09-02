"""CLIP-style text transformer (Radford et al.) — aq-owned."""

from __future__ import annotations

import torch
from torch import nn

from neural.vit.blocks import Block, init_vit_weights, trunc_normal_


class CLIPTextTransformer(nn.Module):
    """Causal text tower → EOT pooled features."""

    def __init__(
        self,
        *,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = nn.ModuleList(
            [Block(width, heads, mlp_ratio=4.0, drop_path=0.0) for _ in range(layers)]
        )
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        self.register_buffer(
            "attn_mask", self._build_causal_mask(context_length), persistent=False
        )
        trunc_normal_(self.token_embedding.weight, std=0.02)
        trunc_normal_(self.positional_embedding, std=0.01)
        trunc_normal_(self.text_projection, std=width**-0.5)
        self.apply(init_vit_weights)

    @staticmethod
    def _build_causal_mask(n: int) -> torch.Tensor:
        # Additive mask: 0 keep, -inf block. Shape heads-broadcastable via Attention API
        # Our Block.Attention doesn't take attn mask — apply via bias on residual path
        # We'll override forward to use masked attention inline for text.
        mask = torch.empty(n, n).fill_(float("-inf")).triu_(1)
        return mask

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """text: B, L int64 → B, embed_dim (EOT features, L2-normalized externally)."""
        x = self.token_embedding(text)  # B, L, C
        x = x + self.positional_embedding[: x.shape[1]]
        # causal self-attn via custom loop using mask
        for blk in self.transformer:
            x = self._block_causal(blk, x)
        x = self.ln_final(x)
        # EOT = highest token id position per CLIP (argmax of token ids)
        eot = text.argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), eot]
        return pooled @ self.text_projection

    def _block_causal(self, blk: Block, x: torch.Tensor) -> torch.Tensor:
        # replicate Block forward but inject causal mask into attention scores
        h = blk.norm1(x)
        b, n, c = h.shape
        attn = blk.attn
        qkv = attn.qkv(h).reshape(b, n, 3, attn.heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q * attn.scale) @ k.transpose(-2, -1)
        mask = self.attn_mask[:n, :n]
        scores = scores + mask
        weights = attn.attn_drop(scores.softmax(dim=-1))
        h = (weights @ v).transpose(1, 2).reshape(b, n, c)
        h = attn.proj_drop(attn.proj(h))
        x = x + blk.drop_path(h)
        x = x + blk.drop_path(blk.mlp(blk.norm2(x)))
        return x
