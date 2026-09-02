"""Shared Vision Transformer building blocks."""

from __future__ import annotations

import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch: int = 16, in_ch: int = 3, dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.grid = img_size // patch
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B, C, H, W
        return x.flatten(2).transpose(1, 2)  # B, N, C


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rel_pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, rel_pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    nn.init.trunc_normal_(tensor, std=std)
    return tensor


def init_vit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    # x: B, H, W, C → B*nW, window, window, C
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, c)


def window_reverse(windows: torch.Tensor, window: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window / window))
    x = windows.view(b, h // window, w // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class RelativePositionBias(nn.Module):
    """BEiT / Swin-style relative position bias table for a fixed window or grid."""

    def __init__(self, window: int, heads: int):
        super().__init__()
        self.window = window
        self.heads = heads
        self.bias = nn.Parameter(torch.zeros((2 * window - 1) * (2 * window - 1), heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
        coords = torch.flatten(coords, 1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window - 1
        rel[:, :, 1] += window - 1
        rel[:, :, 0] *= 2 * window - 1
        index = rel.sum(-1)
        self.register_buffer("index", index, persistent=False)

    def forward(self) -> torch.Tensor:
        # heads, N, N
        n = self.window * self.window
        bias = self.bias[self.index.view(-1)].view(n, n, self.heads)
        return bias.permute(2, 0, 1).contiguous().unsqueeze(0)
