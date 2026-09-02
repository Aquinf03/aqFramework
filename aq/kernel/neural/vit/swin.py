"""Swin Transformer (Liu et al.) — hierarchical window / shifted-window attention."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import (
    DropPath,
    MLP,
    RelativePositionBias,
    init_vit_weights,
    window_partition,
    window_reverse,
)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window: int, heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window = window
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rel = RelativePositionBias(window, heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: nW*B, N, C
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.rel()
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b // nW, nW, self.heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, n, n)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        heads: int,
        window: int = 7,
        shift: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.h, self.w = input_resolution
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window, heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

        if self.shift > 0:
            # attention mask for SW-MSA
            img = torch.zeros((1, self.h, self.w, 1))
            h_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            w_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img[:, hs, ws, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img, window).view(-1, window * window)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.h, self.w
        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        x_windows = window_partition(x, self.window).view(-1, self.window * self.window, c)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        x = window_reverse(attn_windows.view(-1, self.window, self.window, c), self.window, h, w)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = x.view(b, h * w, c)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int):
        super().__init__()
        self.h, self.w = input_resolution
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = self.h, self.w
        b, l, c = x.shape
        x = x.view(b, h, w, c)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(b, -1, 4 * c)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        heads: int,
        window: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path,
        downsample: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim,
                    input_resolution,
                    heads,
                    window=window,
                    shift=0 if (i % 2 == 0) else window // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            PatchMerging(input_resolution, dim) if downsample else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch: int = 4,
        in_ch: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        heads: tuple[int, ...] = (3, 6, 12, 24),
        window: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch = patch
        assert img_size % patch == 0
        self.patches_resolution = img_size // patch
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.pos_drop = nn.Dropout(drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.layers = nn.ModuleList()
        dim = embed_dim
        res = self.patches_resolution
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=dim,
                input_resolution=(res, res),
                depth=depths[i],
                heads=heads[i],
                window=window,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                downsample=(i < self.num_layers - 1),
            )
            self.layers.append(layer)
            if i < self.num_layers - 1:
                res = res // 2
                dim *= 2
        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(init_vit_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)
        return self.head(x)
