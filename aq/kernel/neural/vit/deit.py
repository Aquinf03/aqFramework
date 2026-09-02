"""DeiT (Touvron et al.) — ViT + distillation token; train with a CNN teacher."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Block, PatchEmbed, init_vit_weights, trunc_normal_


class DistilledVisionTransformer(nn.Module):
    """DeiT student: CLS + distillation tokens, two classification heads."""

    def __init__(
        self,
        img_size: int = 224,
        patch: int = 16,
        in_ch: int = 3,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.patch_embed = PatchEmbed(img_size, patch, in_ch, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 2, dim))
        self.pos_drop = nn.Dropout(drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(dim, heads, mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=dpr[i])
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.head_dist = nn.Linear(dim, num_classes)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.dist_token, std=0.02)
        self.apply(init_vit_weights)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        dist = self.dist_token.expand(b, -1, -1)
        x = torch.cat([cls, dist, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cls, dist = self.forward_features(x)
        logits_cls = self.head(cls)
        logits_dist = self.head_dist(dist)
        if self.training:
            return logits_cls, logits_dist
        # inference: average the two heads (DeiT)
        return (logits_cls + logits_dist) / 2


def deit_loss(
    logits_cls: torch.Tensor,
    logits_dist: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 3.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Hard-label CE on CLS + soft KL distillation on dist head (DeiT)."""
    ce = F.cross_entropy(logits_cls, labels)
    t = temperature
    distill = F.kl_div(
        F.log_softmax(logits_dist / t, dim=-1),
        F.softmax(teacher_logits / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)
    return (1.0 - alpha) * ce + alpha * distill
