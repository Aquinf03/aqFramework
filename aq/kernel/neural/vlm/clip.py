"""CLIP / SigLIP dual-encoder models and losses."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from neural.vit.factory import resolve_arch

from .text_encoder import CLIPTextTransformer
from .vision_tower import (
    OpenCLIPVisualTower,
    build_aq_vit_backbone,
    load_open_clip_visual,
    wants_open_clip,
)


class CLIPVisionTower(nn.Module):
    """ViT backbone + projection to shared embed dim (CLIP image tower)."""

    def __init__(
        self,
        *,
        arch: str = "vit-b/16",
        img_size: int = 224,
        embed_dim: int = 512,
        in_ch: int = 3,
        pretrained: str | bool | None = None,
    ):
        super().__init__()
        self.arch = str(arch)
        self.pretrained = pretrained
        self._provider = "aq"

        if wants_open_clip(arch, pretrained if isinstance(pretrained, str) else None) or (
            isinstance(pretrained, str) and pretrained
        ):
            tag: str | bool | None = pretrained if pretrained not in (None, False) else None
            if tag is True:
                tag = None
            tower, meta = load_open_clip_visual(arch, pretrained=tag if tag else None, img_size=img_size)
            self.backbone = tower
            self._provider = "open_clip"
            self.arch_key = meta["open_clip_name"]
            width = tower.embed_dim
            # OpenCLIP visual already projects to embed space; optional remap if dims differ
            if width == embed_dim:
                self.proj = None
            else:
                self.proj = nn.Parameter(torch.empty(width, embed_dim))
                nn.init.normal_(self.proj, std=width**-0.5)
            self._open_clip_pooled = True
        else:
            self.arch_key = resolve_arch(arch)
            backbone = build_aq_vit_backbone(arch, img_size=img_size, in_ch=in_ch)
            self.backbone = backbone
            width = backbone.dim
            self.proj = nn.Parameter(torch.empty(width, embed_dim))
            nn.init.normal_(self.proj, std=width**-0.5)
            self._open_clip_pooled = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self._open_clip_pooled:
            feats = self.backbone(images)
            if self.proj is None:
                return feats
            return feats @ self.proj
        feats = self.backbone.forward_features(images)[:, 0]
        return feats @ self.proj


class CLIPModel(nn.Module):
    """CLIP: dual encoders + learnable logit scale."""

    def __init__(
        self,
        *,
        vision_arch: str = "vit-b/16",
        img_size: int = 224,
        embed_dim: int = 512,
        context_length: int = 77,
        vocab_size: int = 49408,
        text_width: int = 512,
        text_heads: int = 8,
        text_layers: int = 12,
        vision_pretrained: str | bool | None = None,
    ):
        super().__init__()
        self.visual = CLIPVisionTower(
            arch=vision_arch,
            img_size=img_size,
            embed_dim=embed_dim,
            pretrained=vision_pretrained,
        )
        self.text = CLIPTextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            width=text_width,
            heads=text_heads,
            layers=text_layers,
            embed_dim=embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.vision_arch = vision_arch
        self.vision_pretrained = vision_pretrained

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual(images), dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(text), dim=-1)

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> dict[str, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logit_scale": logit_scale.detach(),
        }


class SigLIPModel(CLIPModel):
    """SigLIP: same towers, learnable bias for sigmoid loss (Zhai et al.)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP)."""
    b = logits_per_image.shape[0]
    labels = torch.arange(b, device=logits_per_image.device)
    return (
        F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
    ) / 2


def siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
) -> torch.Tensor:
    """Sigmoid loss — no in-batch softmax; works with small batches."""
    logits = image_features @ text_features.t() * logit_scale + logit_bias
    b = logits.shape[0]
    labels = 2 * torch.eye(b, device=logits.device) - 1  # +1 diagonal, -1 off
    return -F.logsigmoid(labels * logits).mean()
