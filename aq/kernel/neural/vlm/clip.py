"""CLIP / SigLIP dual-encoder models and losses (aq-owned)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from neural.vit.factory import build_vit, resolve_arch
from neural.vit.vit import VisionTransformer

from .text_encoder import CLIPTextTransformer


class CLIPVisionTower(nn.Module):
    """ViT backbone + projection to shared embed dim (CLIP image tower)."""

    def __init__(
        self,
        *,
        arch: str = "vit-b/16",
        img_size: int = 224,
        embed_dim: int = 512,
        in_ch: int = 3,
    ):
        super().__init__()
        self.arch = resolve_arch(arch)
        backbone = build_vit(arch, num_classes=0, img_size=img_size, in_ch=in_ch)
        if not isinstance(backbone, VisionTransformer):
            raise SystemExit(f"CLIP vision tower needs a ViT arch, got {arch!r} ({type(backbone)})")
        self.backbone = backbone
        width = backbone.dim
        self.proj = nn.Parameter(torch.empty(width, embed_dim))
        nn.init.normal_(self.proj, std=width**-0.5)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(images)[:, 0]
        return feats @ self.proj


class CLIPModel(nn.Module):
    """OpenAI CLIP: dual encoders + learnable logit scale."""

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
    ):
        super().__init__()
        self.visual = CLIPVisionTower(arch=vision_arch, img_size=img_size, embed_dim=embed_dim)
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
        # init ≈ -10 as in SigLIP paper
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
