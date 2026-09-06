"""Vision tower builders — aq ViT or OpenCLIP (any model id + pretrained tag)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from neural.vit.factory import build_vit, resolve_arch
from neural.vit.vit import VisionTransformer


def require_open_clip():
    try:
        import open_clip
    except ImportError as e:
        raise SystemExit(
            "open_clip is required for pretrained / non-aq OpenCLIP vision towers. "
            "Install: pip install open-clip-torch"
        ) from e
    return open_clip


def open_clip_model_name(vision: str) -> str:
    """Map recipe vision string → open_clip create_model name (e.. ViT-H-14)."""
    s = str(vision).strip()
    for prefix in ("open_clip:", "openclip:", "open-clip:"):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :]
            break
    # already OpenCLIP style
    if re_vit_dash(s):
        return normalize_open_clip_name(s)
    # aq style vit-h/14 → ViT-H-14
    key = None
    try:
        key = resolve_arch(s)
    except SystemExit:
        return normalize_open_clip_name(s)
    mapping = {
        "vit_t_16": "ViT-S-16",  # no tiny in open_clip; caller should use aq
        "vit_s_16": "ViT-S-16",
        "vit_b_16": "ViT-B-16",
        "vit_b_32": "ViT-B-32",
        "vit_l_14": "ViT-L-14",
        "vit_l_16": "ViT-L-16",
        "vit_h_14": "ViT-H-14",
        "vit_g_14": "ViT-g-14",
    }
    if key in mapping:
        return mapping[key]
    return normalize_open_clip_name(s)


def re_vit_dash(s: str) -> bool:
    import re

    return bool(re.fullmatch(r"ViT-[A-Za-z0-9]+-\d+", s)) or bool(
        re.fullmatch(r"vit-[a-z0-9]+-\d+", s, flags=re.I)
    )


def normalize_open_clip_name(s: str) -> str:
    import re

    s = s.strip()
    m = re.fullmatch(r"[Vv]i[Tt]-([A-Za-z0-9]+)-(\d+)", s)
    if m:
        size = m.group(1)
        # preserve giant lowercase g / bigG conventions
        if size.lower() == "g":
            size = "g"
        elif size.lower() == "bigg":
            size = "bigG"
        else:
            size = size.upper() if len(size) <= 2 else size
        return f"ViT-{size}-{m.group(2)}"
    m = re.fullmatch(r"vit-([a-z0-9]+)/(\d+)", s.lower())
    if m:
        size = m.group(1)
        if size == "g":
            size = "g"
        elif size == "bigg":
            size = "bigG"
        else:
            size = size.upper()
        return f"ViT-{size}-{m.group(2)}"
    return s


def wants_open_clip(vision: str, pretrained: str | None) -> bool:
    v = str(vision).lower()
    if pretrained:
        return True
    return v.startswith(("open_clip:", "openclip:", "open-clip:"))


class OpenCLIPVisualTower(nn.Module):
    """
    Wrap open_clip visual so callers get:
      - .dim  (pre-projection width when available, else output dim)
      - forward_features → B, N, C with CLS at index 0 (best-effort)
      - forward → B, embed  (encode_image style, L2 not applied)
    """

    def __init__(self, visual: nn.Module, *, embed_dim: int | None = None):
        super().__init__()
        self.visual = visual
        self.embed_dim = int(embed_dim or getattr(visual, "output_dim", None) or 512)
        # trunk width for LLaVA-style patch stream
        width = getattr(visual, "width", None) or getattr(visual, "embed_dim", None)
        self.dim = int(width or self.embed_dim)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        v = self.visual
        # open_clip VisionTransformer exposes .trunk or transformer blocks
        if hasattr(v, "forward_features"):
            feats = v.forward_features(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            return feats
        if hasattr(v, "trunk") and hasattr(v.trunk, "forward_features"):
            return v.trunk.forward_features(images)
        # fallback: pooled embedding as single token
        pooled = self.forward(images)
        return pooled.unsqueeze(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)


def load_open_clip_visual(
    vision: str,
    *,
    pretrained: str | bool = True,
    img_size: int | None = None,
) -> tuple[OpenCLIPVisualTower, dict[str, Any]]:
    open_clip = require_open_clip()
    name = open_clip_model_name(vision)
    tag = pretrained if pretrained is not False else None
    if tag is True:
        tag = "openai"  # open_clip default when True; may fail for H/14 — user should set tag
    try:
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=tag or None)
    except Exception as e:
        raise SystemExit(
            f"open_clip failed to load vision={vision!r} as {name!r} pretrained={tag!r}: {e}\n"
            f"Pick a model+tag from open_clip.list_pretrained(), e.g. "
            f"vision: ViT-H-14  vision_pretrained: laion2b_s32b_b79k"
        ) from e
    visual = model.visual
    out_dim = getattr(visual, "output_dim", None) or getattr(model, "embed_dim", None) or 512
    tower = OpenCLIPVisualTower(visual, embed_dim=int(out_dim))
    meta = {
        "provider": "open_clip",
        "open_clip_name": name,
        "pretrained": tag,
        "embed_dim": tower.embed_dim,
        "dim": tower.dim,
    }
    return tower, meta


def build_aq_vit_backbone(vision: str, *, img_size: int, in_ch: int = 3) -> VisionTransformer:
    backbone = build_vit(vision, num_classes=0, img_size=img_size, in_ch=in_ch)
    if not isinstance(backbone, VisionTransformer):
        raise SystemExit(f"vision tower needs a ViT arch, got {vision!r} ({type(backbone)})")
    return backbone
