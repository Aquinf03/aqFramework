"""Build aq-owned vision transformers by recipe arch name."""

from __future__ import annotations

from torch import nn

from .beit import BEiT
from .deit import DistilledVisionTransformer
from .swin import SwinTransformer
from .vit import VisionTransformer

ARCH_ALIASES: dict[str, str] = {
    # ViT
    "vit": "vit_b_16",
    "vit-b": "vit_b_16",
    "vit-b/16": "vit_b_16",
    "vit_b_16": "vit_b_16",
    "vit-b16": "vit_b_16",
    "vit-s": "vit_s_16",
    "vit-s/16": "vit_s_16",
    "vit_s_16": "vit_s_16",
    "vit-s16": "vit_s_16",
    "vit-t": "vit_t_16",
    "vit-t/16": "vit_t_16",
    "vit_t_16": "vit_t_16",
    "vit-t16": "vit_t_16",
    "vit-tiny": "vit_t_16",
    "vit-small": "vit_s_16",
    "vit-base": "vit_b_16",
    # Swin
    "swin": "swin_t",
    "swin-t": "swin_t",
    "swin_t": "swin_t",
    "swin-tiny": "swin_t",
    "swin-s": "swin_s",
    "swin_s": "swin_s",
    "swin-small": "swin_s",
    "swin-b": "swin_b",
    "swin_b": "swin_b",
    "swin-base": "swin_b",
    # DeiT
    "deit": "deit_b",
    "deit-t": "deit_t",
    "deit_t": "deit_t",
    "deit-tiny": "deit_t",
    "deit-s": "deit_s",
    "deit_s": "deit_s",
    "deit-small": "deit_s",
    "deit-b": "deit_b",
    "deit_b": "deit_b",
    "deit-base": "deit_b",
    # BEiT
    "beit": "beit_b",
    "beit-b": "beit_b",
    "beit_b": "beit_b",
    "beit-base": "beit_b",
    "beit-l": "beit_l",
    "beit_l": "beit_l",
    "beit-large": "beit_l",
}


def _norm_key(arch: str) -> str:
    s = str(arch).lower().replace(" ", "")
    return ARCH_ALIASES.get(s) or ARCH_ALIASES.get(s.replace("-", "_")) or ARCH_ALIASES.get(
        s.replace("_", "-")
    ) or ""


def list_arches() -> list[str]:
    return sorted(set(ARCH_ALIASES.values()))


def resolve_arch(arch: str) -> str:
    key = _norm_key(arch)
    if not key:
        # try slash forms already in map
        key = ARCH_ALIASES.get(str(arch).lower().replace(" ", "")) or ""
    if not key:
        raise SystemExit(f"unknown vit arch {arch!r}. Supported: {', '.join(list_arches())}")
    return key


def family_of(arch_key: str) -> str:
    if arch_key.startswith("swin"):
        return "swin"
    if arch_key.startswith("deit"):
        return "deit"
    if arch_key.startswith("beit"):
        return "beit"
    return "vit"


def default_image_size(arch: str) -> int:
    key = resolve_arch(arch)
    if key.startswith("swin"):
        return 224  # window-7 hierarchy
    return 224


def default_weight_decay(arch: str) -> float:
    return 0.05


def build_vit(
    arch: str,
    num_classes: int,
    *,
    img_size: int = 224,
    in_ch: int = 3,
    vocab_size: int = 8192,
) -> nn.Module:
    key = resolve_arch(arch)
    # ViT
    if key == "vit_t_16":
        return VisionTransformer(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=192, depth=12, heads=3,
        )
    if key == "vit_s_16":
        return VisionTransformer(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=384, depth=12, heads=6,
        )
    if key == "vit_b_16":
        return VisionTransformer(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=768, depth=12, heads=12,
        )
    # Swin — image_size must be divisible by 32; window 7 needs 56-grid at stage0 → 224
    if key.startswith("swin"):
        if img_size % 32 != 0:
            raise SystemExit(f"swin requires image_size divisible by 32, got {img_size}")
        grid0 = img_size // 4
        window = 7 if grid0 % 7 == 0 else (8 if grid0 % 8 == 0 else 4)
        if key == "swin_t":
            return SwinTransformer(
                img_size=img_size, num_classes=num_classes, in_ch=in_ch,
                embed_dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=window,
            )
        if key == "swin_s":
            return SwinTransformer(
                img_size=img_size, num_classes=num_classes, in_ch=in_ch,
                embed_dim=96, depths=(2, 2, 18, 2), heads=(3, 6, 12, 24), window=window,
            )
        if key == "swin_b":
            return SwinTransformer(
                img_size=img_size, num_classes=num_classes, in_ch=in_ch,
                embed_dim=128, depths=(2, 2, 18, 2), heads=(4, 8, 16, 32), window=window,
            )
    # DeiT
    if key == "deit_t":
        return DistilledVisionTransformer(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=192, depth=12, heads=3,
        )
    if key == "deit_s":
        return DistilledVisionTransformer(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=384, depth=12, heads=6,
        )
    if key == "deit_b":
        return DistilledVisionTransformer(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=768, depth=12, heads=12,
        )
    # BEiT
    if key == "beit_b":
        return BEiT(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=768, depth=12, heads=12, vocab_size=vocab_size,
        )
    if key == "beit_l":
        return BEiT(
            img_size=img_size, patch=16, in_ch=in_ch, num_classes=num_classes,
            dim=1024, depth=24, heads=16, vocab_size=vocab_size,
        )
    raise SystemExit(f"vit arch {key!r} not wired")
