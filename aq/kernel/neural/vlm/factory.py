"""Build aq VLM components from recipe arch names."""

from __future__ import annotations

from .clip import CLIPModel, SigLIPModel

CLIP_ARCHES = {
    "clip": "clip",
    "clip-b": "clip",
    "clip-base": "clip",
    "siglip": "siglip",
    "siglip-b": "siglip",
    "siglip-base": "siglip",
}

GEN_ARCHES = {
    "llava": "llava",
    "llava-1.5": "llava",
    "llava_1_5": "llava",
    "flamingo": "flamingo",
    "gpt4v": "llava",  # GPT-4V-style generative = LLaVA-class connector by default
    "gpt-4v": "llava",
    "gpt4v-style": "llava",
    "gpt-4v-style": "llava",
}


def resolve_clip_arch(arch: str) -> str:
    key = str(arch).lower().replace(" ", "")
    out = CLIP_ARCHES.get(key) or CLIP_ARCHES.get(key.replace("_", "-"))
    if not out:
        raise SystemExit(f"unknown clip arch {arch!r}. Supported: {sorted(set(CLIP_ARCHES.values()))}")
    return out


def resolve_gen_arch(arch: str) -> str:
    key = str(arch).lower().replace(" ", "")
    out = GEN_ARCHES.get(key) or GEN_ARCHES.get(key.replace("_", "-"))
    if not out:
        raise SystemExit(
            f"unknown generative vlm arch {arch!r}. Supported: llava | flamingo | gpt4v-style"
        )
    return out


def build_clip(
    arch: str = "clip",
    *,
    vision_arch: str = "vit-b/16",
    img_size: int = 224,
    embed_dim: int = 512,
    vocab_size: int = 49408,
    text_width: int = 512,
    text_heads: int = 8,
    text_layers: int = 12,
    context_length: int = 77,
):
    kind = resolve_clip_arch(arch)
    kwargs = dict(
        vision_arch=vision_arch,
        img_size=img_size,
        embed_dim=embed_dim,
        vocab_size=vocab_size,
        text_width=text_width,
        text_heads=text_heads,
        text_layers=text_layers,
        context_length=context_length,
    )
    if kind == "siglip":
        return SigLIPModel(**kwargs)
    return CLIPModel(**kwargs)
