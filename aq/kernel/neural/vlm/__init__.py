"""Vision–language modules — CLIP / SigLIP / LLaVA / Flamingo (aq-owned connectors + towers)."""

from .factory import build_clip, resolve_clip_arch, resolve_gen_arch

__all__ = ["build_clip", "resolve_clip_arch", "resolve_gen_arch"]
