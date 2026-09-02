"""Aq-owned neural modules (vision CNNs + ViT family)."""

from neural.cnn import ARCH_ALIASES as CNN_ARCH_ALIASES
from neural.cnn import build_cnn, list_arches as list_cnn_arches
from neural.vit import ARCH_ALIASES as VIT_ARCH_ALIASES
from neural.vit import build_vit, list_arches as list_vit_arches

# back-compat
ARCH_ALIASES = CNN_ARCH_ALIASES
list_arches = list_cnn_arches

__all__ = [
    "build_cnn",
    "build_vit",
    "list_cnn_arches",
    "list_vit_arches",
    "list_arches",
    "ARCH_ALIASES",
    "CNN_ARCH_ALIASES",
    "VIT_ARCH_ALIASES",
]
