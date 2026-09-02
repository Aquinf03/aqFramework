"""CNN package — real architectures owned by aq."""

from .factory import ARCH_ALIASES, build_cnn, list_arches

__all__ = ["build_cnn", "list_arches", "ARCH_ALIASES"]
