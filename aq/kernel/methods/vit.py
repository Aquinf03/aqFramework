"""Vision transformers — recipe-only. Architectures in neural/vit (aq-owned)."""

from __future__ import annotations

from pathlib import Path

from backends import vision_vit as backend

__all__ = ["fit", "evaluate", "write_inspect"]


def fit(src: Path, rec: dict) -> dict:
    return backend.fit(src, rec)


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return backend.evaluate(model, src, rec)


def write_inspect(train: Path, model: dict) -> str:
    return backend.write_inspect(train, model)
