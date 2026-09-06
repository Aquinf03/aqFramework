"""CLIP / SigLIP — family: vlm, method: clip."""

from __future__ import annotations

from pathlib import Path

from backends import vlm_clip as backend

__all__ = ["fit", "evaluate", "write_inspect", "generate"]


def fit(src: Path, rec: dict) -> dict:
    return backend.fit(src, rec)


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return backend.evaluate(model, src, rec)


def write_inspect(train: Path, model: dict) -> str:
    return backend.write_inspect(train, model)


def generate(model, prompt, rec, max_tokens=None, temperature=None):
    return backend.generate(model, prompt, rec, max_tokens=max_tokens, temperature=temperature)
