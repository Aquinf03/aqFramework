"""LoRA / QLoRA from recipe.yaml only. Requires recipe.model (hub id or path)."""

from __future__ import annotations

from pathlib import Path

from backends import hf_lm


def fit(src: Path, rec: dict) -> dict:
    # method: lora | qlora — recipe.model required; no toy path
    name = str(rec.get("method") or "lora")
    return hf_lm.fit(src, rec, method_name=name)


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return hf_lm.evaluate(model, src, rec)


def generate(model, prompt, rec, max_tokens=None, temperature=None):
    return hf_lm.generate(model, prompt, rec, max_tokens=max_tokens, temperature=temperature)


def write_inspect(train: Path, model: dict) -> str:
    return hf_lm.write_inspect(train, model)
