"""QLoRA alias — same as lora with 4-bit when bits/quantization set."""

from __future__ import annotations

from pathlib import Path

from backends import hf_lm


def fit(src: Path, rec: dict) -> dict:
    rec = {**rec, "objective": rec.get("objective") or "qlora"}
    if rec.get("bits") is None and not (
        isinstance(rec.get("quantization"), dict) and rec["quantization"].get("load_in_4bit")
    ):
        rec = {**rec, "bits": 4}
    return hf_lm.fit(src, rec, method_name="qlora")


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return hf_lm.evaluate(model, src, rec)


def generate(model, prompt, rec, max_tokens=None, temperature=None):
    return hf_lm.generate(model, prompt, rec, max_tokens=max_tokens, temperature=temperature)


def write_inspect(train: Path, model: dict) -> str:
    return hf_lm.write_inspect(train, model)
