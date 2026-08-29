"""Read recipe knobs from top-level or nested blocks (train/lora/quantization/llm)."""

from __future__ import annotations

from typing import Any


def opt(rec: dict, key: str, default: Any = None, *sections: str) -> Any:
    if key in rec and rec[key] is not None:
        return rec[key]
    for sec in sections or ("train", "lora", "quantization", "llm", "model_args", "peft"):
        block = rec.get(sec)
        if isinstance(block, dict) and block.get(key) is not None:
            return block[key]
    # common aliases inside sections
    aliases = {
        "lr": ("learning_rate",),
        "steps": ("max_steps",),
        "batch_size": ("per_device_train_batch_size",),
        "rank": ("r", "lora_rank"),
        "alpha": ("lora_alpha",),
        "bits": ("load_in_4bit", "load_in_8bit"),
        "max_seq_len": ("max_length", "max_seq_length", "context"),
    }
    for sec in ("train", "lora", "quantization", "llm", "peft"):
        block = rec.get(sec)
        if not isinstance(block, dict):
            continue
        for alt in aliases.get(key, ()):
            if block.get(alt) is not None:
                v = block[alt]
                if alt in ("load_in_4bit",) and v is True:
                    return 4
                if alt in ("load_in_8bit",) and v is True:
                    return 8
                return v
    return default


def require_model_id(rec: dict) -> str:
    from backends.hf_lm import resolve_model_id

    return resolve_model_id(rec)
