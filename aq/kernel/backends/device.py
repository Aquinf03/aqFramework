"""Pick compute device: CUDA (NVIDIA), ROCm (AMD via torch CUDA API), MPS (Apple), CPU."""

from __future__ import annotations

from typing import Any


def require_torch():
    from backends.deps import require_torch as _rt

    return _rt()


def device_kind() -> str:
    torch = require_torch()
    if torch.cuda.is_available():
        # ROCm builds also report cuda.is_available(); hip version marks AMD
        hip = getattr(getattr(torch, "version", None), "hip", None)
        return "rocm" if hip else "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_device():
    torch = require_torch()
    kind = device_kind()
    if kind in ("cuda", "rocm"):
        return torch.device("cuda")
    if kind == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(rec: dict | None = None):
    torch = require_torch()
    name = ""
    if rec:
        from backends.recipe_opt import opt

        name = str(opt(rec, "dtype", "") or "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    kind = device_kind()
    if kind == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if kind == "rocm":
        return torch.float16
    if kind == "mps":
        return torch.float16
    return torch.float32


def move_batch(batch: dict, model) -> dict:
    """Place tensors on the same device as model parameters."""
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch_device()
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(dev)
        else:
            out[k] = v
    return out


def resolve_train_precision(rec: dict | None = None) -> tuple[Any, dict[str, Any]]:
    """(load_dtype, TrainingArguments fp16/bf16 flags) — VRAM-safe + hardware-honest.

    Never enables bf16 on GPUs that do not support it (e.g. T4). Emulated bf16 is
    many× slower than fp16 and is a common “aq is slow vs my script” footgun.
    """
    torch = require_torch()
    kind = device_kind()
    compute = default_dtype(rec)
    off = {"fp16": False, "bf16": False}

    if kind not in ("cuda", "rocm"):
        return compute, off

    if compute == torch.float32:
        return torch.float32, off

    bf16_ok = False
    if kind == "cuda":
        try:
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_ok = False

    # Explicit bf16 on non-bf16 hardware → fp16 (do not emulate).
    if compute == torch.bfloat16 and not bf16_ok:
        return torch.float16, off

    if compute == torch.bfloat16 and bf16_ok:
        return torch.bfloat16, {"fp16": False, "bf16": True}

    # fp16 recipe: promote to bf16 only when hardware supports it.
    if compute == torch.float16 and bf16_ok:
        return torch.bfloat16, {"fp16": False, "bf16": True}

    if compute == torch.float16:
        return torch.float16, off

    return compute, off


def training_precision_flags(dtype) -> dict[str, Any]:
    """Legacy helper — prefer resolve_train_precision(rec)."""
    torch = require_torch()
    kind = device_kind()
    if kind not in ("cuda", "rocm"):
        return {"fp16": False, "bf16": False}
    if dtype == torch.bfloat16:
        try:
            if kind == "cuda" and torch.cuda.is_bf16_supported():
                return {"fp16": False, "bf16": True}
        except Exception:
            pass
        return {"fp16": False, "bf16": False}
    return {"fp16": False, "bf16": False}


def model_load_dtype(rec: dict | None = None):
    """Dtype for from_pretrained under HF Trainer (see resolve_train_precision)."""
    load_dtype, _ = resolve_train_precision(rec)
    return load_dtype


def apply_pretrained_dtype(load_kw: dict[str, Any], dtype) -> None:
    """Set load dtype for transformers from_pretrained."""
    load_kw["dtype"] = dtype


def cuda_alloc_hygiene() -> None:
    """Reduce fragmentation before a heavy train."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if device_kind() not in ("cuda", "rocm"):
        return
    torch = require_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def supports_bnb_4bit() -> bool:
    """bitsandbytes 4-bit is CUDA-only in practice."""
    if device_kind() != "cuda":
        return False
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


# size: is a label only (llm|slm|edge). It does NOT pick a model.
# Always set recipe.model to a hub id or local path.
