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
        # MPS is solid on float16 for many ops; bf16 support varies by OS
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


def training_precision_flags(dtype) -> dict[str, Any]:
    """HF TrainingArguments fp16/bf16 — only enable on CUDA/ROCm."""
    torch = require_torch()
    kind = device_kind()
    if kind not in ("cuda", "rocm"):
        return {"fp16": False, "bf16": False}
    return {
        "fp16": dtype == torch.float16,
        "bf16": dtype == torch.bfloat16,
    }


def supports_bnb_4bit() -> bool:
    """bitsandbytes 4-bit is CUDA-only in practice."""
    if device_kind() != "cuda":
        return False
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


# Public small defaults when recipe sets size: but omits model:
SIZE_DEFAULT_MODELS = {
    "edge": "hf-internal-testing/tiny-random-gpt2",
    "slm": "hf-internal-testing/tiny-random-gpt2",
    "llm": "hf-internal-testing/tiny-random-gpt2",
    "tiny": "hf-internal-testing/tiny-random-gpt2",
}

SIZE_DEFAULT_ENCODER = {
    "edge": "hf-internal-testing/tiny-random-bert",
    "slm": "hf-internal-testing/tiny-random-bert",
    "llm": "hf-internal-testing/tiny-random-bert",
}

SIZE_DEFAULT_SEQ2SEQ = {
    "edge": "hf-internal-testing/tiny-random-t5",
    "slm": "hf-internal-testing/tiny-random-t5",
    "llm": "hf-internal-testing/tiny-random-t5",
}
