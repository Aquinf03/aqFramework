"""Device + machine awareness: CUDA / ROCm / MPS / CPU, dtype, VRAM plan."""

from __future__ import annotations

import sys
from dataclasses import dataclass
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


@dataclass
class MachineInfo:
    kind: str
    name: str
    total_gb: float | None
    free_gb: float | None
    bf16: bool
    index: int = 0

    @property
    def tight(self) -> bool:
        """≈ consumer 8GB and below — auto half + ckpt + batch 1."""
        if self.kind == "cpu":
            return False
        if self.kind == "mps":
            # unified memory; treat as tight unless clearly large
            return self.total_gb is None or self.total_gb <= 18.0
        return self.total_gb is not None and self.total_gb <= 10.0

    @property
    def mid(self) -> bool:
        if self.total_gb is None:
            return self.kind in ("cuda", "rocm", "mps")
        return self.total_gb <= 16.0


def probe_machine(index: int = 0) -> MachineInfo:
    """Read what aq is actually running on (VRAM, bf16, name)."""
    torch = require_torch()
    kind = device_kind()
    if kind in ("cuda", "rocm"):
        try:
            props = torch.cuda.get_device_properties(index)
            total = float(props.total_memory) / (1024**3)
            free = None
            try:
                free_b, _total_b = torch.cuda.mem_get_info(index)
                free = float(free_b) / (1024**3)
            except Exception:
                free = None
            bf16 = False
            try:
                bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16 = False
            name = getattr(props, "name", None) or f"{kind}:{index}"
            return MachineInfo(kind=kind, name=str(name), total_gb=total, free_gb=free, bf16=bf16, index=index)
        except Exception:
            return MachineInfo(kind=kind, name=kind, total_gb=None, free_gb=None, bf16=False, index=index)
    if kind == "mps":
        total = None
        try:
            # macOS: rough unified-memory hint via recommended max working set if present
            freemem = getattr(torch.mps, "recommended_max_memory", None)
            if callable(freemem):
                total = float(freemem()) / (1024**3)
        except Exception:
            total = None
        return MachineInfo(kind="mps", name="Apple MPS", total_gb=total, free_gb=None, bf16=False, index=0)
    return MachineInfo(kind="cpu", name="cpu", total_gb=None, free_gb=None, bf16=False, index=0)


def _recipe_dtype_name(rec: dict | None) -> str:
    if not rec:
        return ""
    from backends.recipe_opt import opt

    return str(opt(rec, "dtype", "") or "").lower()


def default_dtype(rec: dict | None = None):
    """Pick a compute dtype. Accelerators default to half — never silent fp32 on CUDA."""
    torch = require_torch()
    name = _recipe_dtype_name(rec)
    force = False
    if rec:
        from backends.recipe_opt import opt

        force = bool(opt(rec, "force_dtype", False, "train", "vlm", "llm"))
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        # Explicit fp32 on a tight GPU is usually an accident — demote unless forced.
        mach = probe_machine()
        if not force and mach.kind in ("cuda", "rocm", "mps") and (mach.tight or mach.mid):
            gb = f" ({mach.total_gb:.1f}GB)" if mach.total_gb is not None else ""
            print(f"  dtype  recipe asked float32 on {mach.name}{gb}", file=sys.stderr)
            print(
                "         → using half precision (set force_dtype: true to keep fp32)",
                file=sys.stderr,
            )
            if mach.kind == "cuda" and mach.bf16:
                return torch.bfloat16
            return torch.float16
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

    if kind == "cpu":
        return compute, off

    # MPS: load half weights; HF fp16/bf16 Trainer flags are unreliable — leave off.
    if kind == "mps":
        if compute == torch.float32:
            return torch.float32, off
        return torch.float16, off

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

    # Explicit bf16 on non-bf16 hardware → fp16 AMP (do not emulate).
    if compute == torch.bfloat16 and not bf16_ok:
        return torch.float16, {"fp16": True, "bf16": False}

    if compute == torch.bfloat16 and bf16_ok:
        return torch.bfloat16, {"fp16": False, "bf16": True}

    # fp16 recipe: promote to bf16 only when hardware supports it.
    if compute == torch.float16 and bf16_ok:
        return torch.bfloat16, {"fp16": False, "bf16": True}

    if compute == torch.float16:
        return torch.float16, {"fp16": True, "bf16": False}

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
        return {"fp16": True, "bf16": False}
    if dtype == torch.float16:
        return {"fp16": True, "bf16": False}
    return {"fp16": False, "bf16": False}


def model_load_dtype(rec: dict | None = None):
    """Dtype for from_pretrained under HF Trainer (see resolve_train_precision)."""
    load_dtype, _ = resolve_train_precision(rec)
    return load_dtype


def apply_pretrained_dtype(load_kw: dict[str, Any], dtype) -> None:
    """Set load dtype for transformers from_pretrained (new + legacy kw)."""
    load_kw["dtype"] = dtype
    load_kw["torch_dtype"] = dtype  # older transformers


@dataclass
class ComputePlan:
    """What aq will actually run given this machine + recipe."""

    dtype: Any
    prec_flags: dict[str, Any]
    batch_size: int
    grad_accum: int
    gradient_checkpointing: bool
    max_seq_len: int | None
    machine: MachineInfo
    notes: list[str]

    def as_event(self) -> dict[str, Any]:
        dname = str(self.dtype).replace("torch.", "")
        return {
            "device": self.machine.kind,
            "gpu": self.machine.name,
            "vram_gb": self.machine.total_gb,
            "vram_free_gb": self.machine.free_gb,
            "dtype": dname,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_seq_len": self.max_seq_len,
            "plan_notes": self.notes,
        }


def plan_compute(
    rec: dict | None = None,
    *,
    workload: str = "llm",
    default_batch: int = 1,
    default_max_seq: int | None = 512,
) -> ComputePlan:
    """
    Native machine plan for train/serve.

    workload: llm | vlm | vision | clip
    Recipe wins when explicit; aq fills gaps and demotes fp32 on tight GPUs.
    """
    from backends.recipe_opt import opt

    rec = rec or {}
    mach = probe_machine()
    load_dtype, prec = resolve_train_precision(rec)
    notes: list[str] = []

    batch = opt(rec, "batch_size", None, "train", "vlm", "llm", "vision", "clip")
    batch_set = batch is not None
    batch = int(batch if batch is not None else default_batch)

    accum = int(opt(rec, "grad_accum", 1, "train", "vlm", "llm") or 1)

    ckpt = opt(rec, "gradient_checkpointing", None, "train", "vlm", "llm")
    ckpt_set = ckpt is not None
    ckpt = bool(ckpt) if ckpt_set else False

    max_seq = opt(rec, "max_seq_len", None, "train", "vlm", "llm")
    max_seq_set = max_seq is not None
    max_seq_i = int(max_seq) if max_seq is not None else default_max_seq

    # Auto tighten for accelerators when recipe left knobs alone
    if mach.kind in ("cuda", "rocm", "mps"):
        if mach.tight or (workload == "vlm" and mach.mid):
            if not batch_set and batch > 1:
                notes.append(f"batch_size {batch}→1 (VRAM {mach.total_gb:.1f}GB)" if mach.total_gb else "batch_size →1")
                batch = 1
            elif not batch_set:
                batch = 1
            if not ckpt_set and workload in ("vlm", "llm"):
                ckpt = True
                notes.append("gradient_checkpointing on")
            if not max_seq_set and workload == "vlm" and max_seq_i and max_seq_i > 512 and mach.tight:
                notes.append(f"max_seq_len {max_seq_i}→512")
                max_seq_i = 512
        elif workload == "vlm" and not ckpt_set and mach.mid:
            ckpt = True
            notes.append("gradient_checkpointing on (VLM)")

    if mach.total_gb is not None:
        notes.insert(
            0,
            f"{mach.name} · {mach.total_gb:.1f}GB"
            + (f" · {mach.free_gb:.1f}GB free" if mach.free_gb is not None else ""),
        )
    else:
        notes.insert(0, mach.name)

    dname = str(load_dtype).replace("torch.", "")
    notes.append(f"dtype={dname}")

    return ComputePlan(
        dtype=load_dtype,
        prec_flags=prec,
        batch_size=max(1, batch),
        grad_accum=max(1, accum),
        gradient_checkpointing=ckpt,
        max_seq_len=max_seq_i,
        machine=mach,
        notes=notes,
    )


def log_plan(plan: ComputePlan, *, prefix: str = "  plan ") -> None:
    joined = " · ".join(plan.notes)
    print(f"{prefix}{joined}", file=sys.stderr)


def shrink_plan_for_oom(plan: ComputePlan) -> ComputePlan:
    """One step tighter after OOM."""
    torch = require_torch()
    notes = list(plan.notes) + ["OOM retry"]
    dtype = plan.dtype
    prec = dict(plan.prec_flags)
    if dtype == torch.float32:
        dtype = torch.float16 if not plan.machine.bf16 else torch.bfloat16
        prec = {"fp16": dtype == torch.float16, "bf16": dtype == torch.bfloat16}
        notes.append("dtype→half")
    elif dtype == torch.bfloat16:
        # last resort smaller memory footprint sometimes with fp16 + ckpt already on
        pass
    batch = 1
    ckpt = True
    max_seq = plan.max_seq_len
    if max_seq and max_seq > 256:
        max_seq = max(256, max_seq // 2)
        notes.append(f"max_seq_len→{max_seq}")
    accum = max(plan.grad_accum, 4)
    return ComputePlan(
        dtype=dtype,
        prec_flags=prec,
        batch_size=batch,
        grad_accum=accum,
        gradient_checkpointing=ckpt,
        max_seq_len=max_seq,
        machine=plan.machine,
        notes=notes,
    )


def is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return (
        "out of memory" in msg
        or "oom" in msg
        or "cuda error: out of memory" in msg
        or "mps backend out of memory" in msg
    )


def cuda_alloc_hygiene() -> None:
    """Reduce fragmentation before a heavy train."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if device_kind() not in ("cuda", "rocm"):
        return
    torch = require_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


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
