"""Live observability log: artifacts/metrics.jsonl (one JSON object per line).

Same idea as W&B/MLflow streams, but the train folder is the source of truth.
CLI and web can both tail this file.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_state: dict[str, Any] = {
    "train": None,
    "run_id": None,
    "op": None,
    "t0": None,
    "step": -1,
}


def metrics_path(train: Path) -> Path:
    return train / "artifacts" / "metrics.jsonl"


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms() -> int | None:
    t0 = _state.get("t0")
    if t0 is None:
        return None
    return int((time.perf_counter() - float(t0)) * 1000)


def begin(train: Path, *, op: str, run_id: str | None = None, **meta: Any) -> str:
    """Start a metrics session for this process. Returns run_id."""
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    _state["train"] = Path(train)
    _state["run_id"] = rid
    _state["op"] = op
    _state["t0"] = time.perf_counter()
    _state["step"] = -1
    emit(
        "start",
        op=op,
        run_id=rid,
        **{k: v for k, v in meta.items() if v is not None},
    )
    return rid


def end(**meta: Any) -> None:
    emit(
        "end",
        **{k: v for k, v in meta.items() if v is not None},
    )
    _state["train"] = None
    _state["run_id"] = None
    _state["op"] = None
    _state["t0"] = None
    _state["step"] = -1


def emit(event: str, **fields: Any) -> None:
    """Append one observability record. Safe no-op if begin() was never called."""
    train = _state.get("train")
    if train is None:
        return
    train = Path(train)
    body: dict[str, Any] = {
        "ts": _iso(),
        "event": event,
    }
    rid = _state.get("run_id")
    op = _state.get("op")
    if rid:
        body["run_id"] = rid
    if op:
        body["op"] = op
    elapsed = _elapsed_ms()
    if elapsed is not None:
        body["elapsed_ms"] = elapsed
    for k, v in fields.items():
        if v is None:
            continue
        if k in body and k not in ("event",):
            continue
        body[k] = v
    path = metrics_path(train)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(body, default=_json_default) + "\n")
        f.flush()


def step(step: int | None = None, **fields: Any) -> None:
    """Log a training step (loss, lr, …). Auto-increments step if omitted."""
    if step is None:
        _state["step"] = int(_state.get("step") or -1) + 1
        step = int(_state["step"])
    else:
        _state["step"] = int(step)
    emit("step", step=int(step), **fields)


def event(name: str, **fields: Any) -> None:
    """Free-form named event (eval.probe, serve.token, heartbeat, …)."""
    emit(name, **fields)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def model_summary(model: dict) -> dict[str, Any]:
    """Flat, loggable fields from a checkpoint dict (skip giant weight matrices)."""
    skip = {
        "tok",
        "wout",
        "wcls",
        "block",
        "blocks",
        "W_in",
        "W_out",
        "A",
        "B",
        "coef",
        "weights",
        "trees",
        "itos",
        "stoi",
        "tokenizer",
        "merges",
        "vocab",
    }
    out: dict[str, Any] = {}
    for k, v in model.items():
        if k in skip or k.startswith("_"):
            continue
        if isinstance(v, (list, dict)) and k not in ("features", "classes", "probes"):
            # skip nested weight-like blobs
            if isinstance(v, list) and v and isinstance(v[0], (list, float, int)):
                if k not in ("features", "classes"):
                    continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = v
            out[k + "_n"] = len(v)
    return out
