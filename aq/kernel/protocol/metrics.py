"""Live observability log: artifacts/metrics.jsonl (one JSON object per line).

Same idea as W&B/MLflow streams, but the train folder is the source of truth.
CLI and web can both tail this file.

When recipe guard.safety is on, step() also watches for NaN/Inf and loss blow-up
and raises GuardAbort (fail closed).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protocol.guard import GuardAbort, SafetyWatch, parse_guard

_state: dict[str, Any] = {
    "train": None,
    "run_id": None,
    "op": None,
    "t0": None,
    "step": -1,
    "steps": None,
    "guard": None,
    "watch": None,
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


def begin(
    train: Path,
    *,
    op: str,
    run_id: str | None = None,
    recipe: dict | None = None,
    **meta: Any,
) -> str:
    """Start a metrics session for this process. Returns run_id."""
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    cfg = parse_guard(recipe or {})
    _state["train"] = Path(train)
    _state["run_id"] = rid
    _state["op"] = op
    _state["t0"] = time.perf_counter()
    _state["step"] = -1
    _state["guard"] = cfg
    _state["watch"] = SafetyWatch(cfg) if cfg.get("safety") else None
    steps = meta.get("steps")
    if steps is None and recipe:
        steps = recipe.get("steps")
    if steps is not None:
        try:
            _state["steps"] = int(steps)
        except (TypeError, ValueError):
            _state["steps"] = None
    else:
        _state["steps"] = None
    emit(
        "start",
        op=op,
        run_id=rid,
        guard_safety=bool(cfg.get("safety")),
        guard_leak=bool(cfg.get("leak")),
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
    _state["steps"] = None
    _state["guard"] = None
    _state["watch"] = None


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
    _print_live(event, body)


def _fmt_num(v: Any) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.4e}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _print_live(event: str, body: dict[str, Any]) -> None:
    """Always show progress on stderr so `aq train` is live, not silent."""
    if event == "start":
        op = body.get("op") or "run"
        bits = [str(op)]
        if body.get("method"):
            bits.append(str(body["method"]))
        if body.get("family"):
            bits.append(str(body["family"]))
        flags = []
        if body.get("guard_safety"):
            flags.append("safety")
        if body.get("guard_leak"):
            flags.append("leak")
        if flags:
            bits.append("guard:" + "+".join(flags))
        print("  " + "  ".join(bits), file=sys.stderr, flush=True)
        return

    if event == "step":
        step_n = body.get("step", 0)
        total = body.get("steps") or _state.get("steps")
        if total is not None:
            label = f"step  {int(step_n):>4}/{int(total)}"
        else:
            label = f"step  {int(step_n):>4}"
        parts = [label]
        if body.get("loss") is not None:
            parts.append(f"loss  {_fmt_num(body['loss'])}")
        if body.get("lr") is not None:
            parts.append(f"lr  {_fmt_num(body['lr'])}")
        # extra scalar metrics (skip noise)
        skip = {"ts", "event", "run_id", "op", "elapsed_ms", "step", "steps", "loss", "lr", "demo"}
        for k, v in body.items():
            if k in skip or v is None:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                parts.append(f"{k}  {_fmt_num(v)}")
        if body.get("elapsed_ms") is not None:
            parts.append(f"{int(body['elapsed_ms'])}ms")
        print("  " + "   ".join(parts), file=sys.stderr, flush=True)
        return

    if event == "guard.abort":
        msg = body.get("message") or body.get("reason") or "aborted"
        print(f"  abort  {msg}", file=sys.stderr, flush=True)
        return

    if event == "end":
        parts = ["done"]
        if body.get("train_loss") is not None:
            parts.append(f"loss  {_fmt_num(body['train_loss'])}")
        if body.get("score") is not None:
            m = body.get("metric") or "score"
            parts.append(f"{m}  {_fmt_num(body['score'])}")
        if body.get("verdict"):
            parts.append(str(body["verdict"]))
        if body.get("elapsed_ms") is not None:
            parts.append(f"{int(body['elapsed_ms'])}ms")
        print("  " + "   ".join(parts), file=sys.stderr, flush=True)
        return

    if event == "eval.probe":
        path = body.get("path") or "probe"
        metric = body.get("metric") or "score"
        score = body.get("score")
        verdict = body.get("pass")
        v = "pass" if verdict is True else "fail" if verdict is False else "—"
        print(
            f"  probe  {path}   {metric}  {_fmt_num(score)}   {v}",
            file=sys.stderr,
            flush=True,
        )
        return

    if event == "error":
        print(f"  error  {body.get('error')}", file=sys.stderr, flush=True)


def step(step: int | None = None, **fields: Any) -> None:
    """Log a training step (loss, lr, …). Auto-increments step if omitted.

    If guard.safety is on, non-finite or exploding loss aborts the job.
    """
    if step is None:
        _state["step"] = int(_state.get("step") or -1) + 1
        step = int(_state["step"])
    else:
        _state["step"] = int(step)
    watch: SafetyWatch | None = _state.get("watch")
    if watch is not None:
        try:
            watch.check_step(step=int(step), **fields)
        except GuardAbort as e:
            emit(
                "guard.abort",
                reason="safety",
                step=int(step),
                message=str(e),
                loss=fields.get("loss"),
            )
            raise
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
            if isinstance(v, list) and v and isinstance(v[0], (list, float, int)):
                if k not in ("features", "classes"):
                    continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[k] = v
            out[k + "_n"] = len(v)
    return out
