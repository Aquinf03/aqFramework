"""Opt-in training watches. Off unless recipe sets guard.safety / guard.leak.

guard:
  safety: true          # NaN/Inf + loss blow-up → stop the job
  leak: true            # train↔eval overlap → stop before / during work
  max_loss: null        # optional absolute loss ceiling
  blowup_factor: 8      # stop if loss > best * factor (after warmup)
  blowup_warmup: 2      # steps before blow-up check
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


class GuardAbort(SystemExit):
    """Fail-closed stop. Same as SystemExit so aq jobs die with a clear message."""


def _truthy(v: Any) -> bool:
    return v is True or v in ("true", "yes", "on", "safety", "1")


def parse_guard(rec: dict) -> dict[str, Any]:
    raw = rec.get("guard")
    if not isinstance(raw, dict):
        return {
            "safety": False,
            "leak": False,
            "max_loss": None,
            "blowup_factor": 8.0,
            "blowup_warmup": 2,
        }
    mode = raw.get("mode")
    safety = _truthy(raw.get("safety")) or str(mode or "").lower() in ("safety", "critical", "safe")
    leak = _truthy(raw.get("leak"))
    max_loss = raw.get("max_loss")
    if max_loss is not None:
        max_loss = float(max_loss)
    factor = raw.get("blowup_factor")
    factor = 8.0 if factor is None else float(factor)
    warmup = raw.get("blowup_warmup")
    warmup = 2 if warmup is None else int(warmup)
    return {
        "safety": safety,
        "leak": leak,
        "max_loss": max_loss,
        "blowup_factor": max(factor, 1.0),
        "blowup_warmup": max(warmup, 0),
    }


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


class SafetyWatch:
    """Stateful loss watcher for one train session."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.best: float | None = None
        self.enabled = bool(cfg.get("safety"))

    def check_step(self, *, step: int, loss: Any = None, **_extra: Any) -> None:
        if not self.enabled:
            return
        if loss is None:
            return
        if not _finite(loss):
            raise GuardAbort(
                f"guard.safety: non-finite loss at step {step} ({loss!r}). "
                "Training stopped to save compute."
            )
        v = float(loss)
        max_loss = self.cfg.get("max_loss")
        if max_loss is not None and v > float(max_loss):
            raise GuardAbort(
                f"guard.safety: loss {v} exceeded max_loss {max_loss} at step {step}. "
                "Training stopped to save compute."
            )
        if self.best is None or v < self.best:
            self.best = v
            return
        warmup = int(self.cfg.get("blowup_warmup") or 0)
        if step < warmup:
            return
        factor = float(self.cfg.get("blowup_factor") or 8.0)
        if self.best > 0 and v > self.best * factor:
            raise GuardAbort(
                f"guard.safety: loss blew up at step {step} "
                f"(loss={v}, best={self.best}, factor={factor}). "
                "Training stopped to save compute."
            )
        # also catch huge absolute jumps from near-zero best
        if self.best <= 0 and v > 1.0 and v > abs(self.best) + 10.0:
            raise GuardAbort(
                f"guard.safety: loss blew up at step {step} "
                f"(loss={v}, best={self.best}). Training stopped to save compute."
            )


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    if path.suffix == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fingerprint(row: dict, fields: list[str] | None) -> str:
    if fields:
        keys = [k for k in fields if k in row]
    else:
        keys = sorted(row.keys())
    parts = []
    for k in keys:
        parts.append(k + "=" + str(row.get(k, "")).strip())
    return "\x1f".join(parts)


def _key_fields(rec: dict) -> list[str] | None:
    data = rec.get("data") or {}
    keys = []
    for k in ("target", "text", "prompt", "completion", "instruction", "output", "src", "tgt"):
        v = data.get(k)
        if isinstance(v, str) and v:
            keys.append(v)
    return keys or None


def probe_files(train: Path) -> list[Path]:
    d = train / "evals"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_file() and p.suffix in {".csv", ".jsonl"}:
            out.append(p)
    return out


def check_leak(train: Path, rec: dict) -> dict[str, Any] | None:
    """Return a leak report dict if overlap found, else None."""
    data = rec.get("data") or {}
    rel = data.get("path")
    if not rel:
        return None
    src = (train / str(rel)).resolve()
    if not src.is_file():
        return None
    probes = probe_files(train)
    if not probes:
        return None
    fields = _key_fields(rec)
    train_rows = _rows(src)
    train_fp = {_fingerprint(r, fields) for r in train_rows}
    if not train_fp:
        return None
    hits: list[dict[str, Any]] = []
    for probe in probes:
        n = 0
        for r in _rows(probe):
            if _fingerprint(r, fields) in train_fp:
                n += 1
        if n:
            try:
                prel = str(probe.relative_to(train))
            except ValueError:
                prel = str(probe)
            hits.append({"path": prel, "overlap": n})
    if not hits:
        return None
    return {
        "train": str(rel),
        "fields": fields,
        "hits": hits,
        "train_n": len(train_rows),
    }


def assert_no_leak(train: Path, rec: dict) -> None:
    cfg = parse_guard(rec)
    if not cfg.get("leak"):
        return
    report = check_leak(train, rec)
    if not report:
        return
    bits = ", ".join(f"{h['path']} ({h['overlap']} rows)" for h in report["hits"])
    raise GuardAbort(
        f"guard.leak: train data overlaps eval probes: {bits}. "
        "Fix the split or turn off guard.leak."
    )
