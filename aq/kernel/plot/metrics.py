"""Loss / lr curves and fallbacks from artifacts/metrics.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from plot.render import apply_theme, save_fig


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _plot_steps(out_dir: Path, *, fmt: str, dpi: int, rows: list[dict]) -> Path | None:
    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    for row in rows:
        if row.get("event") != "step":
            continue
        step = row.get("step")
        loss = row.get("loss")
        if step is None or loss is None:
            continue
        try:
            steps.append(int(step))
            losses.append(float(loss))
        except (TypeError, ValueError):
            continue
        lr = row.get("lr")
        if lr is not None:
            try:
                lrs.append(float(lr))
            except (TypeError, ValueError):
                lrs.append(float("nan"))

    if not steps:
        return None

    apply_theme()
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(steps, losses, color="#2563eb", label="loss", marker="o", markersize=3)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_title("training loss")
    ax1.grid(True)

    if lrs and len(lrs) == len(steps):
        ax2 = ax1.twinx()
        ax2.plot(steps, lrs, color="#16a34a", label="lr", linestyle="--", alpha=0.85)
        ax2.set_ylabel("lr")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    out = out_dir / f"loss.{fmt}"
    save_fig(out, dpi=dpi)
    return out


def _plot_eval_scores(out_dir: Path, *, fmt: str, dpi: int, rows: list[dict]) -> Path | None:
    points: list[tuple[int, float, str]] = []
    for i, row in enumerate(rows):
        if row.get("event") not in ("eval.probe", "end"):
            continue
        if row.get("event") == "end" and row.get("op") != "eval":
            continue
        score = row.get("score")
        if score is None:
            continue
        try:
            val = float(score)
        except (TypeError, ValueError):
            continue
        metric = str(row.get("metric") or "score")
        points.append((len(points) + 1, val, metric))

    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    metric = points[-1][2]

    apply_theme()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color="#7c3aed", marker="o")
    ax.set_xlabel("eval #")
    ax.set_ylabel(metric)
    ax.set_title("eval scores")
    ax.grid(True)

    out = out_dir / f"eval-scores.{fmt}"
    save_fig(out, dpi=dpi)
    return out


def _plot_train_elapsed(out_dir: Path, *, fmt: str, dpi: int, rows: list[dict]) -> Path | None:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        if row.get("event") != "end" or row.get("op") != "train":
            continue
        ms = row.get("elapsed_ms")
        if ms is None:
            continue
        try:
            xs.append(len(xs) + 1)
            ys.append(float(ms))
        except (TypeError, ValueError):
            continue

    if not xs:
        return None

    apply_theme()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs, ys, color="#2563eb")
    ax.set_xlabel("train run #")
    ax.set_ylabel("elapsed_ms")
    ax.set_title("train duration")
    ax.grid(True, axis="y")

    out = out_dir / f"train-duration.{fmt}"
    save_fig(out, dpi=dpi)
    return out


def plot_metrics(train: Path, out_dir: Path, *, fmt: str, dpi: int) -> Path | None:
    path = train / "artifacts" / "metrics.jsonl"
    if not path.is_file():
        return None

    rows = _read_rows(path)
    for fn in (_plot_steps, _plot_eval_scores, _plot_train_elapsed):
        out = fn(out_dir, fmt=fmt, dpi=dpi, rows=rows)
        if out is not None:
            return out
    return None
