"""Read train artifacts and write matplotlib charts under artifacts/plots/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plot.config import resolve_plot_config
from plot.jobs import plot_jobs
from plot.metrics import plot_metrics
from plot.runs import plot_runs

_RENDERERS = {
    "metrics": plot_metrics,
    "jobs": plot_jobs,
    "runs": plot_runs,
}


def do_plot(train: Path, req: dict[str, Any] | None = None) -> list[str]:
    req = req or {}
    import os

    mpl_dir = train / "artifacts" / ".matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    cfg = resolve_plot_config(train, req)
    fmt = str(cfg["format"])
    dpi = int(cfg["dpi"])
    kind = str(req.get("kind") or cfg.get("kind") or "all").lower()

    out_rel = str(req.get("out") or cfg.get("out") or "artifacts/plots")
    out_file = req.get("out_file")
    if out_file:
        out_dir = Path(out_file).parent
        single_name = Path(out_file).stem
    else:
        out_dir = train / out_rel
        single_name = None

    if kind == "all":
        charts = list(cfg.get("charts") or ["metrics", "jobs", "runs"])
    else:
        charts = [kind]

    written: list[Path] = []
    skipped: list[str] = []
    for name in charts:
        fn = _RENDERERS.get(name)
        if fn is None:
            skipped.append(f"unknown chart: {name}")
            continue
        path = fn(train, out_dir, fmt=fmt, dpi=dpi)
        if path is None:
            skipped.append(f"no data for {name}")
            continue
        if single_name and len(charts) == 1:
            dest = out_dir / f"{single_name}.{fmt}"
            if dest != path:
                dest.parent.mkdir(parents=True, exist_ok=True)
                path.replace(dest)
                path = dest
        written.append(path)

    if not written:
        hint = "; ".join(skipped) if skipped else "nothing to plot"
        raise SystemExit(f"plot: {hint}")

    lines = []
    for p in written:
        try:
            lines.append(str(p.relative_to(train)))
        except ValueError:
            lines.append(str(p))
    return lines
