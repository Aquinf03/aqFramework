"""Merge plot config: recipe.yaml plot: block, ~/.aq/config.json, CLI overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from protocol.recipe import parse_recipe


def _global_plot() -> dict[str, Any]:
    p = Path.home() / ".aq" / "config.json"
    if not p.is_file():
        return {}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    block = cfg.get("plot")
    return block if isinstance(block, dict) else {}


def _recipe_plot(train: Path) -> dict[str, Any]:
    recipe = train / "recipe.yaml"
    if not recipe.is_file():
        return {}
    rec = parse_recipe(recipe)
    block = rec.get("plot")
    return block if isinstance(block, dict) else {}


def resolve_plot_config(train: Path, req: dict[str, Any] | None = None) -> dict[str, Any]:
    req = req or {}
    merged: dict[str, Any] = {
        "format": "png",
        "dpi": 150,
        "out": "artifacts/plots",
        "charts": ["metrics", "jobs", "runs"],
        "auto": False,
    }
    for src in (_global_plot(), _recipe_plot(train)):
        for k, v in src.items():
            if v is not None:
                merged[k] = v
    for k in ("format", "dpi", "out", "kind", "auto"):
        if k in req and req[k] is not None:
            merged[k] = req[k]
    charts = merged.get("charts")
    if isinstance(charts, str):
        merged["charts"] = [c.strip() for c in charts.split(",") if c.strip()]
    elif charts is None:
        merged["charts"] = ["metrics", "jobs", "runs"]
    try:
        merged["dpi"] = int(merged.get("dpi") or 150)
    except (TypeError, ValueError):
        merged["dpi"] = 150
    fmt = str(merged.get("format") or "png").lower().lstrip(".")
    if fmt not in ("png", "svg", "pdf"):
        fmt = "png"
    merged["format"] = fmt
    return merged


def should_auto_plot(train: Path) -> bool:
    cfg = resolve_plot_config(train, {})
    return bool(cfg.get("auto"))
