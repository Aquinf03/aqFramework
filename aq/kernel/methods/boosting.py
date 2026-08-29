"""Gradient boosting via XGBoost / LightGBM / sklearn (recipe.library)."""

from __future__ import annotations

from pathlib import Path

from backends.sklearn_tab import fit_estimator, evaluate as sk_evaluate


def fit(src: Path, rec: dict) -> dict:
    return fit_estimator(src, rec, "boosting")


def write_inspect(train: Path, model: dict) -> str:
    lines = [
        "# inspect",
        "",
        f"backend: {model.get('backend')}",
        f"trees: {model.get('trees')}",
        f"depth: {model.get('depth')}",
        f"lr: {model.get('lr')}",
        f"train_loss: {model.get('train_loss')}",
        "",
    ]
    rel = "artifacts/inspect.md"
    (train / rel).write_text("\n".join(lines), encoding="utf-8")
    return rel


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return sk_evaluate(model, src, rec)
