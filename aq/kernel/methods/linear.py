"""Tabular methods via scikit-learn. Recipe-only — no hand-rolled solvers."""

from __future__ import annotations

from pathlib import Path

from backends.sklearn_tab import fit_estimator, load_xy, predict as sk_predict, evaluate as sk_evaluate

# engine imports load_xy from methods.linear
__all__ = ["fit", "predict", "load_xy", "write_inspect", "evaluate"]


def fit(src: Path, rec: dict) -> dict:
    return fit_estimator(src, rec, "linear")


def predict(model: dict, X: list) -> list:
    return sk_predict(model, X)


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return sk_evaluate(model, src, rec)


def write_inspect(train: Path, model: dict) -> str:
    lines = [
        "# inspect",
        "",
        f"backend: {model.get('backend')}",
        f"kind: {model.get('kind')}",
        f"features: {', '.join(model.get('features') or [])}",
        f"bias: {model.get('bias')}",
        f"weights: {model.get('weights')}",
        f"train_loss: {model.get('train_loss')}",
        "",
    ]
    rel = "artifacts/inspect.md"
    (train / rel).write_text("\n".join(lines), encoding="utf-8")
    return rel
