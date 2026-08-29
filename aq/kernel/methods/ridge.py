"""Ridge via scikit-learn."""

from __future__ import annotations

from pathlib import Path

from backends.sklearn_tab import fit_estimator, evaluate as sk_evaluate, predict as sk_predict
from methods.linear import write_inspect


def fit(src: Path, rec: dict) -> dict:
    return fit_estimator(src, rec, "ridge")


def predict(model: dict, X: list) -> list:
    return sk_predict(model, X)


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return sk_evaluate(model, src, rec)
