"""Logistic regression via scikit-learn."""

from __future__ import annotations

from pathlib import Path

from backends.sklearn_tab import fit_estimator, predict as sk_predict, evaluate as sk_evaluate, generate as sk_generate


def fit(src: Path, rec: dict) -> dict:
    return fit_estimator(src, rec, "logistic")


def predict(model: dict, X: list) -> list:
    return sk_predict(model, X)


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return sk_evaluate(model, src, rec)


def generate(model, prompt, rec, max_tokens=None, temperature=None):
    return sk_generate(model, prompt, rec, max_tokens=max_tokens, temperature=temperature)


def write_inspect(train: Path, model: dict) -> str:
    lines = [
        "# inspect",
        "",
        f"backend: {model.get('backend')}",
        f"classes: {model.get('classes')}",
        f"weights: {model.get('weights')}",
        f"bias: {model.get('bias')}",
        "",
    ]
    rel = "artifacts/inspect.md"
    (train / rel).write_text("\n".join(lines), encoding="utf-8")
    return rel
