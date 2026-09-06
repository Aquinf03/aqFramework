"""Random forest via scikit-learn."""

from __future__ import annotations

from pathlib import Path

from backends.sklearn_tab import fit_estimator, evaluate as sk_evaluate, generate as sk_generate


def fit(src: Path, rec: dict) -> dict:
    return fit_estimator(src, rec, "forest")


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    imps = model.get("feature_importances") or []
    lines = ["# inspect", "", f"backend: {model.get('backend')}", f"trees: {model.get('trees')}", ""]
    for f, c in zip(feats, imps):
        lines.append(f"  {f}: {c}")
    lines.append("")
    rel = "artifacts/inspect.md"
    (train / rel).write_text("\n".join(lines), encoding="utf-8")
    return rel


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    return sk_evaluate(model, src, rec)


def generate(model, prompt, rec, max_tokens=None, temperature=None):
    return sk_generate(model, prompt, rec, max_tokens=max_tokens, temperature=temperature)
