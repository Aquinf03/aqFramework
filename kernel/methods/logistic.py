"""Binary logistic regression. Stdlib only. Probability of class 1 via a sigmoid."""

from __future__ import annotations

from math import exp
from pathlib import Path

from methods.linear import load_xy


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + exp(-z))
    e = exp(z)
    return e / (1.0 + e)


def _encode(y_raw: list) -> tuple[list[float], str, str]:
    names = []
    seen: dict[str, None] = {}
    for v in y_raw:
        s = str(v)
        if s not in seen:
            seen[s] = None
            names.append(s)
    if len(names) != 2:
        raise SystemExit(f"logistic needs exactly two classes, got {names}")
    prefer = ("0", "false", "no", "neg", "negative")
    neg = names[0]
    pos = names[1]
    if names[1].lower() in prefer and names[0].lower() not in prefer:
        neg, pos = names[1], names[0]
    elif names[0].lower() in prefer:
        neg, pos = names[0], names[1]
    y = [0.0 if str(v) == neg else 1.0 for v in y_raw]
    return y, neg, pos


def fit(csv_path: Path, target: str, metric: str) -> dict:
    feats, X, y_raw = load_xy(csv_path, target)
    y, neg, pos = _encode(y_raw)
    p = len(X[0])
    w = [0.0] * p
    b = 0.0
    lr = 0.2
    n = len(X)
    for _ in range(4000):
        dw = [0.0] * p
        db = 0.0
        for i in range(n):
            z = sum(w[j] * X[i][j] for j in range(p)) + b
            err = _sigmoid(z) - y[i]
            db += err
            for j in range(p):
                dw[j] += err * X[i][j]
        b -= lr * db / n
        for j in range(p):
            w[j] -= lr * dw[j] / n
    return {
        "kind": "logistic",
        "task": "classification",
        "features": feats,
        "weights": w,
        "bias": b,
        "classes": [neg, pos],
    }


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    weights = model.get("weights") or []
    bias = model.get("bias")
    classes = model.get("classes") or []
    lines = [
        "# logistic",
        "",
        f"classes: {classes[0]} (0), {classes[1]} (1)" if len(classes) == 2 else "",
        f"intercept: {bias}",
        "",
    ]
    for f, wi in zip(feats, weights):
        lines.append(f"{f}: {wi}")
    lines.append("")
    rel = "artifacts/inspect.md"
    dest = train / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return rel


def predict_row(model: dict, row: list[float]) -> str:
    z = sum(a * b for a, b in zip(model["weights"], row)) + float(model["bias"])
    classes = model["classes"]
    return str(classes[1] if _sigmoid(z) >= 0.5 else classes[0])


def predict(model: dict, X: list[list[float]]) -> list:
    return [predict_row(model, x) for x in X]
