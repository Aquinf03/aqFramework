"""Ordinary least squares. Stdlib only. Enough for aq train to be real."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def _rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_xy(csv_path: Path, target: str) -> tuple[list[str], list[list[float]], list]:
    rows = _rows(csv_path)
    if not rows:
        raise SystemExit(f"empty csv: {csv_path}")
    if target not in rows[0]:
        raise SystemExit(f"target {target!r} not in csv")
    feats = [k for k in rows[0] if k != target and all(_num(r[k]) is not None for r in rows)]
    if not feats:
        raise SystemExit("no numeric feature columns")
    X = [[float(r[k]) for k in feats] for r in rows]
    y = [r[target] for r in rows]
    return feats, X, y


def _is_class(y: list, metric: str) -> bool:
    if metric in ("mse", "rmse", "mae"):
        return False
    if metric == "accuracy":
        return True
    nums = [_num(str(v)) for v in y]
    if any(n is None for n in nums):
        return True
    uniq = {n for n in nums}
    return len(uniq) <= 20 and all(float(n).is_integer() for n in uniq)


def _gauss(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-12:
            raise SystemExit("singular design matrix")
        f = M[i][i]
        M[i] = [x / f for x in M[i]]
        for r in range(n):
            if r == i:
                continue
            g = M[r][i]
            M[r] = [M[r][c] - g * M[i][c] for c in range(n + 1)]
    return [row[-1] for row in M]


def _ols(X: list[list[float]], y: list[float]) -> tuple[list[float], float]:
    n = len(X)
    p = len(X[0])
    # columns: features + bias
    A = [[0.0] * (p + 1) for _ in range(p + 1)]
    b = [0.0] * (p + 1)
    for i in range(n):
        row = X[i] + [1.0]
        for a in range(p + 1):
            b[a] += row[a] * y[i]
            for c in range(p + 1):
                A[a][c] += row[a] * row[c]
    coef = _gauss(A, b)
    return coef[:-1], coef[-1]


def fit(csv_path: Path, target: str, metric: str) -> dict:
    feats, X, y_raw = load_xy(csv_path, target)
    y: list[float] = []
    for v in y_raw:
        n = _num(str(v))
        if n is None:
            raise SystemExit("linear expects a numeric target (regression). logistic is a later family.")
        y.append(n)
    w, b = _ols(X, y)
    return {
        "kind": "linear",
        "task": "regression",
        "features": feats,
        "weights": w,
        "bias": b,
    }


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    weights = model.get("weights") or []
    bias = model.get("bias")
    lines = ["# linear", "", f"intercept: {bias}", ""]
    for f, w in zip(feats, weights):
        lines.append(f"{f}: {w}")
    lines.append("")
    rel = "artifacts/inspect.md"
    dest = train / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return rel


def predict_row(model: dict, row: list[float]) -> float:
    return sum(a * b for a, b in zip(model["weights"], row)) + float(model["bias"])


def predict(model: dict, X: list[list[float]]) -> list:
    return [predict_row(model, x) for x in X]
