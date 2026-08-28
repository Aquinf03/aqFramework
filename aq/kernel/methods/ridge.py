"""Ridge: linear regression with L2 penalty. Stdlib. Does not zero weights."""

from __future__ import annotations

from pathlib import Path

from methods.linear import _gauss, _num, load_xy


def _ridge(X: list[list[float]], y: list[float], lam: float) -> tuple[list[float], float]:
    p = len(X[0])
    A = [[0.0] * (p + 1) for _ in range(p + 1)]
    b = [0.0] * (p + 1)
    for i in range(len(X)):
        row = X[i] + [1.0]
        for a in range(p + 1):
            b[a] += row[a] * y[i]
            for c in range(p + 1):
                A[a][c] += row[a] * row[c]
    for j in range(p):
        A[j][j] += lam
    coef = _gauss(A, b)
    return coef[:-1], coef[-1]


def fit(src: Path, rec: dict) -> dict:
    data = rec.get("data") or {}
    target = str(data.get("target") or "")
    lam = rec.get("lambda")
    if lam is None:
        lam = (rec.get("penalty") or {}).get("lambda") if isinstance(rec.get("penalty"), dict) else 1
    lam = float(lam)
    if lam < 0:
        raise SystemExit("ridge lambda must be >= 0")
    feats, X, y_raw = load_xy(src, target)
    y: list[float] = []
    for v in y_raw:
        n = _num(str(v))
        if n is None:
            raise SystemExit("ridge expects a numeric target")
        y.append(n)
    w, b = _ridge(X, y, lam)
    return {
        "kind": "ridge",
        "task": "regression",
        "features": feats,
        "weights": w,
        "bias": b,
        "lambda": lam,
    }


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    weights = model.get("weights") or []
    lines = [
        "# ridge",
        "",
        f"lambda: {model.get('lambda')}",
        f"intercept: {model.get('bias')}",
        "",
    ]
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
