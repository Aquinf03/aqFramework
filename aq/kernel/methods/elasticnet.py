"""Elastic Net: linear regression with L1 + L2. Stdlib. Mix of lasso and ridge."""

from __future__ import annotations

from pathlib import Path

from methods.linear import _num, load_xy


def _soft(z: float, lam: float) -> float:
    if z > lam:
        return z - lam
    if z < -lam:
        return z + lam
    return 0.0


def _elastic(X: list[list[float]], y: list[float], lam: float, l1: float) -> tuple[list[float], float]:
    n = len(X)
    p = len(X[0])
    w = [0.0] * p
    b = sum(y) / n
    r = [y[i] - b for i in range(n)]
    col2 = [sum(X[i][j] * X[i][j] for i in range(n)) / n for j in range(p)]
    l1 = min(1.0, max(0.0, l1))
    l2 = lam * (1.0 - l1)
    t1 = lam * l1
    for _ in range(8000):
        db = sum(r) / n
        b += db
        r = [r[i] - db for i in range(n)]
        for j in range(p):
            denom = col2[j] + l2
            if denom < 1e-18:
                continue
            rho = sum(X[i][j] * r[i] for i in range(n)) / n + w[j] * col2[j]
            nw = _soft(rho, t1) / denom
            dw = w[j] - nw
            if dw != 0:
                r = [r[i] + X[i][j] * dw for i in range(n)]
            w[j] = nw
    return w, b


def fit(src: Path, rec: dict) -> dict:
    data = rec.get("data") or {}
    target = str(data.get("target") or "")
    lam = rec.get("lambda")
    if lam is None:
        lam = (rec.get("penalty") or {}).get("lambda") if isinstance(rec.get("penalty"), dict) else 0.4
    lam = float(lam)
    l1 = rec.get("l1_ratio")
    if l1 is None:
        l1 = (rec.get("penalty") or {}).get("l1_ratio") if isinstance(rec.get("penalty"), dict) else 0.5
    l1 = float(l1)
    if lam < 0:
        raise SystemExit("elasticnet lambda must be >= 0")
    if not 0 <= l1 <= 1:
        raise SystemExit("elasticnet l1_ratio must be in [0, 1]")
    feats, X, y_raw = load_xy(src, target)
    y: list[float] = []
    for v in y_raw:
        n = _num(str(v))
        if n is None:
            raise SystemExit("elasticnet expects a numeric target")
        y.append(n)
    w, b = _elastic(X, y, lam, l1)
    return {
        "kind": "elasticnet",
        "task": "regression",
        "features": feats,
        "weights": w,
        "bias": b,
        "lambda": lam,
        "l1_ratio": l1,
    }


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    weights = model.get("weights") or []
    lines = [
        "# elasticnet",
        "",
        f"lambda: {model.get('lambda')}",
        f"l1_ratio: {model.get('l1_ratio')}",
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
