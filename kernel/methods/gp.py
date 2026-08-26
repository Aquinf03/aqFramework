"""Gaussian process regression. Stdlib. RBF kernel, exact posterior mean."""

from __future__ import annotations

import math
from pathlib import Path

from methods.linear import _num, load_xy


def _opt(rec: dict, key: str, default):
    v = rec.get(key)
    if v is None:
        nested = rec.get("gp")
        if isinstance(nested, dict):
            v = nested.get(key)
    return default if v is None else v


def _rbf(a: list[float], b: list[float], ell: float, sf2: float) -> float:
    d2 = sum((u - v) ** 2 for u, v in zip(a, b))
    return sf2 * math.exp(-d2 / (2.0 * ell * ell))


def _kvec(X: list[list[float]], x: list[float], ell: float, sf2: float) -> list[float]:
    return [_rbf(row, x, ell, sf2) for row in X]


def _chol(A: list[list[float]]) -> list[list[float]]:
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                v = A[i][i] - s
                if v <= 1e-18:
                    raise ValueError("not SPD")
                L[i][j] = math.sqrt(v)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _chol_solve(L: list[list[float]], b: list[float]) -> list[float]:
    n = len(L)
    z = [0.0] * n
    for i in range(n):
        z[i] = (b[i] - sum(L[i][k] * z[k] for k in range(i))) / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def _ky(X: list[list[float]], ell: float, sf2: float, noise: float) -> list[list[float]]:
    n = len(X)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            v = _rbf(X[i], X[j], ell, sf2)
            if i == j:
                v += noise
            K[i][j] = K[j][i] = v
    return K


def _lml(y: list[float], L: list[list[float]], alpha: list[float]) -> float:
    n = len(y)
    yy = sum(a * b for a, b in zip(y, alpha))
    logdet = 2.0 * sum(math.log(L[i][i]) for i in range(n))
    return -0.5 * (yy + logdet + n * math.log(2.0 * math.pi))


def _fit_one(X: list[list[float]], y: list[float], ell: float, sf2: float, noise: float):
    K = _ky(X, ell, sf2, noise)
    L = _chol(K)
    alpha = _chol_solve(L, y)
    return L, alpha, _lml(y, L, alpha)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 1.0
    m = sum(xs) / len(xs)
    v = sum((a - m) ** 2 for a in xs) / len(xs)
    return math.sqrt(v) if v > 1e-12 else 1.0


def fit(src: Path, rec: dict) -> dict:
    data = rec.get("data") or {}
    target = str(data.get("target") or "")
    kernel = str(_opt(rec, "kernel", "rbf")).lower()
    if kernel != "rbf":
        raise SystemExit("gp kernel must be rbf")
    ell_in = _opt(rec, "lengthscale", None)
    sf_in = _opt(rec, "signal", None)
    noise = float(_opt(rec, "noise", 1e-5))
    if noise < 0:
        raise SystemExit("gp noise must be >= 0")
    feats, X, y_raw = load_xy(src, target)
    y: list[float] = []
    for v in y_raw:
        n = _num(str(v))
        if n is None:
            raise SystemExit("gp expects a numeric target")
        y.append(n)
    mu = sum(y) / len(y)
    yc = [v - mu for v in y]
    sf2 = (float(sf_in) ** 2) if sf_in is not None else _std(yc) ** 2
    if sf2 <= 0:
        sf2 = 1.0
    if ell_in is not None:
        ell = float(ell_in)
        if ell <= 0:
            raise SystemExit("gp lengthscale must be > 0")
        L, alpha, lml = _fit_one(X, yc, ell, sf2, noise)
    else:
        best = None
        for cand in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0):
            try:
                L, alpha, lml = _fit_one(X, yc, cand, sf2, noise)
            except ValueError:
                continue
            if best is None or lml > best[0]:
                best = (lml, cand, L, alpha)
        if best is None:
            raise SystemExit("gp kernel matrix was not SPD")
        lml, ell, L, alpha = best
    return {
        "kind": "gp",
        "task": "regression",
        "features": feats,
        "kernel": "rbf",
        "lengthscale": ell,
        "signal": math.sqrt(sf2),
        "noise": noise,
        "mean": mu,
        "X": X,
        "L": L,
        "alpha": alpha,
        "lml": lml,
    }


def _post(model: dict, row: list[float]) -> tuple[float, float]:
    ell = float(model["lengthscale"])
    sf2 = float(model["signal"]) ** 2
    kstar = _kvec(model["X"], row, ell, sf2)
    mean = float(model["mean"]) + sum(a * k for a, k in zip(model["alpha"], kstar))
    v = _chol_solve(model["L"], kstar)
    var = sf2 - sum(a * b for a, b in zip(kstar, v))
    if var < 0:
        var = 0.0
    return mean, math.sqrt(var)


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    X = model.get("X") or []
    lines = [
        "# gp",
        "",
        f"kernel: {model.get('kernel')}",
        f"lengthscale: {model.get('lengthscale')}",
        f"signal: {model.get('signal')}",
        f"noise: {model.get('noise')}",
        f"mean: {model.get('mean')}",
        "",
        "train posterior (mean ± std):",
    ]
    for row in X:
        m, s = _post(model, row)
        xs = " ".join(f"{f}={v}" for f, v in zip(feats, row))
        lines.append(f"  {xs}  {m} ± {s}")
    lines.append("")
    rel = "artifacts/inspect.md"
    dest = train / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return rel


def predict_row(model: dict, row: list[float]) -> float:
    return _post(model, row)[0]


def predict(model: dict, X: list[list[float]]) -> list:
    return [predict_row(model, x) for x in X]
