"""Gradient boosting of shallow trees. Stdlib only. Tabular method next to linear."""

from __future__ import annotations

from pathlib import Path

from linear import _is_class, load_xy


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _best_stump(X: list[list[float]], res: list[float]) -> dict | None:
    n = len(X)
    if n < 4:
        return None
    p = len(X[0])
    best = None
    best_sse = None
    for j in range(p):
        vals = sorted({row[j] for row in X})
        if len(vals) < 2:
            continue
        for k in range(len(vals) - 1):
            thr = 0.5 * (vals[k] + vals[k + 1])
            left = [res[i] for i in range(n) if X[i][j] <= thr]
            right = [res[i] for i in range(n) if X[i][j] > thr]
            if len(left) < 2 or len(right) < 2:
                continue
            lm, rm = _mean(left), _mean(right)
            sse = sum((r - lm) ** 2 for r in left) + sum((r - rm) ** 2 for r in right)
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best = {"feat": j, "thr": thr, "left": {"leaf": lm}, "right": {"leaf": rm}}
    return best


def _leaf(res: list[float]) -> dict:
    return {"leaf": _mean(res)}


def _split_idx(X: list[list[float]], feat: int, thr: float) -> tuple[list[int], list[int]]:
    left, right = [], []
    for i, row in enumerate(X):
        (left if row[feat] <= thr else right).append(i)
    return left, right


def _build_tree(X: list[list[float]], res: list[float], depth: int) -> dict:
    if depth <= 0 or len(X) < 4:
        return _leaf(res)
    stump = _best_stump(X, res)
    if stump is None:
        return _leaf(res)
    li, ri = _split_idx(X, stump["feat"], stump["thr"])
    Xl = [X[i] for i in li]
    Xr = [X[i] for i in ri]
    rl = [res[i] for i in li]
    rr = [res[i] for i in ri]
    return {
        "feat": stump["feat"],
        "thr": stump["thr"],
        "left": _build_tree(Xl, rl, depth - 1),
        "right": _build_tree(Xr, rr, depth - 1),
    }


def _tree_pred(node: dict, row: list[float]) -> float:
    if "leaf" in node:
        return float(node["leaf"])
    if row[node["feat"]] <= node["thr"]:
        return _tree_pred(node["left"], row)
    return _tree_pred(node["right"], row)


def _boost(X: list[list[float]], y: list[float], trees: int, depth: int, lr: float) -> dict:
    init = _mean(y)
    pred = [init] * len(X)
    fitted = []
    for _ in range(trees):
        res = [y[i] - pred[i] for i in range(len(X))]
        tree = _build_tree(X, res, depth)
        fitted.append(tree)
        for i, row in enumerate(X):
            pred[i] += lr * _tree_pred(tree, row)
    return {"init": init, "trees": fitted, "lr": lr}


def _score_boost(part: dict, row: list[float]) -> float:
    s = float(part["init"])
    lr = float(part["lr"])
    for t in part["trees"]:
        s += lr * _tree_pred(t, row)
    return s


def fit(csv_path: Path, target: str, metric: str) -> dict:
    feats, X, y_raw = load_xy(csv_path, target)
    trees, depth, lr = 20, 2, 0.1
    if _is_class(y_raw, metric):
        classes = sorted({str(v) for v in y_raw})
        parts = []
        for c in classes:
            yk = [1.0 if str(v) == c else 0.0 for v in y_raw]
            parts.append(_boost(X, yk, trees, depth, lr))
        return {
            "kind": "boosting",
            "task": "classification",
            "features": feats,
            "classes": classes,
            "parts": parts,
        }
    y = [float(v) for v in y_raw]
    part = _boost(X, y, trees, depth, lr)
    return {
        "kind": "boosting",
        "task": "regression",
        "features": feats,
        **part,
    }


def predict_row(model: dict, row: list[float]) -> float | str:
    if model["task"] == "classification":
        scores = [_score_boost(p, row) for p in model["parts"]]
        return model["classes"][max(range(len(scores)), key=lambda i: scores[i])]
    return _score_boost(model, row)


def predict(model: dict, X: list[list[float]]) -> list:
    return [predict_row(model, x) for x in X]
