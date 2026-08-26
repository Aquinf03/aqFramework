"""Gradient boosting of shallow trees. Stdlib. Fit residuals in sequence."""

from __future__ import annotations

from pathlib import Path

from methods.linear import _num, load_xy
from methods.tree import _build, _fmt, _walk


def _opt(rec: dict, key: str, default):
    v = rec.get(key)
    if v is None:
        nested = rec.get("boosting")
        if isinstance(nested, dict):
            v = nested.get(key)
    return default if v is None else v


def _count(node: dict, acc: list[int]) -> None:
    if "leaf" in node:
        return
    acc[node["feat"]] += 1
    _count(node["left"], acc)
    _count(node["right"], acc)


def fit(src: Path, rec: dict) -> dict:
    data = rec.get("data") or {}
    target = str(data.get("target") or "")
    n_trees = int(_opt(rec, "trees", 20))
    depth = int(_opt(rec, "depth", 1))
    lr = float(_opt(rec, "lr", 0.3))
    if n_trees < 1:
        raise SystemExit("boosting trees must be >= 1")
    if depth < 1:
        raise SystemExit("boosting depth must be >= 1")
    if lr <= 0:
        raise SystemExit("boosting lr must be > 0")
    feats, X, y_raw = load_xy(src, target)
    y: list[float] = []
    for v in y_raw:
        n = _num(str(v))
        if n is None:
            raise SystemExit("boosting expects a numeric target")
        y.append(n)
    init = sum(y) / len(y)
    pred = [init] * len(X)
    fitted = []
    for _ in range(n_trees):
        res = [y[i] - pred[i] for i in range(len(X))]
        tree = _build(X, res, depth)
        fitted.append(tree)
        for i, row in enumerate(X):
            pred[i] += lr * _walk(tree, row)
    return {
        "kind": "boosting",
        "task": "regression",
        "features": feats,
        "trees": n_trees,
        "depth": depth,
        "lr": lr,
        "init": init,
        "boost": fitted,
    }


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    trees = model.get("boost") or []
    counts = [0] * len(feats)
    for t in trees:
        _count(t, counts)
    lines = [
        "# boosting",
        "",
        f"trees: {model.get('trees')}",
        f"depth: {model.get('depth')}",
        f"lr: {model.get('lr')}",
        f"init: {model.get('init')}",
        "",
        "splits used:",
    ]
    for f, c in zip(feats, counts):
        lines.append(f"  {f}: {c}")
    lines.append("")
    show = min(3, len(trees))
    for i in range(show):
        lines.append(f"round {i + 1}")
        lines.extend(_fmt(trees[i], feats, "  "))
        lines.append("")
    rel = "artifacts/inspect.md"
    dest = train / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return rel


def predict_row(model: dict, row: list[float]) -> float:
    s = float(model["init"])
    lr = float(model["lr"])
    for t in model["boost"]:
        s += lr * _walk(t, row)
    return s


def predict(model: dict, X: list[list[float]]) -> list:
    return [predict_row(model, x) for x in X]
