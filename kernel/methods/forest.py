"""Random forest: bagged trees, random features per split. Stdlib."""

from __future__ import annotations

from random import Random
from pathlib import Path

from methods.linear import _num, load_xy
from methods.tree import _build, _walk


def _count(node: dict, acc: list[int]) -> None:
    if "leaf" in node:
        return
    acc[node["feat"]] += 1
    _count(node["left"], acc)
    _count(node["right"], acc)


def fit(src: Path, rec: dict) -> dict:
    data = rec.get("data") or {}
    target = str(data.get("target") or "")
    n_trees = int(rec.get("trees") or (rec.get("forest") or {}).get("trees") or 20)
    depth = int(rec.get("depth") or (rec.get("forest") or {}).get("depth") or 3)
    seed = int(rec.get("seed") or (rec.get("forest") or {}).get("seed") or 1)
    if n_trees < 1:
        raise SystemExit("forest trees must be >= 1")
    feats, X, y_raw = load_xy(src, target)
    y: list[float] = []
    for v in y_raw:
        n = _num(str(v))
        if n is None:
            raise SystemExit("forest expects a numeric target")
        y.append(n)
    n = len(X)
    p = len(X[0])
    mtry = rec.get("mtry")
    if mtry is None:
        mtry = (rec.get("forest") or {}).get("mtry") if isinstance(rec.get("forest"), dict) else None
    mtry = int(mtry) if mtry is not None else max(1, p // 2)
    rng = Random(seed)
    trees = []
    for _ in range(n_trees):
        idx = [rng.randrange(n) for _ in range(n)]
        Xs = [X[i] for i in idx]
        ys = [y[i] for i in idx]
        trees.append(_build(Xs, ys, depth, rng=rng, mtry=mtry))
    return {
        "kind": "forest",
        "task": "regression",
        "features": feats,
        "trees": n_trees,
        "depth": depth,
        "mtry": mtry,
        "seed": seed,
        "forest": trees,
    }


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    trees = model.get("forest") or []
    counts = [0] * len(feats)
    for t in trees:
        _count(t, counts)
    lines = [
        "# forest",
        "",
        f"trees: {model.get('trees')}",
        f"depth: {model.get('depth')}",
        f"mtry: {model.get('mtry')}",
        f"seed: {model.get('seed')}",
        "",
        "splits used:",
    ]
    for f, c in zip(feats, counts):
        lines.append(f"  {f}: {c}")
    lines.append("")
    rel = "artifacts/inspect.md"
    dest = train / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return rel


def predict_row(model: dict, row: list[float]) -> float:
    trees = model["forest"]
    return sum(_walk(t, row) for t in trees) / len(trees)


def predict(model: dict, X: list[list[float]]) -> list:
    return [predict_row(model, x) for x in X]
