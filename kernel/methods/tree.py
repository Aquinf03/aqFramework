"""CART-style regression tree. Stdlib. Splits on one feature at a time."""

from __future__ import annotations

from pathlib import Path

from methods.linear import _num, load_xy


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _best_split(X: list[list[float]], y: list[float]) -> tuple[int, float] | None:
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
            left = [y[i] for i in range(n) if X[i][j] <= thr]
            right = [y[i] for i in range(n) if X[i][j] > thr]
            if len(left) < 1 or len(right) < 1:
                continue
            lm, rm = _mean(left), _mean(right)
            sse = sum((v - lm) ** 2 for v in left) + sum((v - rm) ** 2 for v in right)
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best = (j, thr)
    return best


def _build(X: list[list[float]], y: list[float], depth: int) -> dict:
    if depth <= 0 or len(X) < 2:
        return {"leaf": _mean(y)}
    split = _best_split(X, y)
    if split is None:
        return {"leaf": _mean(y)}
    j, thr = split
    left_i = [i for i in range(len(X)) if X[i][j] <= thr]
    right_i = [i for i in range(len(X)) if X[i][j] > thr]
    if not left_i or not right_i:
        return {"leaf": _mean(y)}
    m = _mean(y)
    parent = sum((v - m) ** 2 for v in y)
    lm = _mean([y[i] for i in left_i])
    rm = _mean([y[i] for i in right_i])
    child = sum((y[i] - lm) ** 2 for i in left_i) + sum((y[i] - rm) ** 2 for i in right_i)
    if parent - child < 1e-12:
        return {"leaf": m}
    return {
        "feat": j,
        "thr": thr,
        "left": _build([X[i] for i in left_i], [y[i] for i in left_i], depth - 1),
        "right": _build([X[i] for i in right_i], [y[i] for i in right_i], depth - 1),
    }


def _walk(node: dict, row: list[float]) -> float:
    if "leaf" in node:
        return float(node["leaf"])
    if row[node["feat"]] <= node["thr"]:
        return _walk(node["left"], row)
    return _walk(node["right"], row)


def _fmt(node: dict, feats: list[str], indent: str) -> list[str]:
    if "leaf" in node:
        return [f"{indent}leaf: {node['leaf']}"]
    name = feats[node["feat"]] if node["feat"] < len(feats) else str(node["feat"])
    lines = [f"{indent}if {name} <= {node['thr']}"]
    lines.extend(_fmt(node["left"], feats, indent + "  "))
    lines.append(f"{indent}else")
    lines.extend(_fmt(node["right"], feats, indent + "  "))
    return lines


def fit(src: Path, rec: dict) -> dict:
    data = rec.get("data") or {}
    target = str(data.get("target") or "")
    depth = rec.get("depth")
    if depth is None:
        depth = (rec.get("tree") or {}).get("depth") if isinstance(rec.get("tree"), dict) else 3
    depth = int(depth)
    if depth < 1:
        raise SystemExit("tree depth must be >= 1")
    feats, X, y_raw = load_xy(src, target)
    y: list[float] = []
    for v in y_raw:
        n = _num(str(v))
        if n is None:
            raise SystemExit("tree expects a numeric target (regression)")
        y.append(n)
    root = _build(X, y, depth)
    return {
        "kind": "tree",
        "task": "regression",
        "features": feats,
        "depth": depth,
        "tree": root,
    }


def write_inspect(train: Path, model: dict) -> str:
    feats = model.get("features") or []
    lines = ["# tree", "", f"depth: {model.get('depth')}", ""]
    lines.extend(_fmt(model.get("tree") or {"leaf": 0}, feats, ""))
    lines.append("")
    rel = "artifacts/inspect.md"
    dest = train / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return rel


def predict_row(model: dict, row: list[float]) -> float:
    return _walk(model["tree"], row)


def predict(model: dict, X: list[list[float]]) -> list:
    return [predict_row(model, x) for x in X]
