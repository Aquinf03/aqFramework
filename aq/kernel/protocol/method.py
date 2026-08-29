"""Load a train or kernel method module. No registry. Filename is the name."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

RESERVED = {
    "connection",
    "method",
    "recipe",
    "record",
    "revision",
    "run",
    "step",
}


def load_method(train: Path, name: str):
    name = str(name).replace("-", "_")
    if not name or name in RESERVED or name.startswith("_"):
        raise SystemExit(f"bad method name: {name}")
    kernel = Path(__file__).resolve().parent.parent
    for p in (train / "methods" / f"{name}.py", kernel / "methods" / f"{name}.py"):
        if p.is_file():
            spec = importlib.util.spec_from_file_location(f"aq_method_{name}", p)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(
        f"no method {name} (add methods/{name}.py in this train, or kernel/methods/{name}.py)"
    )


def call_fit(mod, src: Path, rec: dict, train: Path | None = None) -> dict:
    if not hasattr(mod, "fit"):
        raise SystemExit("method module needs fit()")
    if train is not None:
        rec = {**rec, "_train": str(Path(train).resolve())}
    n = len(inspect.signature(mod.fit).parameters)
    if n >= 3:
        data = rec.get("data") or {}
        ev = rec.get("eval") or {}
        return mod.fit(src, str(data.get("target") or ""), str(ev.get("metric") or "mse"))
    model = mod.fit(src, rec)
    if not isinstance(model, dict):
        raise SystemExit("fit() must return a dict")
    return model
