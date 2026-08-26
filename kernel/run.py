#!/usr/bin/env python3
"""Kernel worker. aq writes artifacts/request.json. This writes artifacts/result.json.

Not a user CLI. One arg: the train directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from protocol.revision import hash_train
from engine.step import do_checkpoint, do_eval, do_train


# Invocation only. Spec is recipe.yaml, never this file.
REQ_KEYS = {"op", "snapshot", "ckpt", "keep", "probe"}


def dispatch(train: Path, req: dict) -> list[str]:
    req = {k: req[k] for k in REQ_KEYS if k in req}
    op = req.get("op")
    if op == "hash":
        return hash_train(train, bool(req.get("snapshot")))
    if op == "train":
        return do_train(train)
    if op == "eval":
        return do_eval(train, req.get("ckpt"), req.get("probe"))
    if op == "checkpoint":
        return do_checkpoint(train, req.get("keep"))
    raise SystemExit(f"unknown op: {op}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("kernel: need train dir")
    train = Path(sys.argv[1]).resolve()
    art = train / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    req_path = art / "request.json"
    out_path = art / "result.json"
    if not req_path.is_file():
        raise SystemExit("missing artifacts/request.json")
    req = json.loads(req_path.read_text(encoding="utf-8"))
    try:
        lines = dispatch(train, req)
        out_path.write_text(json.dumps({"ok": True, "lines": lines}, indent=2) + "\n", encoding="utf-8")
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        msg = str(e.args[0]) if getattr(e, "args", None) else str(e)
        if not msg:
            msg = "kernel failed"
        out_path.write_text(json.dumps({"ok": False, "error": msg}, indent=2) + "\n", encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
