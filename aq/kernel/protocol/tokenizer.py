"""Tokenizer file in the train. Hashed like weights. Eval loads this file, not a new fit."""

from __future__ import annotations

import json
from pathlib import Path

from protocol.revision import hash_file

REL = "artifacts/tokenizer.json"


def dump(train: Path, spec: dict) -> str:
    blob = json.dumps(spec, sort_keys=True, indent=2) + "\n"
    dest = train / REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(blob, encoding="utf-8")
    digest, _ = hash_file(dest)
    return digest


def load(train: Path) -> dict:
    path = train / REL
    if not path.is_file():
        raise SystemExit("no artifacts/tokenizer.json (tokenizer lives in the train, like weights)")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise SystemExit("artifacts/tokenizer.json is not valid json")
    if not isinstance(spec, dict):
        raise SystemExit("artifacts/tokenizer.json must be an object")
    return spec


def pin(train: Path, model: dict) -> str | None:
    spec = model.get("tokenizer")
    if not isinstance(spec, dict):
        return None
    digest = dump(train, spec)
    model["tokenizer_sha256"] = digest
    return digest


def check(train: Path, model: dict) -> dict | None:
    if not (model.get("tokenizer") or model.get("tokenizer_sha256")):
        return None
    spec = load(train)
    digest, _ = hash_file(train / REL)
    want = model.get("tokenizer_sha256")
    if want and digest != want:
        raise SystemExit(
            "tokenizer.json hash does not match checkpoint (tokenizer is hashed like weights)"
        )
    return spec
