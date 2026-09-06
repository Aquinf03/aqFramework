"""Shared helpers for aq serve / generate() across methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backends.vision_data import IMG_EXTS


def serve_block(rec: dict) -> dict:
    s = rec.get("serve")
    return s if isinstance(s, dict) else {}


def resolve_image(
    prompt: str | None,
    rec: dict,
    *,
    image: str | None = None,
) -> tuple[Path | None, str]:
    """
    Resolve an image for serve.
    Returns (image_path_or_None, text_prompt).
    Prompt may itself be an image path for vision classify.
    """
    train = Path(rec["_train"]) if rec.get("_train") else Path.cwd()
    block = serve_block(rec)
    cand = image or block.get("image") or block.get("img") or block.get("path")
    text = prompt if prompt is not None else block.get("prompt")
    text = "" if text is None else str(text)

    def _file(p: str | Path) -> Path | None:
        path = Path(str(p))
        if path.is_file():
            return path.resolve()
        alt = train / path
        if alt.is_file():
            return alt.resolve()
        return None

    if cand:
        got = _file(cand)
        if not got:
            raise SystemExit(f"serve image not found: {cand}")
        return got, text

    # prompt is an image path (vision classify / clip image-only)
    if text:
        p = Path(text.strip().strip("\"'"))
        if p.suffix.lower() in IMG_EXTS:
            got = _file(p)
            if got:
                return got, ""
    return None, text


def parse_feature_row(prompt: str, n_features: int | None = None) -> list[float]:
    """Parse '[1,2,3]' / '1,2,3' / JSON object values into one feature row."""
    s = prompt.strip()
    if not s:
        raise SystemExit(
            "tabular serve needs a feature row as the prompt "
            '(e.g. aq serve "[1.0, 2.0, 3.0]" or recipe serve.features)'
        )
    try:
        blob = json.loads(s)
    except json.JSONDecodeError:
        blob = None
    if isinstance(blob, list):
        row = [float(x) for x in blob]
    elif isinstance(blob, dict):
        row = [float(v) for v in blob.values()]
    else:
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        try:
            row = [float(p) for p in parts]
        except ValueError as e:
            raise SystemExit(
                f"could not parse feature row from {prompt!r}. "
                'Use JSON like "[1,2,3]" or comma-separated numbers.'
            ) from e
    if n_features is not None and len(row) != n_features:
        raise SystemExit(f"expected {n_features} features, got {len(row)}")
    return row


def result_text(**fields: Any) -> dict:
    """Normalize serve.json payload: always has text + completion aliases."""
    out = dict(fields)
    if "text" not in out and "completion" in out:
        out["text"] = out["completion"]
    if "completion" not in out and "text" in out:
        out["completion"] = out["text"]
    return out
