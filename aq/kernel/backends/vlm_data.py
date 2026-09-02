"""Load image–text pairs and multimodal chat rows for VLM trains."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from backends.recipe_opt import opt
from backends.vision_data import IMAGENET_MEAN, IMAGENET_STD, IMG_EXTS, _require_pil


def _rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    if path.suffix.lower() in (".json",):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob, list):
            return blob
        raise SystemExit("json data must be a list of objects")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_pairs(src: Path, rec: dict) -> list[tuple[Path, str]]:
    """Return (image_path, caption) pairs."""
    data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
    image_key = str(data.get("image") or data.get("path") or data.get("file") or "image")
    text_key = str(data.get("text") or data.get("caption") or data.get("alt") or "text")
    if src.is_dir():
        raise SystemExit(
            "vlm clip/siglip needs a pairs table (jsonl/csv), not ImageFolder. "
            "Each row: image path + caption text."
        )
    rows = _rows(src)
    if not rows:
        raise SystemExit(f"empty data: {src}")
    if image_key not in rows[0]:
        raise SystemExit(f"data needs column {image_key!r}")
    if text_key not in rows[0]:
        raise SystemExit(f"data needs column {text_key!r}")
    train = Path(rec["_train"]) if rec.get("_train") else src.parent
    pairs: list[tuple[Path, str]] = []
    for r in rows:
        rel = Path(str(r[image_key]))
        img = rel if rel.is_file() else (train / rel)
        if not img.is_file():
            img = src.parent / rel
        if not img.is_file():
            raise SystemExit(f"image not found: {r[image_key]}")
        if img.suffix.lower() not in IMG_EXTS:
            raise SystemExit(f"unsupported image type: {img}")
        pairs.append((img, str(r[text_key])))
    return pairs


def resolve_chat(src: Path, rec: dict) -> list[dict[str, Any]]:
    """
    Multimodal chat rows for LLaVA / Flamingo / GPT-4V-style.

    Accepted shapes per row:
      {image, conversations:[{from|role, value|content}, ...]}
      {image, prompt|question, completion|answer|response}
      {image, messages:[{role, content}, ...]}
    """
    data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
    image_key = str(data.get("image") or "image")
    if src.is_dir():
        raise SystemExit("vlm generative needs a jsonl/csv of chat rows, not a directory")
    rows = _rows(src)
    train = Path(rec["_train"]) if rec.get("_train") else src.parent
    out: list[dict[str, Any]] = []
    for r in rows:
        rel = Path(str(r.get(image_key) or r.get("image") or ""))
        if not str(rel):
            raise SystemExit("each chat row needs an image path")
        img = rel if rel.is_file() else (train / rel)
        if not img.is_file():
            img = src.parent / rel
        if not img.is_file():
            raise SystemExit(f"image not found: {rel}")
        conv = r.get("conversations") or r.get("messages")
        if conv is None:
            prompt = r.get("prompt") or r.get("question") or r.get("instruction")
            answer = r.get("completion") or r.get("answer") or r.get("response") or r.get("output")
            if prompt is None or answer is None:
                raise SystemExit(
                    "chat row needs conversations/messages or prompt+completion fields"
                )
            conv = [
                {"from": "human", "value": str(prompt)},
                {"from": "gpt", "value": str(answer)},
            ]
        turns = []
        for t in conv:
            role = str(t.get("from") or t.get("role") or "").lower()
            val = str(t.get("value") or t.get("content") or "")
            if role in ("human", "user"):
                turns.append({"role": "user", "content": val})
            elif role in ("gpt", "assistant"):
                turns.append({"role": "assistant", "content": val})
            else:
                turns.append({"role": role or "user", "content": val})
        out.append({"image": img, "turns": turns})
    if not out:
        raise SystemExit(f"empty chat data: {src}")
    return out


def load_image_tensor(path: Path, image_size: int, *, train: bool = False):
    Image = _require_pil()
    import random
    import torch

    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    if train and random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    buf = torch.tensor(bytearray(img.tobytes()), dtype=torch.uint8)
    t = buf.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    for c in range(3):
        t[c] = (t[c] - mean[c]) / std[c]
    return t


def make_pair_loader(pairs: list[tuple[Path, str]], tokenizer, *, image_size: int, batch_size: int, shuffle: bool, train: bool, num_workers: int = 0):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DS(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            path, text = pairs[i]
            img = load_image_tensor(path, image_size, train=train)
            ids = tokenizer.encode(text)
            return img, torch.tensor(ids, dtype=torch.long)

    return DataLoader(DS(), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def split_pairs(pairs: list, val_frac: float, seed: int) -> tuple[list, list]:
    import random

    if val_frac <= 0:
        return pairs, []
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_frac)) if len(items) > 1 else 0
    return items[n_val:], items[:n_val]
