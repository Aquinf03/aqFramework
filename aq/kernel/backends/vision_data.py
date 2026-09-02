"""Load image classification datasets for vision CNN trains."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from backends.recipe_opt import opt


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ImageNet defaults — override in recipe if your data is different
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _require_pil():
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit(
            "Pillow is required for vision trains.\n"
            "Fix: pip install pillow   in the kernel venv."
        ) from e
    return Image


def resolve_samples(src: Path, rec: dict) -> tuple[list[tuple[Path, int]], list[str]]:
    """Return (path, class_idx) pairs and class name list."""
    data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
    if src.is_dir():
        return _from_imagefolder(src)
    if src.is_file():
        return _from_table(src, rec, data)
    raise SystemExit(f"data.path not found: {src}")


def _from_imagefolder(root: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    classes = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if len(classes) < 2:
        raise SystemExit(
            f"ImageFolder needs ≥2 class subdirs under {root} "
            "(e.g. data/images/cat/, data/images/dog/)."
        )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples: list[tuple[Path, int]] = []
    for c in classes:
        for p in sorted((root / c).rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append((p, class_to_idx[c]))
    if not samples:
        raise SystemExit(f"no images under {root} (supported: {sorted(IMG_EXTS)})")
    return samples, classes


def _from_table(
    path: Path, rec: dict, data: dict
) -> tuple[list[tuple[Path, int]], list[str]]:
    image_key = str(
        data.get("image") or data.get("path_col") or data.get("file") or "path"
    )
    target = str(data.get("target") or data.get("label") or "label")
    rows = _rows(path)
    if not rows:
        raise SystemExit(f"empty data: {path}")
    if image_key not in rows[0]:
        raise SystemExit(f"data needs column {image_key!r} (image path)")
    if target not in rows[0]:
        raise SystemExit(f"data needs column {target!r} (class label)")
    labels = sorted({str(r[target]) for r in rows})
    class_to_idx = {c: i for i, c in enumerate(labels)}
    train = Path(rec["_train"]) if rec.get("_train") else path.parent
    samples: list[tuple[Path, int]] = []
    for r in rows:
        rel = Path(str(r[image_key]))
        img = rel if rel.is_file() else (train / rel)
        if not img.is_file():
            img = path.parent / rel
        if not img.is_file():
            raise SystemExit(f"image not found: {r[image_key]}")
        samples.append((img, class_to_idx[str(r[target])]))
    return samples, labels


def _rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_loader(
    samples: list[tuple[Path, int]],
    *,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    train: bool,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    num_workers: int = 0,
):
    Image = _require_pil()
    import torch
    from torch.utils.data import DataLoader, Dataset

    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    class ImgDS(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, i: int):
            path, y = samples[i]
            img = Image.open(path).convert("RGB")
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            if train:
                import random

                if random.random() < 0.5:
                    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            w, h = img.size
            buf = torch.tensor(bytearray(img.tobytes()), dtype=torch.uint8)
            t = buf.view(h, w, 3).permute(2, 0, 1).float() / 255.0
            for c in range(3):
                t[c] = (t[c] - mean[c]) / std[c]
            return t, int(y)

    return DataLoader(
        ImgDS(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def split_train_val(
    samples: list[tuple[Path, int]], val_frac: float, seed: int
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    import random

    if val_frac <= 0:
        return samples, []
    rng = random.Random(seed)
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    train_s: list = []
    val_s: list = []
    for items in by_class.values():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac)) if len(items) > 1 else 0
        val_s.extend(items[:n_val])
        train_s.extend(items[n_val:] or items)
    rng.shuffle(train_s)
    rng.shuffle(val_s)
    return train_s, val_s
