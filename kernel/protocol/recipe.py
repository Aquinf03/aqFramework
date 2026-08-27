"""recipe.yaml is the kernel spec. Nothing else is."""

from __future__ import annotations

from pathlib import Path


def parse_recipe(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    root: dict = {}
    section: str | None = None
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        if stripped[0] not in " \t":
            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                root[key] = {}
                section = key
            elif ":" in stripped:
                k, v = stripped.split(":", 1)
                root[k.strip()] = _scalar(v)
                section = None
            continue
        if section is None:
            continue
        key = stripped.strip()
        if ":" not in key:
            continue
        k, v = key.split(":", 1)
        if not isinstance(root[section], dict):
            root[section] = {}
        root[section][k.strip()] = _scalar(v)
    return root


def _scalar(v: str):
    s = v.strip()
    if s in ("null", "None", "~", ""):
        return None
    if s in ("true", "yes"):
        return True
    if s in ("false", "no"):
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s.strip("'\"" )


def load_recipe(train: Path) -> dict:
    recipe = train / "recipe.yaml"
    if not recipe.is_file():
        raise SystemExit(f"not a train (need recipe.yaml): {train}")
    rec = parse_recipe(recipe)
    if not rec.get("family"):
        raise SystemExit("recipe.yaml must set family")
    method = rec.get("method")
    if not method:
        raise SystemExit("recipe.yaml must set method")
    data = rec.get("data")
    if not isinstance(data, dict) or not data.get("path"):
        raise SystemExit("recipe.yaml must set data.path")
    if method == "lora":
        if not (data.get("text") or data.get("target")):
            raise SystemExit("recipe.yaml must set data.text (or data.target) for lora")
    elif method == "llm":
        obj = str(rec.get("objective") or "next-token").lower().replace("_", "-")
        if obj in (
            "sft",
            "supervised",
            "supervised-finetune",
            "full-ft",
            "full-finetune",
            "full-fine-tune",
            "lora",
        ):
            if not (
                data.get("prompt")
                or data.get("instruction")
                or data.get("completion")
                or data.get("output")
            ):
                raise SystemExit("sft needs data.prompt and data.completion (or instruction/output)")
        elif not (data.get("text") or data.get("target")):
            raise SystemExit("recipe.yaml must set data.text (or data.target) for llm")
    elif method == "transformer":
        arch = str(rec.get("arch") or "decoder").replace("_", "-")
        if arch == "decoder":
            if not (data.get("text") or data.get("target")):
                raise SystemExit("recipe.yaml must set data.text (or data.target) for transformer decoder")
        elif arch == "encoder":
            if not data.get("text") or not data.get("target"):
                raise SystemExit("recipe.yaml must set data.text and data.target for transformer encoder")
        elif arch in ("encoder-decoder", "encoder_decoder"):
            if not data.get("src") or not data.get("tgt"):
                raise SystemExit("recipe.yaml must set data.src and data.tgt for encoder-decoder")
        else:
            raise SystemExit("transformer arch must be encoder, decoder, or encoder-decoder")
    elif not data.get("target"):
        raise SystemExit("recipe.yaml must set data.target")
    ev = rec.get("eval")
    if not isinstance(ev, dict) or not ev.get("metric"):
        raise SystemExit("recipe.yaml must set eval.metric")
    return rec
