"""Post-train deploy: real prune + aq weight quant dump. No fake GPTQ/AWQ/GGUF theater."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backends.device import device_kind
from backends.recipe_opt import opt


def apply_deploy(model, tok, slot: Path, rec: dict, manifest: dict) -> dict:
    """Apply deploy knobs that we actually implement. Fail closed on unsupported ones."""
    _reject_unsupported(rec)

    deploy: dict[str, Any] = {"device": device_kind()}
    train = Path(rec["_train"])

    prune_frac = opt(rec, "prune", None, "deploy", "llm")
    if prune_frac is not None and prune_frac is not False:
        frac = float(prune_frac)
        n_zero = _magnitude_prune(model, frac)
        pruned_dir = slot / "pruned"
        pruned_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(pruned_dir))
        tok.save_pretrained(str(pruned_dir))
        deploy["prune"] = {
            "fraction": frac,
            "zeros": n_zero,
            "path": str(pruned_dir.relative_to(train)),
        }

    quant = opt(rec, "quant", None, "deploy", "quantization", "llm")
    if quant:
        qname = str(quant).lower()
        if qname not in ("int8", "int4", "binary", "q8", "q4", "int1"):
            raise SystemExit(
                f"unsupported quant={quant!r}. Supported: int8, int4, binary "
                "(writes aq quantized.pt dump; not GPTQ/AWQ/GGUF)"
            )
        qdir = slot / "quantized"
        qdir.mkdir(exist_ok=True)
        qmeta = _quantize_state_dict(model, qname, qdir)
        qmeta["path"] = str(qdir.relative_to(train))
        qmeta["note"] = "aq weight dump for inspection/export; serve still loads HF weights"
        deploy["quant"] = qmeta

    if deploy:
        (slot / "deploy.json").write_text(json.dumps(deploy, indent=2) + "\n", encoding="utf-8")
        manifest["deploy"] = deploy
        if "quant" in deploy:
            manifest["quant"] = deploy["quant"]
        if "prune" in deploy:
            manifest["prune"] = deploy["prune"]
    return manifest


def _reject_unsupported(rec: dict) -> None:
    if opt(rec, "formats", False, "deploy", "llm"):
        raise SystemExit(
            "recipe formats: true is not supported yet "
            "(no real GPTQ/AWQ/GGUF/EXL2 exporter). Remove it or implement a converter."
        )
    if opt(rec, "speculative", False, "deploy", "serve", "llm"):
        raise SystemExit(
            "recipe speculative: true is not supported yet "
            "(no draft model). Remove it."
        )
    if opt(rec, "paged_kv", False, "deploy", "serve", "llm"):
        raise SystemExit(
            "recipe paged_kv: true is not supported yet "
            "(HF use_cache is default; not paged attention). Remove it."
        )


def _magnitude_prune(model, fraction: float) -> int:
    import torch

    fraction = max(0.0, min(0.95, float(fraction)))
    tensors = []
    for p in model.parameters():
        if p.ndim >= 2:
            tensors.append(p.data.abs().flatten())
    if not tensors:
        return 0
    allv = torch.cat(tensors)
    k = int(allv.numel() * fraction)
    if k <= 0:
        return 0
    thresh = torch.kthvalue(allv, k).values.item()
    zeros = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim < 2:
                continue
            mask = p.data.abs() >= thresh
            zeros += int((~mask).sum().item())
            p.data.mul_(mask.to(p.data.dtype))
    return zeros


def _quantize_state_dict(model, qname: str, qdir: Path) -> dict:
    import torch

    if qname in ("q8",):
        qname = "int8"
    if qname in ("q4",):
        qname = "int4"
    if qname in ("int1",):
        qname = "binary"

    sd = model.state_dict()
    out = {}
    meta = {"scheme": qname, "tensors": 0, "format": "aq-quantized-pt"}
    for k, v in sd.items():
        if not torch.is_floating_point(v) or v.numel() < 32:
            out[k] = v.cpu()
            continue
        meta["tensors"] += 1
        if qname == "binary":
            out[k] = {
                "q": (v > 0).to(torch.uint8).cpu(),
                "scale": float(v.abs().mean().item()),
                "bits": 1,
            }
        elif qname == "int4":
            flat = v.detach().float().cpu().flatten()
            scale = flat.abs().max().clamp(min=1e-8) / 7.0
            q = (flat / scale).round().clamp(-8, 7).to(torch.int8)
            out[k] = {"q": q.reshape(v.shape), "scale": float(scale), "bits": 4}
        else:
            flat = v.detach().float().cpu()
            scale = flat.abs().max().clamp(min=1e-8) / 127.0
            q = (flat / scale).round().clamp(-127, 127).to(torch.int8)
            out[k] = {"q": q, "scale": float(scale), "bits": 8}
    torch.save(out, qdir / "quantized.pt")
    (qdir / "quant_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
