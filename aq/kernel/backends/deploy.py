"""Post-train deploy: real prune + aq weight quant dump. No fake GPTQ/AWQ/GGUF theater."""

from __future__ import annotations

import json
import sys
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

    from backends.formats_export import export_formats, formats_requested

    if formats_requested(rec):
        deploy["formats"] = export_formats(model, tok, slot, rec, manifest)

    if deploy:
        (slot / "deploy.json").write_text(json.dumps(deploy, indent=2) + "\n", encoding="utf-8")
        manifest["deploy"] = deploy
        if "quant" in deploy:
            manifest["quant"] = deploy["quant"]
        if "prune" in deploy:
            manifest["prune"] = deploy["prune"]
        if "formats" in deploy:
            manifest["formats"] = deploy["formats"]
    return manifest


def _reject_unsupported(rec: dict) -> None:
    """Hard failures for knobs that would fake export/deploy work."""
    # formats: true is handled by formats_export (real converters / clear skips)
    return


def speculative_requested(rec: dict) -> bool:
    """True when recipe asks for speculative / assisted decode."""
    v = opt(rec, "speculative", False, "deploy", "serve", "llm")
    if isinstance(v, dict):
        return v.get("enabled", True) is not False
    return bool(v)


def draft_model_id(rec: dict, model: dict | None = None) -> str | None:
    """Hub id (or local path) for the draft / assistant model."""
    v = opt(rec, "speculative", None, "deploy", "serve", "llm")
    if isinstance(v, dict):
        for key in ("draft", "draft_model", "model", "assistant"):
            if v.get(key):
                return str(v[key])
    draft = opt(rec, "draft_model", None, "deploy", "serve", "llm")
    if draft:
        return str(draft)
    if model:
        intent = (model.get("serve_intent") or {}).get("speculative") or {}
        if intent.get("draft_model"):
            return str(intent["draft_model"])
        if model.get("draft_model"):
            return str(model["draft_model"])
    return None


def note_serve_intent(rec: dict, manifest: dict) -> None:
    """Serve-time recipe flags — recorded at train; speculative needs a draft_model."""
    intent: dict[str, Any] = {}
    if opt(rec, "paged_kv", False, "deploy", "serve", "llm"):
        intent["paged_kv"] = {
            "requested": True,
            "active": False,
            "note": "aq serve uses Hugging Face use_cache; not vLLM-style paged attention yet",
        }
        print(
            "  serve  paged_kv: true — train ignores; serve uses standard HF KV cache",
            file=sys.stderr,
        )
    if speculative_requested(rec):
        draft = draft_model_id(rec)
        if not draft:
            raise SystemExit(
                "recipe speculative: true needs a draft model.\n"
                "Reason: assisted decode loads a smaller assistant beside the target.\n"
                "Fix: set draft_model: <hub-id> (or speculative: { draft: <hub-id> })."
            )
        intent["speculative"] = {
            "requested": True,
            "active": True,
            "draft_model": draft,
            "note": "aq serve uses Hugging Face assisted generation (assistant_model)",
        }
        manifest["draft_model"] = draft
        print(
            f"  serve  speculative: true — draft={draft}",
            file=sys.stderr,
        )
    if intent:
        manifest["serve_intent"] = intent


def warn_serve_intent(rec: dict, model: dict | None = None) -> None:
    """Remind at serve time when recipe or checkpoint asked for unimplemented serve opts."""
    if opt(rec, "paged_kv", False, "deploy", "serve", "llm") or (
        model and (model.get("serve_intent") or {}).get("paged_kv")
    ):
        print(
            "  serve  paged_kv was requested; using standard HF generation (not paged attention)",
            file=sys.stderr,
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
