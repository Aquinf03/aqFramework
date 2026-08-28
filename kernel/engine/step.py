#!/usr/bin/env python3
"""Kernel steps: train, eval, serve, checkpoint. Compose them yourself. aq eval scores and pass/fails."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from methods.linear import load_xy
from protocol.method import call_fit, load_method
from protocol.recipe import load_recipe
from protocol.record import update_last_run, write_run
from protocol.tokenizer import check as check_tokenizer
from protocol.tokenizer import pin as pin_tokenizer

LOWER = {"mse", "rmse", "mae", "loss"}


def ckpt_dir(train: Path) -> Path:
    d = train / "artifacts" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def last_ckpt(train: Path) -> Path:
    p = ckpt_dir(train) / "last.json"
    if not p.is_file():
        raise SystemExit("no checkpoint (run aq train)")
    return p


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def probe_files(train: Path, name: str | None = None) -> list[Path]:
    d = train / "evals"
    if not d.is_dir():
        return []
    if name:
        for ext in (".csv", ".jsonl"):
            p = d / (name if name.endswith(ext) else name + ext)
            if p.is_file():
                return [p]
        raise SystemExit(f"no eval {name} in {d}")
    out = []
    for p in sorted(d.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_file() and p.suffix in {".csv", ".jsonl"}:
            out.append(p)
    return out


def data_file(train: Path, rec: dict) -> Path:
    rel = (rec.get("data") or {}).get("path")
    if not rel:
        raise SystemExit("recipe.yaml has no data.path")
    src = (train / str(rel)).resolve()
    if not src.is_file():
        raise SystemExit(f"data path not found: {rel}")
    return src


def do_train(train: Path) -> list[str]:
    rec = load_recipe(train)
    method = rec["method"]
    src = data_file(train, rec)
    mod = load_method(train, str(method))
    model = call_fit(mod, src, rec)
    if not isinstance(model, dict):
        raise SystemExit("fit() must return a dict")
    model.setdefault("kind", str(method))
    tok_hash = pin_tokenizer(train, model)
    dest = ckpt_dir(train)
    n = 1 + sum(1 for p in dest.glob("*.json") if p.name != "last.json")
    named = dest / f"{n}.json"
    write_json(named, model)
    shutil.copy2(named, dest / "last.json")
    arts = {
        "checkpoint": "artifacts/checkpoints/" + named.name,
        "checkpoint_last": "artifacts/checkpoints/last.json",
    }
    if tok_hash:
        arts["tokenizer"] = "artifacts/tokenizer.json"
        arts["tokenizer_sha256"] = tok_hash
    if hasattr(mod, "write_inspect"):
        arts["inspect"] = mod.write_inspect(train, model)
    rid = write_run(train, {"artifacts": arts})
    lines = [
        "train",
        "  artifacts/checkpoints/" + named.name,
        "  artifacts/checkpoints/last.json",
        "  artifacts/runs/" + rid + ".json",
    ]
    if arts.get("inspect"):
        lines.append("  " + arts["inspect"])
    if arts.get("tokenizer"):
        lines.append("  " + arts["tokenizer"])
        lines.append("  tokenizer sha256:" + str(arts.get("tokenizer_sha256")))
    return lines


def score(metric: str, y_true: list, y_hat: list) -> float:
    n = len(y_true)
    if n == 0:
        raise SystemExit("empty eval")
    if metric == "accuracy":
        hit = sum(1 for a, b in zip(y_true, y_hat) if str(a) == str(b))
        return hit / n
    yt = [float(v) for v in y_true]
    yh = [float(v) for v in y_hat]
    err = [a - b for a, b in zip(yt, yh)]
    if metric == "mae":
        return sum(abs(e) for e in err) / n
    mse = sum(e * e for e in err) / n
    if metric == "mse":
        return mse
    if metric == "rmse":
        return mse ** 0.5
    if metric == "r2":
        mean = sum(yt) / n
        tot = sum((a - mean) ** 2 for a in yt)
        if tot == 0:
            return 1.0
        return 1.0 - mse * n / tot
    raise SystemExit(f"unknown metric: {metric}")


def _score_file(train: Path, rec: dict, model: dict, src: Path) -> tuple[str, float, int]:
    kind = str(model.get("kind") or rec.get("method") or "linear")
    mod = load_method(train, kind)
    if hasattr(mod, "evaluate") and not hasattr(mod, "predict"):
        metric = str((rec.get("eval") or {}).get("metric") or "loss")
        sc, n = mod.evaluate(model, src, rec)
        return str(metric), sc, n
    if not hasattr(mod, "predict"):
        raise SystemExit(f"method {kind} needs predict() or evaluate()")
    target = (rec.get("data") or {}).get("target")
    if not target:
        raise SystemExit("recipe.yaml has no data.target")
    metric = str((rec.get("eval") or {}).get("metric") or "mse")
    feats, X, y = load_xy(src, str(target))
    if feats != model.get("features"):
        raise SystemExit(f"features do not match data: {src}")
    y_hat = mod.predict(model, X)
    return metric, score(metric, y, y_hat), len(y)


def _file_pass(metric: str, sc: float, min_score) -> bool | None:
    if min_score is None:
        return None
    lo = str(metric) in LOWER
    return sc <= float(min_score) if lo else sc >= float(min_score)


def do_eval(train: Path, ckpt_name: str | None, probe: str | None = None) -> list[str]:
    rec = load_recipe(train)
    ckpt = ckpt_dir(train) / ckpt_name if ckpt_name else last_ckpt(train)
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint: {ckpt.name}")
    model = json.loads(ckpt.read_text(encoding="utf-8"))
    pinned = check_tokenizer(train, model)
    if pinned is not None:
        model["tokenizer"] = pinned
    files = probe_files(train, probe)
    if not files:
        files = [data_file(train, rec)]
    min_score = (rec.get("eval") or {}).get("min_score")
    probes = []
    wsum = 0.0
    ntot = 0
    metric = "mse"
    all_pass: bool | None = True if min_score is not None else None
    for src in files:
        metric, sc, n = _score_file(train, rec, model, src)
        try:
            rel = str(src.relative_to(train))
        except ValueError:
            rel = str(src)
        fp = _file_pass(metric, sc, min_score)
        probes.append({"path": rel, "score": sc, "n": n, "pass": fp})
        wsum += sc * n
        ntot += n
        if all_pass is True and fp is False:
            all_pass = False
    if ntot == 0:
        raise SystemExit("empty eval")
    sc = wsum / ntot
    verdict = "skip" if min_score is None else ("pass" if all_pass else "fail")
    out = {
        "metric": metric,
        "score": sc,
        "n": ntot,
        "pass": all_pass,
        "min_score": min_score,
        "checkpoint": str(ckpt.relative_to(train)),
        "probes": probes,
    }
    write_json(train / "artifacts" / "eval.json", out)
    rid = update_last_run(
        train,
        {
            "metrics": {"metric": metric, "score": sc, "n": ntot, "probes": probes},
            "pass": all_pass,
            "artifacts": {"eval": "artifacts/eval.json", "checkpoint": out["checkpoint"]},
        },
    )
    lines = [
        "eval",
        "  " + str(out["metric"]),
        "  " + str(sc),
        "  " + verdict,
        "  " + out["checkpoint"],
        "  artifacts/runs/" + rid + ".json",
    ]
    if len(probes) > 1 or (probes and probes[0]["path"].startswith("evals/")):
        for p in probes:
            lines.append("  " + p["path"] + "  " + str(p["score"]))
    return lines


def do_serve(
    train: Path,
    ckpt_name: str | None,
    prompt: str | None,
    max_tokens: int | None,
    temperature: float | None,
) -> list[str]:
    rec = load_recipe(train)
    ckpt = ckpt_dir(train) / ckpt_name if ckpt_name else last_ckpt(train)
    if not ckpt.is_file():
        raise SystemExit(f"no checkpoint: {ckpt.name}")
    model = json.loads(ckpt.read_text(encoding="utf-8"))
    pinned = check_tokenizer(train, model)
    if pinned is not None:
        model["tokenizer"] = pinned
    if not prompt:
        serve = rec.get("serve") if isinstance(rec.get("serve"), dict) else {}
        prompt = serve.get("prompt")
    if not prompt:
        raise SystemExit("serve needs a prompt (argv or recipe serve.prompt)")
    kind = str(model.get("kind") or rec.get("method") or "linear")
    mod = load_method(train, kind)
    if not hasattr(mod, "generate"):
        raise SystemExit(f"method {kind} has no generate()")
    out = mod.generate(model, str(prompt), rec, max_tokens=max_tokens, temperature=temperature)
    out["checkpoint"] = str(ckpt.relative_to(train))
    write_json(train / "artifacts" / "serve.json", out)
    rid = update_last_run(
        train,
        {
            "artifacts": {
                "serve": "artifacts/serve.json",
                "checkpoint": out["checkpoint"],
            },
        },
    )
    lines = [
        "serve",
        "  " + str(out.get("text") or ""),
        "  " + out["checkpoint"],
        "  artifacts/serve.json",
        "  artifacts/runs/" + rid + ".json",
    ]
    if out.get("completion") not in (None, ""):
        lines.append("  completion: " + str(out.get("completion")))
    lines.append("  tokens: " + str(out.get("tokens")))
    return lines


def do_checkpoint(train: Path, keep: str | None) -> list[str]:
    dest = ckpt_dir(train)
    if keep:
        src = last_ckpt(train)
        name = keep if keep.endswith(".json") else keep + ".json"
        shutil.copy2(src, dest / name)
        return ["checkpoint", "  artifacts/checkpoints/" + name]
    names = sorted(p.name for p in dest.glob("*.json"))
    lines = ["checkpoint"]
    if not names:
        lines.append("  (none)")
        return lines
    for n in names:
        lines.append("  " + n)
    return lines
