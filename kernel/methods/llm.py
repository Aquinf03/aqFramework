"""LLM size class and deploy knobs. Decoder + BPE. Stdlib. Not a hub download."""

from __future__ import annotations

import json
import math
from pathlib import Path

from methods import tok as tokenize
from protocol.tokenizer import dump as dump_tokenizer
from methods.linear import _rows
from methods.transformer import (
    Var,
    add_const,
    backward,
    block,
    embed,
    matmul,
    nll,
    sgd,
    _causal_mask,
    _init_block,
    _pack,
    _pos,
    _rand,
    _texts_field,
)

PRESET = {
    "llm": {"d_model": 16, "d_ff": 32, "layers": 2, "context": 24},
    "slm": {"d_model": 8, "layers": 1, "d_ff": 16, "context": 16},
    "edge": {"d_model": 8, "layers": 1, "d_ff": 16, "context": 8},
}


def _opt(rec: dict, key: str, default):
    v = rec.get(key)
    if v is None:
        nested = rec.get("llm")
        if isinstance(nested, dict):
            v = nested.get(key)
    return default if v is None else v


def _docs(src: Path, rec: dict) -> tuple[list[str], list[str], dict]:
    data = rec.get("data") or {}
    field = str(data.get("text") or data.get("target") or "text")
    src_key = str(data.get("source") or "source")
    mix = rec.get("mixture") if isinstance(rec.get("mixture"), dict) else {}
    weights = {str(k): float(v) for k, v in mix.items() if v is not None}
    rows = _rows(src)
    texts: list[str] = []
    sources: list[str] = []
    for r in rows:
        t = str(r.get(field) or "").strip()
        if not t:
            continue
        texts.append(t)
        sources.append(str(r.get(src_key) or "default"))
    if not texts:
        raise SystemExit("llm: no texts")
    return texts, sources, weights


def _size(rec: dict) -> str:
    s = str(_opt(rec, "size", "llm")).lower()
    if s in ("tiny", "tinyml", "edge"):
        return "edge"
    if s not in PRESET:
        raise SystemExit("llm size must be llm, slm, or edge")
    return s


def _decode(ids: list[int], tok: Var, blks: list[dict], wout: Var) -> Var:
    x = add_const(embed(ids, tok), _pos(len(ids), len(tok.data[0])))
    mask = _causal_mask(len(ids))
    for blk in blks:
        x = block(x, blk, mask)
    return matmul(x, wout)


def _mm(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k = len(a), len(a[0])
    n = len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(n)] for i in range(m)]


def _add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _relu(a: list[list[float]]) -> list[list[float]]:
    return [[x if x > 0 else 0.0 for x in row] for row in a]


def _softmax_row(row: list[float]) -> list[float]:
    mx = max(row)
    ex = [math.exp(x - mx) for x in row]
    z = sum(ex) or 1.0
    return [e / z for e in ex]


def _blk_float(x: list[list[float]], p: dict, k_cache=None, v_cache=None) -> tuple:
    q = _mm(x, p["wq"])
    k = _mm(x, p["wk"])
    v = _mm(x, p["wv"])
    if k_cache:
        k = k_cache + k
        v = v_cache + v
    d = len(q[0])
    scale = 1.0 / math.sqrt(d)
    tq, tk = len(q), len(k)
    scores = [[sum(q[i][u] * k[j][u] for u in range(d)) * scale for j in range(tk)] for i in range(tq)]
    off = tk - tq
    for i in range(tq):
        for j in range(tk):
            if j > i + off:
                scores[i][j] = -1e9
    a = [_softmax_row(row) for row in scores]
    ctx = [[sum(a[i][j] * v[j][c] for j in range(tk)) for c in range(d)] for i in range(tq)]
    h = _add(x, _mm(ctx, p["wo"]))
    h = _add(h, _mm(_relu(_mm(h, p["w1"])), p["w2"]))
    return h, k, v


def _mats(model: dict) -> tuple:
    tok = model["tok"]
    wout = model["wout"]
    names = ["wq", "wk", "wv", "wo", "w1", "w2"]
    blks = [{k: pack[k] for k in names} for pack in model["blocks"]]
    return tok, blks, wout


def _decode_float(ids: list[int], tok, blks, wout) -> list[list[float]]:
    d = len(tok[0])
    x = [tok[i][:] for i in ids]
    pos = _pos(len(ids), d)
    x = _add(x, pos)
    for blk in blks:
        x, _, _ = _blk_float(x, blk)
    return _mm(x, wout)


def _nll_ids(ids: list[int], tok, blks, wout) -> tuple[float, int]:
    logits = _decode_float(ids[:-1], tok, blks, wout)
    total = 0.0
    n = 0
    for i, gold in enumerate(ids[1:]):
        p = _softmax_row(logits[i])
        total += -math.log(max(p[gold], 1e-12))
        n += 1
    return total, n


def _q_int(mat: list[list[float]], bits: int) -> dict:
    bits = int(bits)
    if bits == 1:
        q = [[1 if x >= 0 else -1 for x in row] for row in mat]
        return {"bits": 1, "q": q, "scale": 1.0}
    lo, hi = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    mx = max((abs(x) for row in mat for x in row), default=1.0) or 1.0
    scale = mx / max(hi, 1)
    q = [[int(max(lo, min(hi, round(x / scale)))) for x in row] for row in mat]
    return {"bits": bits, "q": q, "scale": scale}


def _deq(blob: dict) -> list[list[float]]:
    s = float(blob["scale"])
    return [[v * s for v in row] for row in blob["q"]]


def _quant_model(model: dict, bits: int) -> dict:
    out = dict(model)
    out["quant"] = f"int{bits}" if bits > 1 else "binary"
    out["tok_q"] = _q_int(model["tok"], bits)
    out["wout_q"] = _q_int(model["wout"], bits)
    out["blocks_q"] = [{k: _q_int(pack[k], bits) for k in pack} for pack in model["blocks"]]
    return out


def _load_float(model: dict) -> tuple:
    if model.get("tok_q"):
        tok = _deq(model["tok_q"])
        wout = _deq(model["wout_q"])
        blks = [{k: _deq(pack[k]) for k in pack} for pack in model["blocks_q"]]
        return tok, blks, wout
    return _mats(model)


def _prune(mat: list[list[float]], frac: float) -> tuple[list[list[float]], float]:
    vals = sorted(abs(x) for row in mat for x in row)
    if not vals:
        return mat, 0.0
    cut = vals[min(len(vals) - 1, int(len(vals) * frac))]
    out = [[0.0 if abs(x) < cut else x for x in row] for row in mat]
    z = sum(1 for row in out for x in row if x == 0.0)
    n = len(out) * len(out[0])
    return out, z / n if n else 0.0


def _prune_model(model: dict, frac: float) -> dict:
    spars = []
    tok, s = _prune(model["tok"], frac)
    spars.append(s)
    wout, s = _prune(model["wout"], frac)
    spars.append(s)
    blocks = []
    for pack in model["blocks"]:
        npk = {}
        for k, mat in pack.items():
            npk[k], s = _prune(mat, frac)
            spars.append(s)
        blocks.append(npk)
    out = dict(model)
    out["tok"] = tok
    out["wout"] = wout
    out["blocks"] = blocks
    out["prune"] = frac
    out["sparsity"] = sum(spars) / len(spars)
    return out


def _write_formats(train: Path, model: dict) -> list[str]:
    art = train / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    q8 = _quant_model(model, 8)
    q4 = _quant_model(model, 4)
    names = []
    gptq = {"kind": "gptq-lite", "bits": 4, "group": "tensor", "wout_q": q4["wout_q"]}
    awq = {"kind": "awq-lite", "bits": 4, "wout_q": q4["wout_q"]}
    exl2 = {"kind": "exl2-lite", "bits": 4, "wout_q": q4["wout_q"]}
    for name, obj in (
        ("weights.q8.json", {"kind": "q8", "tok_q": q8["tok_q"], "wout_q": q8["wout_q"]}),
        ("weights.gptq.json", gptq),
        ("weights.awq.json", awq),
        ("weights.exl2.json", exl2),
    ):
        (art / name).write_text(json.dumps(obj) + "\n", encoding="utf-8")
        names.append("artifacts/" + name)
    payload = json.dumps({"kind": "gguf-lite", "vocab": model.get("vocab"), "layers": model.get("layers")}).encode()
    (art / "model.gguf").write_bytes(b"GGUF" + len(payload).to_bytes(8, "little") + payload)
    names.append("artifacts/model.gguf")
    return names


def _kl(p: list[float], q: list[float]) -> float:
    return sum(pi * (math.log(max(pi, 1e-12)) - math.log(max(qi, 1e-12))) for pi, qi in zip(p, q))


def _train_core(
    texts: list[str],
    rec: dict,
    size: str,
    layers=None,
    d=None,
    dff=None,
    ctx=None,
    sources: list[str] | None = None,
    weights: dict | None = None,
) -> dict:
    preset = PRESET[size]
    d = int(d if d is not None else _opt(rec, "d_model", preset["d_model"]))
    dff = int(dff if dff is not None else _opt(rec, "d_ff", preset["d_ff"]))
    layers = int(layers if layers is not None else _opt(rec, "layers", preset["layers"]))
    ctx = int(ctx if ctx is not None else _opt(rec, "context", preset["context"]))
    steps = int(_opt(rec, "steps", 40))
    lr = float(_opt(rec, "lr", 0.05))
    merges = int(_opt(rec, "merges", 24))
    seed = [int(_opt(rec, "seed", 1))]
    tok_spec = tokenize.train(texts, rec)
    seed_i = int(_opt(rec, "seed", 1))
    windows, pack_stats = tokenize.pack_ex(
        texts, tok_spec, ctx, sources, weights or None, seed_i
    )
    vsz = tokenize.vocab_size(tok_spec)
    tok = Var(_rand(vsz, d, 0.2, seed))
    blks = [_init_block(d, dff, seed) for _ in range(layers)]
    wout = Var(_rand(d, vsz, 0.2, seed))
    params = [tok, wout]
    for blk in blks:
        params.extend(blk.values())
    last = 0.0
    for _ in range(steps):
        total = 0.0
        for ids in windows:
            logits = _decode(ids[:-1], tok, blks, wout)
            loss = nll(logits, ids[1:])
            backward(loss)
            sgd(params, lr)
            total += loss.data[0][0]
        last = total / len(windows)
    n_pred = sum(max(len(w) - 1, 0) for w in windows)
    return {
        "kind": "llm",
        "class": size,
        "arch": "decoder",
        "objective": "next-token",
        "causal": True,
        "d_model": d,
        "d_ff": dff,
        "layers": layers,
        "heads": 1,
        "context": ctx,
        "merges": merges,
        "train_loss": last,
        "vocab": vsz,
        "tokenizer": tok_spec,
        "tokenizer_sha256": tokenize.sha256(tok_spec),
        "tok": tok.data,
        "wout": wout.data,
        "blocks": [_pack(b) for b in blks],
        "n_windows": len(windows),
        "n_tokens": pack_stats["tokens"],
        "pack": pack_stats.get("pack"),
        "mixture": pack_stats.get("mixture"),
        "packed_tokens": pack_stats.get("packed_tokens"),
        "pack_docs": pack_stats.get("docs"),
        "pack_seed": pack_stats.get("seed"),
        "n_pred": n_pred,
        "windows": windows,
    }


def _distill(texts: list[str], rec: dict, sources=None, weights=None) -> dict:
    teacher = _train_core(texts, rec, "llm", sources=sources, weights=weights)
    student = _train_core(texts, rec, "slm", sources=sources, weights=weights)
    tok_spec = teacher["tokenizer"]
    ctx = int(student["context"])
    seed_i = int(_opt(rec, "seed", 1))
    windows, _ = tokenize.pack_ex(texts, tok_spec, ctx, sources, weights, seed_i)
    t_tok, t_blks, t_wout = _load_float(teacher)
    d, dff, layers = student["d_model"], student["d_ff"], student["layers"]
    seed = [int(_opt(rec, "seed", 1)) + 7]
    vsz = teacher["vocab"]
    tok = Var(_rand(vsz, d, 0.2, seed))
    blks = [_init_block(d, dff, seed) for _ in range(layers)]
    wout = Var(_rand(d, vsz, 0.2, seed))
    params = [tok, wout]
    for blk in blks:
        params.extend(blk.values())
    steps = int(_opt(rec, "distill_steps", _opt(rec, "steps", 40)))
    lr = float(_opt(rec, "lr", 0.05))
    last = 0.0
    for _ in range(steps):
        total = 0.0
        for ids in windows:
            t_log = _decode_float(ids[:-1], t_tok, t_blks, t_wout)
            logits = _decode(ids[:-1], tok, blks, wout)
            ce = nll(logits, ids[1:])
            kl = 0.0
            for i in range(len(ids) - 1):
                p = _softmax_row(t_log[i])
                q = _softmax_row(logits.data[i])
                kl += _kl(p, q)
            kl /= max(len(ids) - 1, 1)
            # CE backward only; KL is a report. Distill signal: extra CE on teacher argmax
            hard = [max(range(len(t_log[i])), key=lambda j: t_log[i][j]) for i in range(len(ids) - 1)]
            extra = nll(logits, hard)
            backward(ce)
            sgd(params, lr)
            backward(extra)
            sgd(params, lr * 0.5)
            total += ce.data[0][0] + 0.5 * kl
        last = total / len(windows)
    out = {
        "kind": "llm",
        "class": "distilled",
        "arch": "decoder",
        "d_model": d,
        "d_ff": dff,
        "layers": layers,
        "heads": 1,
        "context": ctx,
        "train_loss": last,
        "vocab": vsz,
        "tokenizer": tok_spec,
        "tokenizer_sha256": tokenize.sha256(tok_spec),
        "tok": tok.data,
        "wout": wout.data,
        "blocks": [_pack(b) for b in blks],
        "n_windows": len(windows),
        "n_tokens": sum(len(w) for w in windows),
        "teacher_layers": teacher["layers"],
        "distill": True,
    }
    return out


def fit(src: Path, rec: dict) -> dict:
    texts, sources, weights = _docs(src, rec)
    if _opt(rec, "distill", False):
        model = _distill(texts, rec, sources, weights)
    else:
        size = _size(rec)
        model = _train_core(texts, rec, size, sources=sources, weights=weights)
        model.pop("windows", None)
    bits = _opt(rec, "quant", None)
    if bits is not None and bits is not False:
        b = 8 if bits in (True, "int8", "q8") else 4 if bits in ("int4", "q4", 4) else 1 if bits in ("binary", "int1", 1) else int(bits)
        model = _quant_model(model, b)
    frac = _opt(rec, "prune", None)
    if frac:
        model = _prune_model(model, float(frac))
    if _opt(rec, "speculative", False):
        model["speculative"] = True
        model["draft_layers"] = 1
    if _opt(rec, "paged_kv", False):
        model["paged_kv"] = True
        model["page_size"] = int(_opt(rec, "page_size", 4))
    if _opt(rec, "formats", False):
        model["formats"] = True
    return model


def write_inspect(train: Path, model: dict) -> str:
    tok = model.get("tokenizer") or {}
    digest = str(model.get("tokenizer_sha256") or tokenize.sha256(tok))
    tdir = train / "artifacts"
    tdir.mkdir(parents=True, exist_ok=True)
    dump_tokenizer(train, tok)
    extra = []
    if model.get("formats"):
        extra = _write_formats(train, model)
    lines = [
        "# llm",
        "",
        f"class: {model.get('class')}",
        "arch: decoder",
        f"objective: {model.get('objective') or 'next-token'}",
        f"causal: {str(bool(model.get('causal', True))).lower()}",
        f"n_pred: {model.get('n_pred')}",
        f"layers: {model.get('layers')}",
        f"heads: {model.get('heads')}",
        f"d_model: {model.get('d_model')}",
        f"d_ff: {model.get('d_ff')}",
        f"context: {model.get('context')}",
        f"vocab: {model.get('vocab')}",
        f"tokenizer: {tok.get('kind') or 'bpe'}",
        f"tokenizer_sha256: {digest}",
        f"train_loss: {model.get('train_loss')}",
        f"windows: {model.get('n_windows')}",
        f"pack: {model.get('pack') or 'eos'}",
        f"docs: {model.get('pack_docs')}",
    ]
    mix = model.get("mixture") or {}
    if mix:
        lines.append("mixture:")
        for k, v in mix.items():
            lines.append(f"  {k}: {v}")
    packed = model.get("packed_tokens") or {}
    if packed:
        lines.append("packed_tokens:")
        for k, v in packed.items():
            lines.append(f"  {k}: {v}")
    if model.get("quant"):
        lines.append(f"quant: {model.get('quant')}")
    if model.get("prune") is not None:
        lines.append(f"prune: {model.get('prune')}")
        lines.append(f"sparsity: {model.get('sparsity')}")
    if model.get("distill"):
        lines.append("distill: true")
        lines.append(f"teacher_layers: {model.get('teacher_layers')}")
    if model.get("speculative"):
        lines.append("speculative: true")
        lines.append(f"draft_layers: {model.get('draft_layers')}")
    if model.get("paged_kv"):
        lines.append("paged_kv: true")
        lines.append(f"page_size: {model.get('page_size')}")
    if extra:
        lines.append("formats:")
        for n in extra:
            lines.append("  " + n)
    lines.append("")
    rel = "artifacts/inspect.md"
    (train / rel).write_text("\n".join(lines), encoding="utf-8")
    return rel


def _speculative_rate(ids: list[int], tok, blks, wout) -> float:
    if len(ids) < 3:
        return 1.0
    logits = _decode_float(ids[:-1], tok, blks, wout)
    hit = 0
    n = 0
    for i in range(len(ids) - 2):
        draft = max(range(len(logits[i])), key=lambda j: logits[i][j])
        target = max(range(len(logits[i + 1])), key=lambda j: logits[i + 1][j])
        hit += int(draft == ids[i + 1])
        n += 1
        _ = target
    return hit / n if n else 1.0


def _paged_pages(n_tok: int, page: int) -> int:
    return (max(n_tok, 1) + page - 1) // page


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    texts, sources, weights = _docs(src, rec)
    if model.get("mixture"):
        weights = model["mixture"]
    tok_spec = model["tokenizer"]
    ctx = int(model.get("context") or 24)
    seed_i = int(model.get("pack_seed") or _opt(rec, "seed", 1))
    windows, _ = tokenize.pack_ex(texts, tok_spec, ctx, sources, weights, seed_i)
    tok, blks, wout = _load_float(model)
    total = 0.0
    n = 0
    accepts = []
    pages = 0
    page = int(model.get("page_size") or 4)
    for ids in windows:
        t, k = _nll_ids(ids, tok, blks, wout)
        total += t
        n += k
        if model.get("speculative"):
            accepts.append(_speculative_rate(ids, tok, blks, wout))
        if model.get("paged_kv"):
            pages += _paged_pages(len(ids), page)
    if n == 0:
        raise SystemExit("llm eval: no tokens")
    if model.get("speculative") and accepts:
        model["accept_rate"] = sum(accepts) / len(accepts)
    if model.get("paged_kv"):
        model["kv_pages"] = pages
    return total / n, n
