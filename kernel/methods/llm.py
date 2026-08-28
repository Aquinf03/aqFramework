"""LLM size class and deploy knobs. Decoder + BPE. Stdlib. Not a hub download."""

from __future__ import annotations

import json
import math
from pathlib import Path

from methods import tok as tokenize
from protocol.revision import hash_file
from protocol.tokenizer import dump as dump_tokenizer
from methods.linear import _rows
from methods.transformer import (
    Var,
    add,
    add_const,
    backward,
    block,
    embed,
    matmul,
    nll,
    nll_at,
    sgd,
    scale,
    cross_attn,
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


def _sft_pairs(src: Path, rec: dict) -> list[tuple[str, str]]:
    data = rec.get("data") or {}
    pf = str(data.get("prompt") or data.get("instruction") or "prompt")
    cf = str(data.get("completion") or data.get("output") or "completion")
    rows = _rows(src)
    pairs: list[tuple[str, str]] = []
    for r in rows:
        p = str(r.get(pf) or "").strip()
        c = str(r.get(cf) or "").strip()
        if p and c:
            pairs.append((p, c))
    if not pairs:
        raise SystemExit("sft: no prompt/completion rows")
    return pairs


def _sft_example(
    prompt: str,
    completion: str,
    tok_spec: dict,
    ctx: int,
    loss_on: str = "completion",
) -> tuple[list[int], list[tuple[int, int]]]:
    eos = tokenize.special_id(tok_spec, "eos")
    p_ids = tokenize.encode(prompt, tok_spec)
    if p_ids and p_ids[-1] == eos:
        p_ids = p_ids[:-1]
    c_ids = tokenize.encode(completion, tok_spec)
    if not c_ids:
        raise SystemExit("sft: empty completion tokens")
    seq = p_ids + c_ids
    p_len = len(p_ids)
    if len(seq) > ctx:
        drop = len(seq) - ctx
        seq = seq[drop:]
        p_len = max(p_len - drop, 0)
    pairs = [(i, seq[i + 1]) for i in range(len(seq) - 1)]
    if loss_on != "all":
        pairs = [(i, g) for i, g in pairs if i + 1 >= p_len]
    if not pairs:
        raise SystemExit("sft: no tokens to train in context")
    return seq, pairs


def _size(rec: dict) -> str:
    s = str(_opt(rec, "size", "llm")).lower()
    if s in ("tiny", "tinyml", "edge"):
        return "edge"
    if s not in PRESET:
        raise SystemExit("llm size must be llm, slm, or edge")
    return s


def _hidden(ids: list[int], tok: Var, blks: list[dict]) -> Var:
    x = add_const(embed(ids, tok), _pos(len(ids), len(tok.data[0])))
    mask = _causal_mask(len(ids))
    for blk in blks:
        x = block(x, blk, mask)
    return x


def _decode(ids: list[int], tok: Var, blks: list[dict], wout: Var) -> Var:
    return matmul(_hidden(ids, tok, blks), wout)


def _encode_tokens(ids: list[int], tok: Var, blks: list[dict], wout: Var) -> Var:
    x = add_const(embed(ids, tok), _pos(len(ids), len(tok.data[0])))
    for blk in blks:
        x = block(x, blk, None)
    return matmul(x, wout)


def _mask_pairs(ids: list[int], rate: float, seed: int, mid: int) -> tuple[list[int], list[tuple[int, int]]]:
    import random

    n = len(ids)
    cand = list(range(max(n - 1, 1)))
    rng = random.Random(seed + n + sum(ids))
    k = max(1, int(len(cand) * rate))
    k = min(k, len(cand))
    pos = sorted(rng.sample(cand, k))
    masked = ids[:]
    pairs = []
    for i in pos:
        pairs.append((i, ids[i]))
        masked[i] = mid
    return masked, pairs


def _span_corrupt(ids: list[int], seed: int, sid: int, bos: int, eos: int) -> tuple[list[int], list[int], list[int]]:
    import random

    rng = random.Random(seed + len(ids))
    core = ids[:-1] if len(ids) > 1 else ids
    if len(core) < 2:
        enc = [sid] + ([eos] if ids[-1:] == [eos] else [])
        tgt = core + [eos]
        return enc, [bos], tgt
    start = rng.randint(0, max(len(core) - 2, 0))
    sl = min(2, len(core) - start)
    span = core[start : start + sl]
    enc = core[:start] + [sid] + core[start + sl :]
    if ids and ids[-1] == eos:
        enc = enc + [eos]
    tin = [bos] + span
    tgt = span + [eos]
    return enc, tin, tgt


def _ed_tokens(src: list[int], tgt_in: list[int], tok: Var, enc: list[dict], dec: list[dict], xq, xk, xv, xo, wout: Var) -> Var:
    e = add_const(embed(src, tok), _pos(len(src), len(tok.data[0])))
    for blk in enc:
        e = block(e, blk, None)
    y = add_const(embed(tgt_in, tok), _pos(len(tgt_in), len(tok.data[0])))
    y = block(y, dec[0], _causal_mask(len(tgt_in)))
    y = add(y, cross_attn(y, e, xq, xk, xv, xo))
    return matmul(y, wout)


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


def _hidden_float(ids: list[int], tok, blks) -> list[list[float]]:
    d = len(tok[0])
    x = [tok[i][:] for i in ids]
    x = _add(x, _pos(len(ids), d))
    for blk in blks:
        x, _, _ = _blk_float(x, blk)
    return x


def _decode_float(ids: list[int], tok, blks, wout) -> list[list[float]]:
    return _mm(_hidden_float(ids, tok, blks), wout)


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


def _ckpt_path(src: Path, rec: dict) -> Path | None:
    init = rec.get("init") if isinstance(rec.get("init"), dict) else {}
    rel = init.get("checkpoint") or rec.get("from_ckpt")
    if not rel:
        return None
    p = Path(str(rel))
    if not p.is_absolute():
        p = (src.parent / p).resolve()
    if not p.is_file():
        raise SystemExit(f"init checkpoint not found: {rel} (train the parent first)")
    return p


def _blocks_from(parent: dict, layers: int) -> list[dict]:
    names = ["wq", "wk", "wv", "wo", "w1", "w2"]
    packs = parent.get("blocks") or []
    if len(packs) != layers:
        raise SystemExit("parent checkpoint layer count does not match")
    out = []
    for pack in packs:
        out.append({k: Var([row[:] for row in pack[k]]) for k in names})
    return out


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
    parent: dict | None = None,
    parent_rel: str | None = None,
    parent_sha: str | None = None,
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
    if parent:
        tok_spec = parent.get("tokenizer")
        if not isinstance(tok_spec, dict):
            raise SystemExit("parent checkpoint has no tokenizer")
        d = int(parent.get("d_model") or d)
        dff = int(parent.get("d_ff") or dff)
        layers = int(parent.get("layers") or layers)
        ctx = int(_opt(rec, "context", parent.get("context") or ctx))
        if _opt(rec, "lr", None) is None:
            lr = 0.02
    else:
        tok_spec = tokenize.train(texts, rec)
    seed_i = int(_opt(rec, "seed", 1))
    windows, pack_stats = tokenize.pack_ex(
        texts, tok_spec, ctx, sources, weights or None, seed_i
    )
    vsz = tokenize.vocab_size(tok_spec)
    if parent:
        tok = Var([row[:] for row in parent["tok"]])
        wout = Var([row[:] for row in parent["wout"]])
        blks = _blocks_from(parent, layers)
        if len(tok.data) != vsz or len(wout.data[0]) != vsz:
            raise SystemExit("parent vocab does not match tokenizer")
    else:
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
        "objective": "continued-pretrain" if parent else "next-token",
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
        **(
            {
                "parent": parent_rel,
                "parent_sha256": parent_sha,
            }
            if parent
            else {}
        ),
    }


def _fim_layout(ids: list[int], seed: int, pre: int, suf: int, mid: int, eos: int) -> list[int]:
    import random

    core = ids[:-1] if ids and ids[-1] == eos else list(ids)
    if len(core) < 4:
        return [pre] + core + [eos]
    rng = random.Random(seed + len(ids) + (sum(ids) % 10007))
    a = rng.randint(1, len(core) - 2)
    b = rng.randint(a + 1, len(core) - 1)
    prefix, middle, suffix = core[:a], core[a:b], core[b:]
    return [pre] + prefix + [suf] + suffix + [mid] + middle + [eos]


def _fim_windows(texts: list[str], tok_spec: dict, ctx: int, seed: int) -> list[list[int]]:
    pre = tokenize.special_id(tok_spec, "prefix")
    suf = tokenize.special_id(tok_spec, "suffix")
    mid = tokenize.special_id(tok_spec, "middle")
    eos = tokenize.special_id(tok_spec, "eos")
    stream: list[int] = []
    for t in texts:
        ids = tokenize.encode(t, tok_spec)
        if not ids:
            continue
        stream.extend(_fim_layout(ids, seed, pre, suf, mid, eos))
    return tokenize._windows(stream, ctx)


def _train_fim(texts: list[str], rec: dict, size: str, sources=None, weights=None) -> dict:
    preset = PRESET[size]
    d = int(_opt(rec, "d_model", preset["d_model"]))
    dff = int(_opt(rec, "d_ff", preset["d_ff"]))
    layers = int(_opt(rec, "layers", preset["layers"]))
    ctx = int(_opt(rec, "context", preset["context"]))
    steps = int(_opt(rec, "steps", 40))
    lr = float(_opt(rec, "lr", 0.05))
    seed = [int(_opt(rec, "seed", 1))]
    tok_spec = tokenize.train(texts, rec)
    seed_i = int(_opt(rec, "seed", 1))
    windows = _fim_windows(texts, tok_spec, ctx, seed_i)
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
        "objective": "fim",
        "causal": True,
        "fim_order": "psm",
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
        "pack": "fim-psm",
        "pack_docs": len(texts),
        "pack_seed": seed_i,
        "n_pred": n_pred,
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


def _train_mtp(texts: list[str], rec: dict, size: str, sources=None, weights=None) -> dict:
    preset = PRESET[size]
    d = int(_opt(rec, "d_model", preset["d_model"]))
    dff = int(_opt(rec, "d_ff", preset["d_ff"]))
    layers = int(_opt(rec, "layers", preset["layers"]))
    ctx = int(_opt(rec, "context", preset["context"]))
    steps = int(_opt(rec, "steps", 40))
    lr = float(_opt(rec, "lr", 0.05))
    n_p = int(_opt(rec, "n_predict", 2))
    if n_p < 2:
        raise SystemExit("mtp n_predict must be >= 2")
    seed = [int(_opt(rec, "seed", 1))]
    tok_spec = tokenize.train(texts, rec)
    windows, pack_stats = tokenize.pack_ex(
        texts, tok_spec, ctx, sources, weights or None, int(_opt(rec, "seed", 1))
    )
    vsz = tokenize.vocab_size(tok_spec)
    tok = Var(_rand(vsz, d, 0.2, seed))
    blks = [_init_block(d, dff, seed) for _ in range(layers)]
    wouts = [Var(_rand(d, vsz, 0.2, seed)) for _ in range(n_p)]
    params = [tok, *wouts]
    for blk in blks:
        params.extend(blk.values())
    last = 0.0
    n_pred = 0
    for _ in range(steps):
        total = 0.0
        n_pred = 0
        for ids in windows:
            if len(ids) < n_p + 1:
                continue
            h = _hidden(ids, tok, blks)
            loss = None
            n_here = 0
            for k in range(1, n_p + 1):
                pairs = [(i, ids[i + k]) for i in range(len(ids) - k)]
                if not pairs:
                    continue
                part = nll_at(matmul(h, wouts[k - 1]), pairs)
                loss = part if loss is None else add(loss, part)
                n_here += len(pairs)
            if loss is None:
                continue
            loss = scale(loss, 1.0 / n_p)
            backward(loss)
            sgd(params, lr)
            total += loss.data[0][0]
            n_pred += n_here
        last = total / max(len(windows), 1)
    return {
        "kind": "llm",
        "class": size,
        "arch": "decoder",
        "objective": "mtp",
        "causal": True,
        "n_predict": n_p,
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
        "wout": wouts[0].data,
        "wouts": [w.data for w in wouts],
        "blocks": [_pack(b) for b in blks],
        "n_windows": len(windows),
        "n_tokens": pack_stats["tokens"],
        "pack": pack_stats.get("pack"),
        "mixture": pack_stats.get("mixture"),
        "packed_tokens": pack_stats.get("packed_tokens"),
        "pack_docs": pack_stats.get("docs"),
        "pack_seed": pack_stats.get("seed"),
        "n_pred": n_pred,
    }


def _train_mlm(texts: list[str], rec: dict, size: str, sources=None, weights=None) -> dict:
    preset = PRESET[size]
    d = int(_opt(rec, "d_model", preset["d_model"]))
    dff = int(_opt(rec, "d_ff", preset["d_ff"]))
    layers = int(_opt(rec, "layers", preset["layers"]))
    ctx = int(_opt(rec, "context", preset["context"]))
    steps = int(_opt(rec, "steps", 40))
    lr = float(_opt(rec, "lr", 0.05))
    seed = [int(_opt(rec, "seed", 1))]
    rate = float(_opt(rec, "mask_rate", 0.3))
    tok_spec = tokenize.train(texts, rec)
    windows, pack_stats = tokenize.pack_ex(
        texts, tok_spec, ctx, sources, weights or None, int(_opt(rec, "seed", 1))
    )
    vsz = tokenize.vocab_size(tok_spec)
    mid = tokenize.special_id(tok_spec, "mask")
    tok = Var(_rand(vsz, d, 0.2, seed))
    blks = [_init_block(d, dff, seed) for _ in range(layers)]
    wout = Var(_rand(d, vsz, 0.2, seed))
    params = [tok, wout]
    for blk in blks:
        params.extend(blk.values())
    last = 0.0
    n_pred = 0
    for _ in range(steps):
        total = 0.0
        n_pred = 0
        for ids in windows:
            masked, pairs = _mask_pairs(ids, rate, int(_opt(rec, "seed", 1)), mid)
            logits = _encode_tokens(masked, tok, blks, wout)
            loss = nll_at(logits, pairs)
            backward(loss)
            sgd(params, lr)
            total += loss.data[0][0]
            n_pred += len(pairs)
        last = total / len(windows)
    return {
        "kind": "llm",
        "class": size,
        "arch": "encoder",
        "objective": "mlm",
        "causal": False,
        "mask_rate": rate,
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
        "n_tokens": pack_stats["tokens"],
        "pack": pack_stats.get("pack"),
        "mixture": pack_stats.get("mixture"),
        "packed_tokens": pack_stats.get("packed_tokens"),
        "pack_docs": pack_stats.get("docs"),
        "pack_seed": pack_stats.get("seed"),
        "n_pred": n_pred,
    }


def _train_span(texts: list[str], rec: dict, size: str, sources=None, weights=None) -> dict:
    preset = PRESET[size]
    d = int(_opt(rec, "d_model", preset["d_model"]))
    dff = int(_opt(rec, "d_ff", preset["d_ff"]))
    ctx = int(_opt(rec, "context", preset["context"]))
    steps = int(_opt(rec, "steps", 40))
    lr = float(_opt(rec, "lr", 0.05))
    seed = [int(_opt(rec, "seed", 1))]
    tok_spec = tokenize.train(texts, rec)
    windows, pack_stats = tokenize.pack_ex(
        texts, tok_spec, ctx, sources, weights or None, int(_opt(rec, "seed", 1))
    )
    vsz = tokenize.vocab_size(tok_spec)
    sid = tokenize.special_id(tok_spec, "span")
    bos = tokenize.special_id(tok_spec, "bos")
    eos = tokenize.special_id(tok_spec, "eos")
    tok = Var(_rand(vsz, d, 0.2, seed))
    enc = [_init_block(d, dff, seed)]
    dec = [_init_block(d, dff, seed)]
    xq = Var(_rand(d, d, 0.2, seed))
    xk = Var(_rand(d, d, 0.2, seed))
    xv = Var(_rand(d, d, 0.2, seed))
    xo = Var(_rand(d, d, 0.2, seed))
    wout = Var(_rand(d, vsz, 0.2, seed))
    params = [tok, wout, xq, xk, xv, xo, *enc[0].values(), *dec[0].values()]
    last = 0.0
    n_pred = 0
    for _ in range(steps):
        total = 0.0
        n_pred = 0
        for ids in windows:
            enc_in, tin, tgt = _span_corrupt(ids, int(_opt(rec, "seed", 1)), sid, bos, eos)
            logits = _ed_tokens(enc_in, tin, tok, enc, dec, xq, xk, xv, xo, wout)
            loss = nll(logits, tgt)
            backward(loss)
            sgd(params, lr)
            total += loss.data[0][0]
            n_pred += len(tgt)
        last = total / len(windows)
    return {
        "kind": "llm",
        "class": size,
        "arch": "encoder-decoder",
        "objective": "span",
        "causal": False,
        "d_model": d,
        "d_ff": dff,
        "layers": 1,
        "heads": 1,
        "context": ctx,
        "train_loss": last,
        "vocab": vsz,
        "tokenizer": tok_spec,
        "tokenizer_sha256": tokenize.sha256(tok_spec),
        "tok": tok.data,
        "wout": wout.data,
        "enc": [_pack(b) for b in enc],
        "dec": [_pack(b) for b in dec],
        "xq": xq.data,
        "xk": xk.data,
        "xv": xv.data,
        "xo": xo.data,
        "n_windows": len(windows),
        "n_tokens": pack_stats["tokens"],
        "pack": pack_stats.get("pack"),
        "mixture": pack_stats.get("mixture"),
        "packed_tokens": pack_stats.get("packed_tokens"),
        "pack_docs": pack_stats.get("docs"),
        "pack_seed": pack_stats.get("seed"),
        "n_pred": n_pred,
    }


def _train_sft(
    src: Path,
    rec: dict,
    size: str,
    parent: dict,
    parent_rel: str | None,
    parent_sha: str | None,
    loss_on: str = "completion",
) -> dict:
    pairs_txt = _sft_pairs(src, rec)
    tok_spec = parent.get("tokenizer")
    if not isinstance(tok_spec, dict):
        raise SystemExit("sft parent has no tokenizer")
    d = int(parent.get("d_model") or PRESET[size]["d_model"])
    dff = int(parent.get("d_ff") or PRESET[size]["d_ff"])
    layers = int(parent.get("layers") or PRESET[size]["layers"])
    ctx = int(_opt(rec, "context", parent.get("context") or PRESET[size]["context"]))
    steps = int(_opt(rec, "steps", 40))
    lr = float(_opt(rec, "lr", 0.05))
    examples = [_sft_example(p, c, tok_spec, ctx, loss_on) for p, c in pairs_txt]
    tok = Var([row[:] for row in parent["tok"]])
    wout = Var([row[:] for row in parent["wout"]])
    blks = _blocks_from(parent, layers)
    params = [tok, wout]
    for blk in blks:
        params.extend(blk.values())
    last = 0.0
    n_pred = 0
    for _ in range(steps):
        total = 0.0
        n_pred = 0
        for seq, pairs in examples:
            logits = _decode(seq[:-1], tok, blks, wout)
            # logits[i] predicts seq[i+1]; pairs use indices into full seq
            adj = [(i, g) for i, g in pairs if i < len(seq) - 1]
            if not adj:
                continue
            loss = nll_at(logits, adj)
            backward(loss)
            sgd(params, lr)
            total += loss.data[0][0]
            n_pred += len(adj)
        last = total / max(len(examples), 1)
    return {
        "kind": "llm",
        "class": size,
        "arch": "decoder",
        "objective": "full-ft" if loss_on == "all" else "sft",
        "causal": True,
        "loss_on": loss_on,
        "n_pairs": len(examples),
        "d_model": d,
        "d_ff": dff,
        "layers": layers,
        "heads": 1,
        "context": ctx,
        "train_loss": last,
        "vocab": tokenize.vocab_size(tok_spec),
        "tokenizer": tok_spec,
        "tokenizer_sha256": tokenize.sha256(tok_spec),
        "tok": tok.data,
        "wout": wout.data,
        "blocks": [_pack(b) for b in blks],
        "n_windows": len(examples),
        "n_pred": n_pred,
        "parent": parent_rel,
        "parent_sha256": parent_sha,
    }


def _decode_lora(ids: list[int], tok: Var, blks: list[dict], wout: Var, A: Var, B: Var, ab: float) -> Var:
    h = _hidden(ids, tok, blks)
    return add(matmul(h, wout), scale(matmul(matmul(h, A), B), ab))


def _decode_lora_float(
    ids: list[int], tok, blks, wout, A, B, ab: float
) -> list[list[float]]:
    h = _hidden_float(ids, tok, blks)
    base = _mm(h, wout)
    delta = _mm(_mm(h, A), B)
    return _add(base, [[x * ab for x in row] for row in delta])


def _pick_token(logits: list[float], temp: float, seed: int | None = None) -> int:
    if not logits:
        raise SystemExit("serve: empty logits")
    if temp <= 0:
        return max(range(len(logits)), key=lambda j: logits[j])
    import random

    rng = random.Random(seed)
    scaled = [x / temp for x in logits]
    mx = max(scaled)
    ex = [math.exp(x - mx) for x in scaled]
    z = sum(ex) or 1.0
    probs = [e / z for e in ex]
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1


def _serve_opts(rec: dict, max_tokens=None, temperature=None) -> tuple[int, float, int | None]:
    serve = rec.get("serve") if isinstance(rec.get("serve"), dict) else {}
    mt = max_tokens
    if mt is None:
        mt = serve.get("max_tokens")
    if mt is None:
        mt = _opt(rec, "max_tokens", 16)
    temp = temperature
    if temp is None:
        temp = serve.get("temperature")
    if temp is None:
        temp = _opt(rec, "temperature", 0)
    seed = serve.get("seed")
    if seed is None:
        seed = _opt(rec, "seed", None)
    return int(mt), float(temp), (int(seed) if seed is not None else None)


def generate(
    model: dict,
    prompt: str,
    rec: dict,
    max_tokens=None,
    temperature=None,
) -> dict:
    tok_spec = model.get("tokenizer")
    if not isinstance(tok_spec, dict):
        raise SystemExit("serve: checkpoint has no tokenizer")
    ctx = int(model.get("context") or 24)
    mt, temp, seed = _serve_opts(rec, max_tokens, temperature)
    if mt < 1:
        raise SystemExit("serve max_tokens must be >= 1")
    eos = tokenize.special_id(tok_spec, "eos")
    prompt = str(prompt)
    if not prompt.strip():
        raise SystemExit("serve needs a non-empty prompt")
    prompt_ids = tokenize.encode(prompt, tok_spec)
    if prompt_ids and prompt_ids[-1] == eos:
        prompt_ids = prompt_ids[:-1]
    if not prompt_ids:
        raise SystemExit("serve: prompt encodes to no tokens")
    obj = str(model.get("objective") or "next-token")
    tok, blks, wout = _load_float(model)
    lora = None
    if obj in ("lora", "qlora"):
        rank = int(model.get("rank") or 1)
        alpha = float(model.get("alpha") or rank)
        lora = (model["lora_A"], model["lora_B"], alpha / rank)
    ids = prompt_ids[:]
    import random

    rng = random.Random(seed if seed is not None else 0)
    for step in range(mt):
        if len(ids) >= ctx:
            break
        window = ids[-ctx:] if len(ids) > ctx else ids
        if lora is not None:
            A, B, ab = lora
            logits = _decode_lora_float(window, tok, blks, wout, A, B, ab)
        else:
            logits = _decode_float(window, tok, blks, wout)
        pick_seed = None if seed is None else seed + step + len(ids)
        if temp > 0 and seed is not None:
            next_id = _pick_token(logits[-1], temp, pick_seed)
        elif temp > 0:
            scaled = [x / temp for x in logits[-1]]
            mx = max(scaled)
            ex = [math.exp(x - mx) for x in scaled]
            z = sum(ex) or 1.0
            probs = [e / z for e in ex]
            r = rng.random()
            c = 0.0
            next_id = len(probs) - 1
            for i, p in enumerate(probs):
                c += p
                if r <= c:
                    next_id = i
                    break
        else:
            next_id = _pick_token(logits[-1], 0)
        if next_id == eos:
            break
        ids.append(next_id)
    completion_ids = ids[len(prompt_ids) :]
    text = tokenize.decode(ids, tok_spec)
    completion = tokenize.decode(completion_ids, tok_spec) if completion_ids else ""
    return {
        "prompt": prompt,
        "text": text,
        "completion": completion,
        "tokens": len(completion_ids),
        "max_tokens": mt,
        "temperature": temp,
        "objective": obj,
        "causal": bool(model.get("causal", True)),
    }


def _train_lora(
    src: Path,
    rec: dict,
    size: str,
    parent: dict,
    parent_rel: str | None,
    parent_sha: str | None,
    bits: int | None = None,
) -> dict:
    pairs_txt = _sft_pairs(src, rec)
    tok_spec = parent.get("tokenizer")
    if not isinstance(tok_spec, dict):
        raise SystemExit("lora parent has no tokenizer")
    d = int(parent.get("d_model") or PRESET[size]["d_model"])
    layers = int(parent.get("layers") or PRESET[size]["layers"])
    ctx = int(_opt(rec, "context", parent.get("context") or PRESET[size]["context"]))
    steps = int(_opt(rec, "steps", 40))
    lr = float(_opt(rec, "lr", 0.05))
    rank = int(_opt(rec, "rank", 4))
    alpha = float(_opt(rec, "alpha", 8))
    if rank < 1:
        raise SystemExit("lora rank must be >= 1")
    ab = alpha / rank
    examples = [_sft_example(p, c, tok_spec, ctx, "completion") for p, c in pairs_txt]
    qmeta = None
    base = parent
    if bits:
        q = _quant_model(
            {"tok": parent["tok"], "wout": parent["wout"], "blocks": parent["blocks"]},
            bits,
        )
        tok_f, blks_f, wout_f = _load_float(q)
        base = {**parent, "tok": tok_f, "wout": wout_f, "blocks": blks_f}
        qmeta = {"quant": q["quant"], "tok_q": q["tok_q"], "wout_q": q["wout_q"], "blocks_q": q["blocks_q"]}
    tok = Var([row[:] for row in base["tok"]])
    wout = Var([row[:] for row in base["wout"]])
    blks = _blocks_from(base, layers)
    seed = [int(_opt(rec, "seed", 1))]
    vsz = tokenize.vocab_size(tok_spec)
    A = Var(_rand(d, rank, 0.02, seed))
    B = Var([[0.0] * vsz for _ in range(rank)])
    adapters = [A, B]
    last = 0.0
    n_pred = 0
    for _ in range(steps):
        total = 0.0
        n_pred = 0
        for seq, pairs in examples:
            logits = _decode_lora(seq[:-1], tok, blks, wout, A, B, ab)
            adj = [(i, g) for i, g in pairs if i < len(seq) - 1]
            if not adj:
                continue
            loss = nll_at(logits, adj)
            backward(loss)
            sgd(adapters, lr)
            total += loss.data[0][0]
            n_pred += len(adj)
        last = total / max(len(examples), 1)
    out = {
        "kind": "llm",
        "class": size,
        "arch": "decoder",
        "objective": "qlora" if bits else "lora",
        "causal": True,
        "loss_on": "completion",
        "frozen": True,
        "rank": rank,
        "alpha": alpha,
        "n_pairs": len(examples),
        "d_model": d,
        "d_ff": parent.get("d_ff"),
        "layers": layers,
        "heads": 1,
        "context": ctx,
        "train_loss": last,
        "vocab": vsz,
        "tokenizer": tok_spec,
        "tokenizer_sha256": tokenize.sha256(tok_spec),
        "tok": tok.data,
        "wout": wout.data,
        "lora_A": A.data,
        "lora_B": B.data,
        "blocks": [_pack(b) for b in blks],
        "n_windows": len(examples),
        "n_pred": n_pred,
        "parent": parent_rel,
        "parent_sha256": parent_sha,
    }
    if bits:
        out["quant"] = f"int{bits}"
        if qmeta:
            out.update(qmeta)
    return out


def fit(src: Path, rec: dict) -> dict:
    obj = str(_opt(rec, "objective", "next-token")).lower().replace("_", "-")
    size = _size(rec)
    ckpt = _ckpt_path(src, rec)
    parent = None
    parent_rel = None
    parent_sha = None
    if ckpt is not None:
        parent = json.loads(ckpt.read_text(encoding="utf-8"))
        init = rec.get("init") if isinstance(rec.get("init"), dict) else {}
        parent_rel = str(init.get("checkpoint") or rec.get("from_ckpt"))
        parent_sha, _ = hash_file(ckpt)
    if obj in ("sft", "supervised", "supervised-finetune"):
        if parent is None:
            raise SystemExit("sft needs init.checkpoint")
        return _train_sft(src, rec, size, parent, parent_rel, parent_sha)
    if obj in ("full-ft", "full-finetune", "full-fine-tune"):
        if parent is None:
            raise SystemExit("full fine-tune needs init.checkpoint")
        return _train_sft(src, rec, size, parent, parent_rel, parent_sha, "all")
    if obj == "lora":
        if parent is None:
            raise SystemExit("lora needs init.checkpoint")
        return _train_lora(src, rec, size, parent, parent_rel, parent_sha)
    if obj == "qlora":
        if parent is None:
            raise SystemExit("qlora needs init.checkpoint")
        bits = int(_opt(rec, "bits", 4))
        return _train_lora(src, rec, size, parent, parent_rel, parent_sha, bits)
    texts, sources, weights = _docs(src, rec)
    if _opt(rec, "distill", False):
        model = _distill(texts, rec, sources, weights)
    elif obj in ("mlm", "masked", "masked-lm"):
        model = _train_mlm(texts, rec, size, sources, weights)
    elif obj in ("span", "span-corruption"):
        model = _train_span(texts, rec, size, sources, weights)
    elif obj in ("mtp", "multi-token", "multi-token-prediction"):
        model = _train_mtp(texts, rec, size, sources, weights)
    elif obj in ("fim", "fill-in-the-middle", "fill-in-middle"):
        model = _train_fim(texts, rec, size, sources, weights)
    elif parent or obj in ("continued-pretrain", "cpt", "continue"):
        if parent is None:
            raise SystemExit("continued pretrain needs init.checkpoint")
        model = _train_core(
            texts,
            rec,
            size,
            sources=sources,
            weights=weights,
            parent=parent,
            parent_rel=parent_rel,
            parent_sha=parent_sha,
        )
        model.pop("windows", None)
    else:
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
        f"arch: {model.get('arch') or 'decoder'}",
        f"objective: {model.get('objective') or 'next-token'}",
        f"causal: {str(bool(model.get('causal', True))).lower()}",
        f"n_pred: {model.get('n_pred')}",
    ]
    if model.get("n_predict") is not None:
        lines.append(f"n_predict: {model.get('n_predict')}")
    if model.get("fim_order"):
        lines.append(f"fim_order: {model.get('fim_order')}")
    if model.get("parent"):
        lines.append(f"parent: {model.get('parent')}")
        lines.append(f"parent_sha256: {model.get('parent_sha256')}")
    if model.get("loss_on"):
        lines.append(f"loss_on: {model.get('loss_on')}")
    if model.get("n_pairs") is not None:
        lines.append(f"n_pairs: {model.get('n_pairs')}")
    if model.get("rank") is not None:
        lines.append(f"rank: {model.get('rank')}")
        lines.append(f"alpha: {model.get('alpha')}")
        lines.append(f"frozen: {str(bool(model.get('frozen'))).lower()}")
    if model.get("mask_rate") is not None:
        lines.append(f"mask_rate: {model.get('mask_rate')}")
    lines += [
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


def _ce_pairs(logits: list[list[float]], pairs: list[tuple[int, int]]) -> tuple[float, int]:
    total = 0.0
    for i, gold in pairs:
        row = logits[i]
        mx = max(row)
        ex = [math.exp(x - mx) for x in row]
        z = sum(ex) or 1.0
        total += -math.log(max(ex[gold] / z, 1e-12))
    return total, len(pairs)


def _ce_seq(logits: list[list[float]], tgt: list[int]) -> tuple[float, int]:
    total = 0.0
    for i, gold in enumerate(tgt):
        row = logits[i]
        mx = max(row)
        ex = [math.exp(x - mx) for x in row]
        z = sum(ex) or 1.0
        total += -math.log(max(ex[gold] / z, 1e-12))
    return total, len(tgt)


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    obj = str(model.get("objective") or "next-token")
    if obj in ("sft", "full-ft", "lora", "qlora"):
        tok_spec = model["tokenizer"]
        ctx = int(model.get("context") or 24)
        names = ["wq", "wk", "wv", "wo", "w1", "w2"]
        tokv = Var(model["tok"])
        blks = [{k: Var([row[:] for row in pack[k]]) for k in names} for pack in model["blocks"]]
        wout = Var(model["wout"])
        loss_on = str(model.get("loss_on") or "completion")
        total = 0.0
        n = 0
        A = B = None
        ab = 1.0
        if obj in ("lora", "qlora"):
            A = Var(model["lora_A"])
            B = Var(model["lora_B"])
            rank = int(model.get("rank") or 1)
            alpha = float(model.get("alpha") or rank)
            ab = alpha / rank
        for p, c in _sft_pairs(src, rec):
            seq, pairs = _sft_example(p, c, tok_spec, ctx, loss_on)
            if obj in ("lora", "qlora") and A is not None and B is not None:
                logits = _decode_lora(seq[:-1], tokv, blks, wout, A, B, ab)
            else:
                logits = _decode(seq[:-1], tokv, blks, wout)
            t, k = _ce_pairs(logits.data, pairs)
            total += t
            n += k
        if n == 0:
            raise SystemExit("sft eval: no tokens")
        return total / n, n
    texts, sources, weights = _docs(src, rec)
    if model.get("mixture"):
        weights = model["mixture"]
    tok_spec = model["tokenizer"]
    ctx = int(model.get("context") or 24)
    seed_i = int(model.get("pack_seed") or _opt(rec, "seed", 1))
    windows, _ = tokenize.pack_ex(texts, tok_spec, ctx, sources, weights, seed_i)
    obj = str(model.get("objective") or "next-token")
    if obj == "fim":
        windows = _fim_windows(texts, tok_spec, ctx, seed_i)
    if obj == "mlm":
        names = ["wq", "wk", "wv", "wo", "w1", "w2"]
        tokv = Var(model["tok"])
        blks = [{k: Var([row[:] for row in pack[k]]) for k in names} for pack in model["blocks"]]
        wout = Var(model["wout"])
        mid = tokenize.special_id(tok_spec, "mask")
        rate = float(model.get("mask_rate") or 0.3)
        total = 0.0
        n = 0
        for ids in windows:
            masked, pairs = _mask_pairs(ids, rate, seed_i, mid)
            logits = _encode_tokens(masked, tokv, blks, wout)
            t, k = _ce_pairs(logits.data, pairs)
            total += t
            n += k
        if n == 0:
            raise SystemExit("mlm eval: no masks")
        return total / n, n
    if obj == "mtp":
        names = ["wq", "wk", "wv", "wo", "w1", "w2"]
        tokm = model["tok"]
        blks = [{k: pack[k] for k in names} for pack in model["blocks"]]
        wouts = model.get("wouts") or [model["wout"]]
        n_p = int(model.get("n_predict") or len(wouts))
        h = None
        total = 0.0
        n = 0
        for ids in windows:
            if len(ids) < 2:
                continue
            h = _hidden_float(ids, tokm, blks)
            for k in range(1, n_p + 1):
                logits = _mm(h, wouts[k - 1])
                for i in range(len(ids) - k):
                    gold = ids[i + k]
                    row = logits[i]
                    mx = max(row)
                    ex = [math.exp(x - mx) for x in row]
                    z = sum(ex) or 1.0
                    total += -math.log(max(ex[gold] / z, 1e-12))
                    n += 1
        if n == 0:
            raise SystemExit("mtp eval: no tokens")
        return total / n, n
    if obj == "span":
        names = ["wq", "wk", "wv", "wo", "w1", "w2"]
        tokv = Var(model["tok"])
        enc = [{k: Var([row[:] for row in pack[k]]) for k in names} for pack in model["enc"]]
        dec = [{k: Var([row[:] for row in pack[k]]) for k in names} for pack in model["dec"]]
        xq, xk, xv, xo = Var(model["xq"]), Var(model["xk"]), Var(model["xv"]), Var(model["xo"])
        wout = Var(model["wout"])
        sid = tokenize.special_id(tok_spec, "span")
        bos = tokenize.special_id(tok_spec, "bos")
        eos = tokenize.special_id(tok_spec, "eos")
        total = 0.0
        n = 0
        for ids in windows:
            enc_in, tin, tgt = _span_corrupt(ids, seed_i, sid, bos, eos)
            logits = _ed_tokens(enc_in, tin, tokv, enc, dec, xq, xk, xv, xo, wout)
            t, k = _ce_seq(logits.data, tgt)
            total += t
            n += k
        if n == 0:
            raise SystemExit("span eval: no tokens")
        return total / n, n
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
