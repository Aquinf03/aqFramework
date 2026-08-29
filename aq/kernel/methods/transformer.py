"""Tiny transformer. Stdlib. Encoder, decoder, encoder-decoder. No torch."""

from __future__ import annotations

from protocol import metrics as aq_metrics

import math
from pathlib import Path

from methods.linear import _rows


def _opt(rec: dict, key: str, default):
    v = rec.get(key)
    if v is None:
        nested = rec.get("transformer")
        if isinstance(nested, dict):
            v = nested.get(key)
    return default if v is None else v


def _zeros(r: int, c: int) -> list[list[float]]:
    return [[0.0] * c for _ in range(r)]


def _rand(r: int, c: int, scale: float, seed: list[int]) -> list[list[float]]:
    # one-stream LCG so fit is reproducible without random module state fights
    out = _zeros(r, c)
    s = seed[0]
    for i in range(r):
        for j in range(c):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF
            out[i][j] = ((s / 0x7FFFFFFF) * 2.0 - 1.0) * scale
    seed[0] = s
    return out


def _shape(a: list[list[float]]) -> tuple[int, int]:
    return len(a), len(a[0]) if a else 0


class Var:
    __slots__ = ("data", "grad", "_back", "_kids")

    def __init__(self, data: list[list[float]]):
        self.data = data
        self.grad = None
        self._back = lambda: None
        self._kids: tuple[Var, ...] = ()


def _zlike(a: list[list[float]]) -> list[list[float]]:
    return _zeros(len(a), len(a[0]) if a else 0)


def _ensure(v: Var) -> None:
    if v.grad is None:
        v.grad = _zlike(v.data)


def matmul(a: Var, b: Var) -> Var:
    m, k = _shape(a.data)
    n = _shape(b.data)[1]
    out = [
        [sum(a.data[i][t] * b.data[t][j] for t in range(k)) for j in range(n)]
        for i in range(m)
    ]
    v = Var(out)

    def back() -> None:
        _ensure(a)
        _ensure(b)
        g = v.grad
        for i in range(m):
            for t in range(k):
                s = 0.0
                for j in range(n):
                    s += g[i][j] * b.data[t][j]
                a.grad[i][t] += s
        for t in range(k):
            for j in range(n):
                s = 0.0
                for i in range(m):
                    s += a.data[i][t] * g[i][j]
                b.grad[t][j] += s

    v._back = back
    v._kids = (a, b)
    return v


def add(a: Var, b: Var) -> Var:
    m, n = _shape(a.data)
    out = [[a.data[i][j] + b.data[i][j] for j in range(n)] for i in range(m)]
    v = Var(out)

    def back() -> None:
        _ensure(a)
        _ensure(b)
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += v.grad[i][j]
                b.grad[i][j] += v.grad[i][j]

    v._back = back
    v._kids = (a, b)
    return v


def scale(a: Var, s: float) -> Var:
    m, n = _shape(a.data)
    out = [[a.data[i][j] * s for j in range(n)] for i in range(m)]
    v = Var(out)

    def back() -> None:
        _ensure(a)
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += v.grad[i][j] * s

    v._back = back
    v._kids = (a,)
    return v


def relu(a: Var) -> Var:
    m, n = _shape(a.data)
    out = [[a.data[i][j] if a.data[i][j] > 0 else 0.0 for j in range(n)] for i in range(m)]
    v = Var(out)

    def back() -> None:
        _ensure(a)
        for i in range(m):
            for j in range(n):
                if a.data[i][j] > 0:
                    a.grad[i][j] += v.grad[i][j]

    v._back = back
    v._kids = (a,)
    return v


def softmax_rows(a: Var) -> Var:
    m, n = _shape(a.data)
    out = _zeros(m, n)
    for i in range(m):
        mx = max(a.data[i])
        ex = [math.exp(a.data[i][j] - mx) for j in range(n)]
        z = sum(ex) or 1.0
        out[i] = [e / z for e in ex]
    v = Var(out)

    def back() -> None:
        _ensure(a)
        g = v.grad
        p = v.data
        for i in range(m):
            dot = sum(g[i][j] * p[i][j] for j in range(n))
            for j in range(n):
                a.grad[i][j] += p[i][j] * (g[i][j] - dot)

    v._back = back
    v._kids = (a,)
    return v


def embed(ids: list[int], w: Var) -> Var:
    t = len(ids)
    d = len(w.data[0])
    out = [w.data[i][:] for i in ids]
    v = Var(out)

    def back() -> None:
        _ensure(w)
        for p, i in enumerate(ids):
            for j in range(d):
                w.grad[i][j] += v.grad[p][j]

    v._back = back
    v._kids = (w,)
    return v


def mean_rows(a: Var) -> Var:
    m, n = _shape(a.data)
    out = [[sum(a.data[i][j] for i in range(m)) / m for j in range(n)]]
    v = Var(out)

    def back() -> None:
        _ensure(a)
        inv = 1.0 / m
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += v.grad[0][j] * inv

    v._back = back
    v._kids = (a,)
    return v


def add_const(a: Var, c: list[list[float]]) -> Var:
    m, n = _shape(a.data)
    out = [[a.data[i][j] + c[i][j] for j in range(n)] for i in range(m)]
    v = Var(out)

    def back() -> None:
        _ensure(a)
        for i in range(m):
            for j in range(n):
                a.grad[i][j] += v.grad[i][j]

    v._back = back
    v._kids = (a,)
    return v


def backward(loss: Var) -> None:
    seen: set[int] = set()
    order: list[Var] = []

    def walk(v: Var) -> None:
        i = id(v)
        if i in seen:
            return
        seen.add(i)
        for k in v._kids:
            walk(k)
        order.append(v)

    walk(loss)
    for v in order:
        v.grad = _zlike(v.data)
    loss.grad = [[1.0]]
    for v in reversed(order):
        v._back()


def sgd(params: list[Var], lr: float) -> None:
    for p in params:
        if p.grad is None:
            continue
        r, c = _shape(p.data)
        for i in range(r):
            for j in range(c):
                p.data[i][j] -= lr * p.grad[i][j]


def nll(logits: Var, targets: list[int]) -> Var:
    t = len(targets)
    n = len(logits.data[0])
    ps: list[list[float]] = []
    acc = 0.0
    for i in range(t):
        mx = max(logits.data[i])
        ex = [math.exp(logits.data[i][j] - mx) for j in range(n)]
        z = sum(ex) or 1.0
        p = [e / z for e in ex]
        ps.append(p)
        acc -= math.log(max(p[targets[i]], 1e-12))
    acc /= max(t, 1)
    loss = Var([[acc]])

    def back() -> None:
        _ensure(logits)
        g = loss.grad[0][0] / max(t, 1)
        for i in range(t):
            for j in range(n):
                logits.grad[i][j] += g * (ps[i][j] - (1.0 if j == targets[i] else 0.0))

    loss._back = back
    loss._kids = (logits,)
    return loss


def nll_at(logits: Var, pairs: list[tuple[int, int]]) -> Var:
    n = len(logits.data[0])
    t = len(pairs)
    if t < 1:
        raise SystemExit("nll_at: no positions")
    ps: list[list[float]] = []
    acc = 0.0
    for i, gold in pairs:
        mx = max(logits.data[i])
        ex = [math.exp(logits.data[i][j] - mx) for j in range(n)]
        z = sum(ex) or 1.0
        p = [e / z for e in ex]
        ps.append(p)
        acc -= math.log(max(p[gold], 1e-12))
    acc /= t
    loss = Var([[acc]])

    def back() -> None:
        _ensure(logits)
        g = loss.grad[0][0] / t
        for k, (i, gold) in enumerate(pairs):
            for j in range(n):
                logits.grad[i][j] += g * (ps[k][j] - (1.0 if j == gold else 0.0))

    loss._back = back
    loss._kids = (logits,)
    return loss


def _causal_mask(t: int) -> list[list[float]]:
    m = _zeros(t, t)
    for i in range(t):
        for j in range(t):
            if j > i:
                m[i][j] = -1e9
    return m


def attn(x: Var, wq: Var, wk: Var, wv: Var, wo: Var, mask: list[list[float]] | None) -> Var:
    q = matmul(x, wq)
    k = matmul(x, wk)
    v = matmul(x, wv)
    kt = Var([[k.data[i][j] for i in range(len(k.data))] for j in range(len(k.data[0]))])

    def back_t() -> None:
        _ensure(k)
        g = kt.grad
        for i in range(len(k.data)):
            for j in range(len(k.data[0])):
                k.grad[i][j] += g[j][i]

    kt._back = back_t
    kt._kids = (k,)
    d = len(q.data[0])
    scores = scale(matmul(q, kt), 1.0 / math.sqrt(d))
    if mask is not None:
        scores = add_const(scores, mask)
    a = softmax_rows(scores)
    ctx = matmul(a, v)
    return matmul(ctx, wo)


def cross_attn(qsrc: Var, kv: Var, wq: Var, wk: Var, wv: Var, wo: Var) -> Var:
    q = matmul(qsrc, wq)
    k = matmul(kv, wk)
    v = matmul(kv, wv)
    kt = Var([[k.data[i][j] for i in range(len(k.data))] for j in range(len(k.data[0]))])

    def back_t() -> None:
        _ensure(k)
        g = kt.grad
        for i in range(len(k.data)):
            for j in range(len(k.data[0])):
                k.grad[i][j] += g[j][i]

    kt._back = back_t
    kt._kids = (k,)
    d = len(q.data[0])
    scores = scale(matmul(q, kt), 1.0 / math.sqrt(d))
    a = softmax_rows(scores)
    ctx = matmul(a, v)
    return matmul(ctx, wo)


def ffn(x: Var, w1: Var, w2: Var) -> Var:
    return matmul(relu(matmul(x, w1)), w2)


def block(x: Var, p: dict, mask: list[list[float]] | None) -> Var:
    h = add(x, attn(x, p["wq"], p["wk"], p["wv"], p["wo"], mask))
    return add(h, ffn(h, p["w1"], p["w2"]))


def _pack(vs: dict[str, Var]) -> dict:
    return {k: [row[:] for row in v.data] for k, v in vs.items()}


def _load(pack: dict, names: list[str]) -> dict[str, Var]:
    return {k: Var([row[:] for row in pack[k]]) for k in names}


def _init_block(d: int, dff: int, seed: list[int]) -> dict[str, Var]:
    s = 0.2
    return {
        "wq": Var(_rand(d, d, s, seed)),
        "wk": Var(_rand(d, d, s, seed)),
        "wv": Var(_rand(d, d, s, seed)),
        "wo": Var(_rand(d, d, s, seed)),
        "w1": Var(_rand(d, dff, s, seed)),
        "w2": Var(_rand(dff, d, s, seed)),
    }


def _block_names() -> list[str]:
    return ["wq", "wk", "wv", "wo", "w1", "w2"]


def _vocab(texts: list[str]) -> tuple[list[str], dict[str, int]]:
    chars = sorted({c for t in texts for c in t})
    itos = ["<b>"] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return itos, stoi


def _ids(s: str, stoi: dict[str, int], cap: int) -> list[int]:
    ids = [stoi.get(c, 0) for c in s[:cap]]
    return ids if ids else [0]


def _texts_field(path: Path, field: str) -> list[str]:
    rows = _rows(path)
    if not rows:
        raise SystemExit(f"empty data: {path}")
    if field not in rows[0]:
        raise SystemExit(f"field {field!r} not in data")
    return [str(r[field]) for r in rows if str(r.get(field) or "").strip()]


def _pairs(path: Path, a: str, b: str) -> list[tuple[str, str]]:
    rows = _rows(path)
    if not rows:
        raise SystemExit(f"empty data: {path}")
    for k in (a, b):
        if k not in rows[0]:
            raise SystemExit(f"field {k!r} not in data")
    return [(str(r[a]), str(r[b])) for r in rows]


def _arch(rec: dict) -> str:
    a = str(_opt(rec, "arch", "decoder")).replace("_", "-")
    if a not in {"encoder", "decoder", "encoder-decoder"}:
        raise SystemExit("transformer arch must be encoder, decoder, or encoder-decoder")
    return a


def _pos(t: int, d: int) -> list[list[float]]:
    out = _zeros(t, d)
    for i in range(t):
        for j in range(d):
            if j % 2 == 0:
                out[i][j] = math.sin(i / (10000 ** (j / d)))
            else:
                out[i][j] = math.cos(i / (10000 ** ((j - 1) / d)))
    return out


def _dec_forward(ids: list[int], tok: Var, blk: dict, wout: Var) -> Var:
    x = add_const(embed(ids, tok), _pos(len(ids), len(tok.data[0])))
    h = block(x, blk, _causal_mask(len(ids)))
    return matmul(h, wout)


def _enc_forward(ids: list[int], tok: Var, blk: dict, wcls: Var) -> Var:
    x = add_const(embed(ids, tok), _pos(len(ids), len(tok.data[0])))
    h = block(x, blk, None)
    return matmul(mean_rows(h), wcls)


def _ed_forward(src: list[int], tgt_in: list[int], tok: Var, enc: dict, dec: dict, xq, xk, xv, xo, wout: Var) -> Var:
    e = add_const(embed(src, tok), _pos(len(src), len(tok.data[0])))
    e = block(e, enc, None)
    y = add_const(embed(tgt_in, tok), _pos(len(tgt_in), len(tok.data[0])))
    y = block(y, dec, _causal_mask(len(tgt_in)))
    y = add(y, cross_attn(y, e, xq, xk, xv, xo))
    return matmul(y, wout)


def fit(src: Path, rec: dict) -> dict:
    arch = _arch(rec)
    d = int(_opt(rec, "d_model", 8))
    dff = int(_opt(rec, "d_ff", d * 2))
    steps = int(_opt(rec, "steps", 80))
    lr = float(_opt(rec, "lr", 0.08))
    cap = int(_opt(rec, "seq", 12))
    seed = [int(_opt(rec, "seed", 1))]
    data = rec.get("data") or {}
    if d < 2:
        raise SystemExit("transformer d_model must be >= 2")
    if steps < 1:
        raise SystemExit("transformer steps must be >= 1")

    if arch == "decoder":
        field = str(data.get("text") or data.get("target") or "text")
        texts = _texts_field(src, field)
        itos, stoi = _vocab(texts)
        vsz = len(itos)
        tok = Var(_rand(vsz, d, 0.2, seed))
        blk = _init_block(d, dff, seed)
        wout = Var(_rand(d, vsz, 0.2, seed))
        params = [tok, wout, *blk.values()]
        seqs = [_ids(t, stoi, cap) for t in texts]
        seqs = [s for s in seqs if len(s) >= 2]
        if not seqs:
            raise SystemExit("transformer decoder: texts too short")
        last = 0.0
        for step_i in range(steps):
            total = 0.0
            for ids in seqs:
                logits = _dec_forward(ids[:-1], tok, blk, wout)
                loss = nll(logits, ids[1:])
                backward(loss)
                sgd(params, lr)
                total += loss.data[0][0]
            last = total / len(seqs)
            aq_metrics.step(step=step_i, loss=last, lr=lr)
        return {
            "kind": "transformer",
            "arch": arch,
            "d_model": d,
            "d_ff": dff,
            "layers": 1,
            "heads": 1,
            "seq": cap,
            "itos": itos,
            "stoi": stoi,
            "train_loss": last,
            "tok": tok.data,
            "wout": wout.data,
            "block": _pack(blk),
        }

    if arch == "encoder":
        text_f = str(data.get("text") or "text")
        y_f = str(data.get("target") or "y")
        pairs = _pairs(src, text_f, y_f)
        itos, stoi = _vocab([a for a, _ in pairs])
        classes = sorted({b for _, b in pairs})
        ctoi = {c: i for i, c in enumerate(classes)}
        vsz = len(itos)
        ncls = len(classes)
        tok = Var(_rand(vsz, d, 0.2, seed))
        blk = _init_block(d, dff, seed)
        wcls = Var(_rand(d, ncls, 0.2, seed))
        params = [tok, wcls, *blk.values()]
        last = 0.0
        for step_i in range(steps):
            total = 0.0
            for text, lab in pairs:
                ids = _ids(text, stoi, cap)
                logits = _enc_forward(ids, tok, blk, wcls)
                loss = nll(logits, [ctoi[lab]])
                backward(loss)
                sgd(params, lr)
                total += loss.data[0][0]
            last = total / len(pairs)
            aq_metrics.step(step=step_i, loss=last, lr=lr)
        return {
            "kind": "transformer",
            "arch": arch,
            "d_model": d,
            "d_ff": dff,
            "layers": 1,
            "heads": 1,
            "seq": cap,
            "itos": itos,
            "stoi": stoi,
            "classes": classes,
            "train_loss": last,
            "tok": tok.data,
            "wcls": wcls.data,
            "block": _pack(blk),
        }

    src_f = str(data.get("src") or "src")
    tgt_f = str(data.get("tgt") or "tgt")
    pairs = _pairs(src, src_f, tgt_f)
    itos, stoi = _vocab([a + b for a, b in pairs])
    vsz = len(itos)
    bos = 0
    tok = Var(_rand(vsz, d, 0.2, seed))
    enc = _init_block(d, dff, seed)
    dec = _init_block(d, dff, seed)
    xq = Var(_rand(d, d, 0.2, seed))
    xk = Var(_rand(d, d, 0.2, seed))
    xv = Var(_rand(d, d, 0.2, seed))
    xo = Var(_rand(d, d, 0.2, seed))
    wout = Var(_rand(d, vsz, 0.2, seed))
    params = [tok, wout, xq, xk, xv, xo, *enc.values(), *dec.values()]
    last = 0.0
    for step_i in range(steps):
        total = 0.0
        for a, b in pairs:
            sids = _ids(a, stoi, cap)
            tids = _ids(b, stoi, cap)
            tin = [bos] + tids[:-1]
            logits = _ed_forward(sids, tin, tok, enc, dec, xq, xk, xv, xo, wout)
            loss = nll(logits, tids)
            backward(loss)
            sgd(params, lr)
            total += loss.data[0][0]
        last = total / len(pairs)
        aq_metrics.step(step=step_i, loss=last, lr=lr)
    return {
        "kind": "transformer",
        "arch": arch,
        "d_model": d,
        "d_ff": dff,
        "layers": 1,
        "heads": 1,
        "seq": cap,
        "itos": itos,
        "stoi": stoi,
        "train_loss": last,
        "tok": tok.data,
        "wout": wout.data,
        "enc": _pack(enc),
        "dec": _pack(dec),
        "xq": xq.data,
        "xk": xk.data,
        "xv": xv.data,
        "xo": xo.data,
    }


def write_inspect(train: Path, model: dict) -> str:
    lines = [
        "# transformer",
        "",
        f"arch: {model.get('arch')}",
        f"layers: {model.get('layers')}",
        f"heads: {model.get('heads')}",
        f"d_model: {model.get('d_model')}",
        f"d_ff: {model.get('d_ff')}",
        f"ffn: relu",
        f"attn: mha",
        f"vocab: {len(model.get('itos') or [])}",
        f"train_loss: {model.get('train_loss')}",
        "",
    ]
    if model.get("classes"):
        lines.append("classes: " + ", ".join(str(c) for c in model["classes"]))
        lines.append("")
    rel = "artifacts/inspect.md"
    dest = train / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return rel


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    arch = str(model.get("arch") or "decoder")
    data = rec.get("data") or {}
    cap = int(model.get("seq") or 12)
    stoi = model["stoi"]
    metric = str((rec.get("eval") or {}).get("metric") or "loss")
    tok = Var(model["tok"])

    if arch == "decoder":
        field = str(data.get("text") or data.get("target") or "text")
        texts = _texts_field(src, field)
        blk = {k: Var(model["block"][k]) for k in _block_names()}
        wout = Var(model["wout"])
        total = 0.0
        n = 0
        hits = 0
        for t in texts:
            ids = _ids(t, stoi, cap)
            if len(ids) < 2:
                continue
            logits = _dec_forward(ids[:-1], tok, blk, wout)
            for i, gold in enumerate(ids[1:]):
                pred = max(range(len(logits.data[i])), key=lambda j: logits.data[i][j])
                hits += int(pred == gold)
                row = logits.data[i]
                mx = max(row)
                ex = [math.exp(x - mx) for x in row]
                z = sum(ex)
                total += -math.log(max(ex[gold] / z, 1e-12))
                n += 1
        if n == 0:
            raise SystemExit("transformer eval: no tokens")
        if metric == "accuracy":
            return hits / n, n
        return total / n, n

    if arch == "encoder":
        text_f = str(data.get("text") or "text")
        y_f = str(data.get("target") or "y")
        pairs = _pairs(src, text_f, y_f)
        blk = {k: Var(model["block"][k]) for k in _block_names()}
        wcls = Var(model["wcls"])
        classes = model["classes"]
        ctoi = {c: i for i, c in enumerate(classes)}
        hits = 0
        total = 0.0
        for text, lab in pairs:
            ids = _ids(text, stoi, cap)
            logits = _enc_forward(ids, tok, blk, wcls)
            gold = ctoi[str(lab)]
            pred = max(range(len(logits.data[0])), key=lambda j: logits.data[0][j])
            hits += int(pred == gold)
            row = logits.data[0]
            mx = max(row)
            ex = [math.exp(x - mx) for x in row]
            z = sum(ex)
            total += -math.log(max(ex[gold] / z, 1e-12))
        n = len(pairs)
        if metric == "loss":
            return total / n, n
        return hits / n, n

    src_f = str(data.get("src") or "src")
    tgt_f = str(data.get("tgt") or "tgt")
    pairs = _pairs(src, src_f, tgt_f)
    enc = {k: Var(model["enc"][k]) for k in _block_names()}
    dec = {k: Var(model["dec"][k]) for k in _block_names()}
    xq, xk, xv, xo = Var(model["xq"]), Var(model["xk"]), Var(model["xv"]), Var(model["xo"])
    wout = Var(model["wout"])
    bos = 0
    hits = 0
    total = 0.0
    ntok = 0
    for a, b in pairs:
        sids = _ids(a, stoi, cap)
        tids = _ids(b, stoi, cap)
        tin = [bos] + tids[:-1]
        logits = _ed_forward(sids, tin, tok, enc, dec, xq, xk, xv, xo, wout)
        for i, gold in enumerate(tids):
            pred = max(range(len(logits.data[i])), key=lambda j: logits.data[i][j])
            hits += int(pred == gold)
            row = logits.data[i]
            mx = max(row)
            ex = [math.exp(x - mx) for x in row]
            z = sum(ex)
            total += -math.log(max(ex[gold] / z, 1e-12))
            ntok += 1
    if ntok == 0:
        raise SystemExit("transformer eval: no tokens")
    if metric == "accuracy":
        return hits / ntok, ntok
    return total / ntok, ntok
