"""Tokenizer algorithms. Stdlib. kind: bpe | unigram | wordpiece | byte."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

BOS = "<b>"
EOS = "<e>"
UNK = "<u>"
MASK = "<m>"
SENT = "<x>"
PRE = "<P>"
SUF = "<S>"
MID = "<M>"


def _merge_once(seq: list[str], a: str, b: str, n: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            out.append(n)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(texts: list[str], merges: int) -> dict:
    corpus = [[c for c in t] + [EOS] for t in texts if t]
    if not corpus:
        raise SystemExit("tokenizer: empty texts")
    alphabet = sorted({c for seq in corpus for c in seq if c != EOS})
    itos = [BOS, EOS, UNK, MASK, SENT, PRE, SUF, MID] + alphabet
    stoi = {t: i for i, t in enumerate(itos)}
    pair_merges: list[list[str]] = []
    n_merges = max(0, int(merges))
    for k in range(n_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for seq in corpus:
            for x, y in zip(seq, seq[1:]):
                counts[(x, y)] += 1
        if not counts:
            break
        (a, b), c = counts.most_common(1)[0]
        if c < 2:
            break
        new = a + b
        if new in stoi:
            new = a + "\u0000" + b + str(k)
        pair_merges.append([a, b, new])
        stoi[new] = len(itos)
        itos.append(new)
        corpus = [_merge_once(seq, a, b, new) for seq in corpus]
    return {"kind": "bpe", "itos": itos, "merges": pair_merges}


def _encode_bpe(text: str, tok: dict) -> list[int]:
    stoi = {t: i for i, t in enumerate(tok["itos"])}
    unk = stoi.get(UNK, 0)
    seq = [c if c in stoi else UNK for c in text] + [EOS]
    for a, b, n in tok.get("merges") or []:
        seq = _merge_once(seq, a, b, n)
    return [stoi.get(s, unk) for s in seq]


def train_byte(_texts: list[str], **_k) -> dict:
    return {
        "kind": "byte",
        "bos": 0,
        "eos": 1,
        "byte_off": 2,
        "mask": 258,
        "span": 259,
        "prefix": 260,
        "suffix": 261,
        "middle": 262,
    }


def _encode_byte(text: str, tok: dict) -> list[int]:
    off = int(tok.get("byte_off") or 2)
    eos = int(tok.get("eos") or 1)
    return [off + b for b in text.encode("utf-8")] + [eos]


def vocab_size(tok: dict) -> int:
    k = str(tok.get("kind") or "bpe")
    if k == "byte":
        return int(tok.get("middle") or 262) + 1
    return len(tok.get("itos") or [])


def special_id(tok: dict, name: str) -> int:
    k = str(tok.get("kind") or "bpe")
    if k == "byte":
        defaults = {
            "bos": 0,
            "eos": 1,
            "mask": 258,
            "span": 259,
            "prefix": 260,
            "suffix": 261,
            "middle": 262,
        }
        return int(tok.get(name, defaults.get(name, 0)))
    names = {
        "bos": BOS,
        "eos": EOS,
        "unk": UNK,
        "mask": MASK,
        "span": SENT,
        "prefix": PRE,
        "suffix": SUF,
        "middle": MID,
    }
    piece = names.get(name, name)
    itos = tok.get("itos") or []
    if piece in itos:
        return itos.index(piece)
    return 0


def train_wordpiece(texts: list[str], merges: int) -> dict:
    words: list[list[str]] = []
    for t in texts:
        for w in t.split():
            if not w:
                continue
            pieces = [w[0]] + ["##" + c for c in w[1:]]
            words.append(pieces)
    if not words:
        raise SystemExit("wordpiece: empty texts")
    alphabet = sorted({p for seq in words for p in seq})
    itos = [BOS, EOS, UNK, MASK, SENT, PRE, SUF, MID] + alphabet
    stoi = {t: i for i, t in enumerate(itos)}
    pair_merges: list[list[str]] = []
    n_merges = max(0, int(merges))
    corpus = [seq[:] for seq in words]
    for k in range(n_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for seq in corpus:
            for x, y in zip(seq, seq[1:]):
                counts[(x, y)] += 1
        if not counts:
            break
        (a, b), c = counts.most_common(1)[0]
        if c < 2:
            break
        if b.startswith("##"):
            new = a + b[2:]
        else:
            new = a + b
        if a.startswith("##"):
            new = "##" + new.replace("##", "")
        if new in stoi:
            new = a + "\u0000" + b + str(k)
        pair_merges.append([a, b, new])
        stoi[new] = len(itos)
        itos.append(new)
        corpus = [_merge_once(seq, a, b, new) for seq in corpus]
    return {"kind": "wordpiece", "itos": itos, "merges": pair_merges}


def _encode_wordpiece(text: str, tok: dict) -> list[int]:
    stoi = {t: i for i, t in enumerate(tok["itos"])}
    unk = stoi.get(UNK, 0)
    ids: list[int] = []
    for word in text.split():
        i = 0
        first = True
        while i < len(word):
            hit = None
            for j in range(len(word), i, -1):
                piece = word[i:j] if first else "##" + word[i:j]
                if piece in stoi:
                    hit = (j, piece)
                    break
            if hit is None:
                ids.append(unk)
                i += 1
                first = False
            else:
                ids.append(stoi[hit[1]])
                i = hit[0]
                first = False
    ids.append(stoi.get(EOS, 1))
    return ids


def _ngrams(word: str, nmax: int) -> list[str]:
    out = list(word)
    for n in range(2, min(nmax, len(word)) + 1):
        for i in range(0, len(word) - n + 1):
            out.append(word[i : i + n])
    return out


def train_unigram(texts: list[str], vocab: int) -> dict:
    blob = "\n".join(t for t in texts if t)
    if not blob:
        raise SystemExit("unigram: empty texts")
    chars = sorted(set(blob))
    counts: Counter[str] = Counter()
    for t in texts:
        for w in t.split() or [t]:
            counts.update(_ngrams(w, 8))
        counts.update(c for c in t if c.isspace())
    for c in chars:
        counts[c] += 1
    pieces = [BOS, EOS, UNK, MASK, SENT, PRE, SUF, MID] + chars
    extras = [p for p, n in counts.most_common() if p not in pieces and n >= 2]
    want = max(len(pieces) + 1, int(vocab))
    for p in extras:
        if len(pieces) >= want:
            break
        pieces.append(p)
    total = sum(counts[p] for p in pieces if p not in {BOS, EOS, UNK}) or 1
    scores = []
    for p in pieces:
        if p in {BOS, EOS, UNK}:
            scores.append(-99.0)
        else:
            scores.append(math.log(max(counts[p], 1) / total))
    return {"kind": "unigram", "itos": pieces, "scores": scores}


def _encode_unigram(text: str, tok: dict) -> list[int]:
    itos = tok["itos"]
    scores = tok.get("scores") or [0.0] * len(itos)
    stoi = {t: i for i, t in enumerate(itos)}
    unk = stoi.get(UNK, 0)
    n = len(text)
    best = [-1e18] * (n + 1)
    prev = [-1] * (n + 1)
    piece_at = [""] * (n + 1)
    best[0] = 0.0
    maxlen = max((len(p) for p in itos if p not in {BOS, EOS, UNK}), default=1)
    for i in range(n):
        if best[i] < -1e17:
            continue
        for L in range(1, min(maxlen, n - i) + 1):
            p = text[i : i + L]
            j = stoi.get(p)
            if j is None:
                continue
            sc = best[i] + float(scores[j])
            if sc > best[i + L]:
                best[i + L] = sc
                prev[i + L] = i
                piece_at[i + L] = p
        if best[i + 1] < best[i] - 50:
            sc = best[i] + float(scores[unk]) if UNK in stoi else best[i] - 20
            if sc > best[i + 1]:
                best[i + 1] = sc
                prev[i + 1] = i
                piece_at[i + 1] = UNK
    ids: list[int] = []
    i = n
    if prev[n] < 0 and n:
        ids = [unk] * n
    else:
        path: list[str] = []
        while i > 0:
            p = piece_at[i]
            path.append(p)
            i = prev[i]
            if i < 0:
                break
        ids = [stoi.get(p, unk) for p in reversed(path)]
    ids.append(stoi.get(EOS, 1))
    return ids or [stoi.get(EOS, 1)]


def train(texts: list[str], rec: dict) -> dict:
    kind = str((rec.get("tokenizer") or rec.get("tok") or "bpe")).lower()
    kind = {"bytes": "byte", "byte-level": "byte", "wp": "wordpiece", "unigram-lm": "unigram"}.get(kind, kind)
    merges = int(rec.get("merges") if rec.get("merges") is not None else 24)
    vocab = int(rec.get("vocab") if rec.get("vocab") is not None else 48)
    nested = rec.get("llm")
    if isinstance(nested, dict):
        if rec.get("merges") is None and nested.get("merges") is not None:
            merges = int(nested["merges"])
        if rec.get("vocab") is None and nested.get("vocab") is not None:
            vocab = int(nested["vocab"])
        if rec.get("tokenizer") is None and nested.get("tokenizer"):
            kind = str(nested["tokenizer"]).lower()
    if kind == "byte":
        return train_byte(texts)
    if kind == "wordpiece":
        return train_wordpiece(texts, merges)
    if kind == "unigram":
        return train_unigram(texts, vocab)
    if kind != "bpe":
        raise SystemExit("tokenizer must be bpe, unigram, wordpiece, or byte")
    return train_bpe(texts, merges)


def encode(text: str, tok: dict) -> list[int]:
    k = str(tok.get("kind") or "bpe")
    if k == "byte":
        return _encode_byte(text, tok)
    if k == "wordpiece":
        return _encode_wordpiece(text, tok)
    if k == "unigram":
        return _encode_unigram(text, tok)
    return _encode_bpe(text, tok)


def _windows(stream: list[int], ctx: int) -> list[list[int]]:
    if len(stream) < 2:
        raise SystemExit("pack: not enough tokens")
    ctx = max(2, int(ctx))
    windows: list[list[int]] = []
    for i in range(0, len(stream) - 1, ctx):
        chunk = stream[i : i + ctx]
        if len(chunk) < 2:
            continue
        windows.append(chunk)
    if not windows:
        raise SystemExit("pack: no windows")
    return windows


def pack_ex(
    texts: list[str],
    tok: dict,
    ctx: int,
    sources: list[str] | None = None,
    weights: dict | None = None,
    seed: int = 1,
) -> tuple[list[list[int]], dict]:
    """Concat docs with the tokenizer EOS already on each doc. Fill context windows.

    If weights is set, docs are drawn by source mixture (one epoch).
    """
    import random
    from collections import defaultdict, deque

    if not texts:
        raise SystemExit("pack: empty texts")
    srcs = sources if sources is not None else ["default"] * len(texts)
    if len(srcs) != len(texts):
        raise SystemExit("pack: sources length must match texts")
    by: dict[str, deque[str]] = defaultdict(deque)
    for s, t in zip(srcs, texts):
        if t:
            by[str(s)].append(t)
    if not by:
        raise SystemExit("pack: empty texts")
    wraw = {str(k): float(v) for k, v in (weights or {}).items() if float(v) > 0}
    order: list[tuple[str, str]] = []
    if wraw:
        rng = random.Random(int(seed))
        left = {k: deque(v) for k, v in by.items()}
        while any(left.values()):
            keys = [k for k, q in left.items() if q]
            ww = [wraw.get(k, 0.0) for k in keys]
            if sum(ww) <= 0:
                keys = [k for k, q in left.items() if q]
                ww = [1.0] * len(keys)
            s = rng.choices(keys, ww)[0]
            order.append((s, left[s].popleft()))
    else:
        order = list(zip(srcs, texts))
    stream: list[int] = []
    tok_by: dict[str, int] = defaultdict(int)
    n_docs = 0
    for s, t in order:
        ids = encode(t, tok)
        if not ids:
            continue
        stream.extend(ids)
        tok_by[s] += len(ids)
        n_docs += 1
    windows = _windows(stream, ctx)
    mix = dict(wraw) if wraw else {k: 1.0 / len(by) for k in by}
    z = sum(mix.values()) or 1.0
    mix = {k: mix[k] / z for k in sorted(mix)}
    stats = {
        "pack": "eos",
        "context": max(2, int(ctx)),
        "docs": n_docs,
        "windows": len(windows),
        "tokens": len(stream),
        "mixture": mix,
        "packed_tokens": dict(sorted(tok_by.items())),
        "seed": int(seed),
    }
    return windows, stats


def pack(
    texts: list[str],
    tok: dict,
    ctx: int,
    sources: list[str] | None = None,
    weights: dict | None = None,
    seed: int = 1,
) -> list[list[int]]:
    windows, _ = pack_ex(texts, tok, ctx, sources, weights, seed)
    return windows


def sha256(tok: dict) -> str:
    blob = json.dumps(tok, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
