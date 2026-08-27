"""Byte-pair tokenizer. Stdlib. Lives in the train and is hashed with the checkpoint."""

from __future__ import annotations

import hashlib
import json
from collections import Counter


BOS = "<b>"
EOS = "<e>"


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
        raise SystemExit("bpe: empty texts")
    alphabet = sorted({c for seq in corpus for c in seq if c != EOS})
    itos = [BOS, EOS] + alphabet
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


def encode(text: str, tok: dict) -> list[int]:
    stoi = {t: i for i, t in enumerate(tok["itos"])}
    seq = [c if c in stoi else BOS for c in text] + [EOS]
    for a, b, n in tok["merges"]:
        seq = _merge_once(seq, a, b, n)
    return [stoi.get(s, 0) for s in seq]


def pack(texts: list[str], tok: dict, ctx: int) -> list[list[int]]:
    stream: list[int] = []
    for t in texts:
        ids = encode(t, tok)
        if not ids:
            continue
        stream.extend(ids)
    if len(stream) < 2:
        raise SystemExit("llm pack: not enough tokens")
    ctx = max(2, int(ctx))
    windows: list[list[int]] = []
    for i in range(0, len(stream) - 1, ctx):
        chunk = stream[i : i + ctx]
        if len(chunk) < 2:
            continue
        windows.append(chunk)
    if not windows:
        raise SystemExit("llm pack: no windows")
    return windows


def sha256(tok: dict) -> str:
    blob = json.dumps(
        {"kind": tok.get("kind"), "itos": tok.get("itos"), "merges": tok.get("merges")},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
