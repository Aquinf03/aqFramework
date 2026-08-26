"""LoRA on a tiny causal LM. Stdlib path so aq train works without torch.

If torch+transformers+peft are installed and recipe.model is a hub id, that path can replace this later.
This is the first working LLM LoRA step: frozen weights, trained A/B, next-token loss.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

V = 128
D = 16
R = 4
CTX = 32
LR = 0.05
EPOCHS = 25
SCALE = 2.0  # alpha / r


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_texts(path: Path, field: str) -> list[str]:
    if path.suffix == ".jsonl":
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            t = obj.get(field)
            if t:
                out.append(str(t))
        if not out:
            raise SystemExit(f"no {field!r} in jsonl")
        return out
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty csv: {path}")
    if field not in rows[0]:
        raise SystemExit(f"text field {field!r} not in csv")
    return [str(r[field]) for r in rows if str(r[field]).strip()]


def _ids(s: str) -> list[int]:
    return [min(ord(c), V - 1) for c in s[:CTX]]


def _zeros(n: int, m: int) -> list[list[float]]:
    return [[0.0] * m for _ in range(n)]


def _rand(n: int, m: int, rng: random.Random, s: float) -> list[list[float]]:
    return [[rng.gauss(0.0, s) for _ in range(m)] for _ in range(n)]


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    z = sum(ex)
    return [e / z for e in ex]


def _forward(W_in, W_out, A, B, tok: int) -> list[float]:
    h = W_in[tok]
    logits = [_dot(h, W_out[j]) for j in range(V)]
    # LoRA: h @ A @ B, A is D x r, B is r x V stored as B[k][v]
    z = [_dot(h, [A[d][k] for d in range(D)]) for k in range(R)]
    for v in range(V):
        logits[v] += SCALE * sum(z[k] * B[k][v] for k in range(R))
    return logits


def _step(W_in, W_out, A, B, prev: int, nxt: int) -> float:
    logits = _forward(W_in, W_out, A, B, prev)
    p = _softmax(logits)
    loss = -math.log(max(p[nxt], 1e-12))
    # dL/dlogits
    g = p[:]
    g[nxt] -= 1.0
    h = W_in[prev]
    z = [_dot(h, [A[d][k] for d in range(D)]) for k in range(R)]
    # B: r x V
    for k in range(R):
        for v in range(V):
            B[k][v] -= LR * SCALE * z[k] * g[v]
    # A: D x r ; dL/dA[d,k] = SCALE * h[d] * sum_v B[k][v] * g[v]
    for k in range(R):
        acc = sum(B[k][v] * g[v] for v in range(V))
        for d in range(D):
            A[d][k] -= LR * SCALE * h[d] * acc
    return loss


def fit(csv_path: Path, rec: dict) -> dict:
    data = rec.get("data") or {}
    field = str(data.get("text") or data.get("target") or "text")
    texts = load_texts(csv_path, field)
    rng = random.Random(0)
    W_in = _rand(V, D, rng, 0.1)
    W_out = _rand(V, D, rng, 0.1)  # rows = vocab, cols = D; logits[v] = h · W_out[v]
    A = _zeros(D, R)
    B = _zeros(R, V)
    pairs = []
    for t in texts:
        ids = _ids(t)
        for i in range(len(ids) - 1):
            pairs.append((ids[i], ids[i + 1]))
    if not pairs:
        raise SystemExit("lora: texts too short")
    last = 0.0
    for _ in range(EPOCHS):
        rng.shuffle(pairs)
        total = 0.0
        for prev, nxt in pairs:
            total += _step(W_in, W_out, A, B, prev, nxt)
        last = total / len(pairs)
    return {
        "kind": "lora",
        "backend": "tiny",
        "task": "lm",
        "r": R,
        "d": D,
        "v": V,
        "scale": SCALE,
        "W_in": W_in,
        "W_out": W_out,
        "A": A,
        "B": B,
        "train_loss": last,
        "n_pairs": len(pairs),
    }


def evaluate(model: dict, csv_path: Path, rec: dict) -> tuple[float, int]:
    data = rec.get("data") or {}
    field = str(data.get("text") or data.get("target") or "text")
    texts = load_texts(csv_path, field)
    W_in, W_out, A, B = model["W_in"], model["W_out"], model["A"], model["B"]
    total = 0.0
    n = 0
    for t in texts:
        ids = _ids(t)
        for i in range(len(ids) - 1):
            logits = _forward(W_in, W_out, A, B, ids[i])
            p = _softmax(logits)
            total += -math.log(max(p[ids[i + 1]], 1e-12))
            n += 1
    if n == 0:
        raise SystemExit("lora eval: no pairs")
    return total / n, n
