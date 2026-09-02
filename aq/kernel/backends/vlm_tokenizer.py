"""CLIP text tokenizer — BPE-free wordpiece built from train captions (offline-first).

Optional: set `text_tokenizer: openai/clip-vit-base-patch32` (or any HF tok) in recipe
to use a pretrained CLIP tokenizer instead.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


SOT = "<|startoftext|>"
EOT = "<|endoftext|>"
PAD = "<|pad|>"
UNK = "<|unk|>"


class SimpleClipTokenizer:
    """Whitespace + punctuation tokenizer with fixed context length (CLIP=77)."""

    def __init__(self, stoi: dict[str, int], context_length: int = 77):
        self.stoi = dict(stoi)
        self.itos = {i: t for t, i in self.stoi.items()}
        self.context_length = context_length
        self.sot_id = self.stoi[SOT]
        self.eot_id = self.stoi[EOT]
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.vocab_size = len(self.stoi)

    def tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        # split punctuation
        text = re.sub(r"([.,!?;:\"'()\[\]{}])", r" \1 ", text)
        return [t for t in text.split(" ") if t]

    def encode(self, text: str) -> list[int]:
        toks = self.tokenize(text)
        ids = [self.sot_id]
        for t in toks:
            ids.append(self.stoi.get(t, self.unk_id))
            if len(ids) >= self.context_length - 1:
                break
        ids.append(self.eot_id)
        if len(ids) < self.context_length:
            ids = ids + [self.pad_id] * (self.context_length - len(ids))
        return ids[: self.context_length]

    def batch_encode(self, texts: list[str]):
        import torch

        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"stoi": self.stoi, "context_length": self.context_length},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "SimpleClipTokenizer":
        blob = json.loads(path.read_text(encoding="utf-8"))
        return cls(blob["stoi"], context_length=int(blob.get("context_length") or 77))


def build_tokenizer_from_texts(
    texts: list[str],
    *,
    context_length: int = 77,
    max_vocab: int = 49408,
    min_freq: int = 1,
) -> SimpleClipTokenizer:
    counter: Counter[str] = Counter()
    tmp = SimpleClipTokenizer({SOT: 0, EOT: 1, PAD: 2, UNK: 3}, context_length)
    for t in texts:
        counter.update(tmp.tokenize(t))
    specials = [SOT, EOT, PAD, UNK]
    words = [w for w, c in counter.most_common(max_vocab - len(specials)) if c >= min_freq]
    stoi = {t: i for i, t in enumerate(specials + words)}
    return SimpleClipTokenizer(stoi, context_length=context_length)


def load_hf_clip_tokenizer(name: str, context_length: int = 77):
    """Optional HF CLIP tokenizer (requires network/cache)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    # wrap to match SimpleClipTokenizer.encode API

    class HFWrap:
        def __init__(self):
            self.inner = tok
            self.context_length = context_length
            self.vocab_size = len(tok)
            # CLIP uses sot/eot
            self.sot_id = tok.convert_tokens_to_ids(SOT) if SOT in tok.get_vocab() else tok.bos_token_id or 0
            self.eot_id = tok.convert_tokens_to_ids(EOT) if EOT in tok.get_vocab() else tok.eos_token_id or 0
            self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        def encode(self, text: str) -> list[int]:
            ids = self.inner(
                text,
                max_length=self.context_length,
                truncation=True,
                padding="max_length",
                return_tensors=None,
            )["input_ids"]
            return list(ids)

        def batch_encode(self, texts: list[str]):
            import torch

            return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)

        def save(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.inner.save_pretrained(path)

    return HFWrap()
