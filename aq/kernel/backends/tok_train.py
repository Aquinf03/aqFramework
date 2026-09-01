"""Train / pin tokenizers in the train (BPE, Unigram, WordPiece, byte)."""

from __future__ import annotations

import json
from pathlib import Path

from backends.recipe_opt import opt


def train_tokenizer(texts: list[str], rec: dict, dest: Path) -> dict:
    """Fit a tokenizer from texts; write HF-compatible files under dest."""
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    except ImportError as e:
        raise SystemExit(
            "tokenizers package required. pip install tokenizers"
        ) from e

    algo = str(opt(rec, "tokenizer", opt(rec, "tok", "bpe"), "llm") or "bpe").lower()
    merges = int(opt(rec, "merges", 1000, "llm") or 1000)
    vocab = int(opt(rec, "vocab", 0, "llm") or 0)
    dest.mkdir(parents=True, exist_ok=True)

    special = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "<fim_prefix>", "<fim_middle>", "<fim_suffix>"]

    if algo in ("byte", "bytes", "byte-level", "bytelevel"):
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=max(vocab, merges + len(special), 256),
            special_tokens=special,
            show_progress=False,
        )
        algo_name = "byte"
    elif algo in ("unigram",):
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.UnigramTrainer(
            vocab_size=max(vocab, 500),
            special_tokens=special,
            show_progress=False,
        )
        algo_name = "unigram"
    elif algo in ("wordpiece", "wp"):
        tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordPieceTrainer(
            vocab_size=max(vocab, merges + len(special), 500),
            special_tokens=special,
            show_progress=False,
        )
        algo_name = "wordpiece"
    else:
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=max(vocab, merges + len(special), 500),
            special_tokens=special,
            show_progress=False,
        )
        algo_name = "bpe"

    tok.train_from_iterator(texts, trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tok.token_to_id("[BOS]")), ("[EOS]", tok.token_to_id("[EOS]"))],
    )
    out = dest / "tokenizer.json"
    tok.save(str(out))
    # Also write a tiny wrapper so transformers can load if needed
    meta = {
        "algo": algo_name,
        "vocab_size": tok.get_vocab_size(),
        "path": "tokenizer.json",
    }
    (dest / "aq_tokenizer_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def ensure_mask_token(tok) -> bool:
    """Add [MASK] when missing (needed for mlm on causal tokenizers). Returns True if vocab grew."""
    if tok.mask_token is not None and tok.mask_token_id is not None:
        return False
    tok.add_special_tokens({"mask_token": "[MASK]"})
    return True


def ensure_pad_token(tok) -> bool:
    """Ensure tokenizer can pad batches. Returns True if vocab grew (caller should resize embeddings)."""
    if tok.pad_token is not None:
        return False
    if tok.eos_token is not None:
        tok.pad_token = tok.eos_token
        return False
    if tok.unk_token is not None:
        tok.pad_token = tok.unk_token
        return False
    tok.add_special_tokens({"pad_token": "[PAD]"})
    return True


def load_hf_tokenizer(model_id: str, local_tok_dir: Path | None = None) -> tuple:
    """Load HF tokenizer. Returns (tokenizer, resize_embeddings)."""
    from backends.deps import require_transformers

    transformers = require_transformers()
    if local_tok_dir and (local_tok_dir / "tokenizer.json").is_file():
        try:
            tok = transformers.PreTrainedTokenizerFast(
                tokenizer_file=str(local_tok_dir / "tokenizer.json"),
                bos_token="[BOS]",
                eos_token="[EOS]",
                pad_token="[PAD]",
                unk_token="[UNK]",
                mask_token="[MASK]",
            )
            return tok, False
        except Exception:
            pass
    tok = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tok, ensure_pad_token(tok)
