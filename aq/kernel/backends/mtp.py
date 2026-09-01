"""Multi-token prediction (MTP) auxiliary heads on a causal LM backbone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backends.deps import require_torch
from backends.recipe_opt import opt


def n_predict_from_rec(rec: dict) -> int:
    n = int(opt(rec, "n_predict", 2, "llm") or 2)
    if n < 2:
        raise SystemExit("objective mtp needs n_predict >= 2 (predict t+1, t+2, …)")
    if n > 8:
        raise SystemExit("objective mtp supports n_predict 2–8")
    return n


def build_mtp_rows(tok, texts: list[str], max_len: int, n_predict: int) -> list[dict]:
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows: list[dict] = []
    for text in texts:
        enc = tok(text, truncation=True, max_length=max_len, padding=False)
        ids = list(enc["input_ids"])
        attn = list(enc["attention_mask"])
        L = len(ids)
        label_rows: list[list[int]] = []
        for i in range(L):
            shifts: list[int] = []
            for k in range(n_predict):
                j = i + k + 1
                if j >= L or ids[j] == pad_id:
                    shifts.append(-100)
                else:
                    shifts.append(int(ids[j]))
            label_rows.append(shifts)
        rows.append({"input_ids": ids, "attention_mask": attn, "labels": label_rows})
    if not rows:
        raise SystemExit("empty mtp data")
    return rows


def mtp_collator(tok):
    torch = require_torch()

    def collate(features: list[dict]) -> dict[str, Any]:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        max_len = max(len(f["input_ids"]) for f in features)
        n_predict = len(features[0]["labels"][0])
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_n)
            attention_mask.append(f["attention_mask"] + [0] * pad_n)
            labels.append(f["labels"] + [[-100] * n_predict] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def wrap_mtp(model, n_predict: int):
    torch = require_torch()
    from torch import nn
    from transformers.modeling_outputs import CausalLMOutput

    class MTPWrapper(nn.Module):
        def __init__(self, inner, n_predict: int):
            super().__init__()
            self.inner = inner
            self.n_predict = n_predict
            cfg = inner.config
            self.extra_heads = nn.ModuleList(
                [nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) for _ in range(n_predict - 1)]
            )
            for head in self.extra_heads:
                nn.init.normal_(head.weight, std=getattr(cfg, "initializer_range", 0.02))

        @property
        def config(self):
            return self.inner.config

        def gradient_checkpointing_enable(self, **kwargs):
            if hasattr(self.inner, "gradient_checkpointing_enable"):
                self.inner.gradient_checkpointing_enable(**kwargs)

        def save_pretrained(self, dest: str, **kwargs):
            save_mtp(Path(dest), self)

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            kwargs.pop("labels", None)
            kwargs["output_hidden_states"] = True
            kwargs["use_cache"] = False
            outputs = self.inner(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            hidden = outputs.hidden_states[-1]
            logits_main = self.inner.lm_head(hidden)

            if labels is None:
                return CausalLMOutput(logits=logits_main, hidden_states=outputs.hidden_states)

            import torch.nn.functional as F

            total = logits_main.new_zeros(())
            n_heads = 0
            for k in range(self.n_predict):
                logits_k = logits_main if k == 0 else self.extra_heads[k - 1](hidden)
                lab = labels[:, :, k].reshape(-1)
                flat = logits_k.reshape(-1, logits_k.size(-1))
                valid = lab != -100
                if valid.any():
                    total = total + F.cross_entropy(flat[valid], lab[valid])
                    n_heads += 1
            loss = total / max(n_heads, 1)
            return CausalLMOutput(loss=loss, logits=logits_main)

    return MTPWrapper(model, n_predict)


def save_mtp(model_dir: Path, wrapper) -> None:
    torch = require_torch()
    model_dir.mkdir(parents=True, exist_ok=True)
    wrapper.inner.save_pretrained(str(model_dir))
    heads = {f"head.{i}": h.state_dict() for i, h in enumerate(wrapper.extra_heads)}
    torch.save(heads, model_dir / "mtp_heads.pt")
    (model_dir / "mtp_meta.json").write_text(
        json.dumps({"n_predict": wrapper.n_predict}, indent=2) + "\n", encoding="utf-8"
    )


def load_mtp(model_dir: Path, *, device=None):
    torch = require_torch()
    from backends.deps import require_transformers

    transformers = require_transformers()
    meta = json.loads((model_dir / "mtp_meta.json").read_text(encoding="utf-8"))
    n_predict = int(meta["n_predict"])
    inner = transformers.AutoModelForCausalLM.from_pretrained(str(model_dir))
    wrapper = wrap_mtp(inner, n_predict)
    heads = torch.load(model_dir / "mtp_heads.pt", map_location="cpu", weights_only=True)
    for i, h in enumerate(wrapper.extra_heads):
        h.load_state_dict(heads[f"head.{i}"])
    if device is not None:
        wrapper.to(device)
    wrapper.eval()
    return wrapper
