# Transformer method

File: `aq/kernel/methods/transformer.py`  
Recipe: `method: transformer`

Tiny, kernel-owned transformer for the coverage walk - **not** a Hugging Face wrap.

## Architectures

Set `arch` (top-level or under `transformer:`):

| `arch` | Data columns | Role |
|--------|--------------|------|
| `decoder` | `text` or `target` | Causal LM-style |
| `encoder` | `text` + `target` | Encode → classify / score |
| `encoder-decoder` | `src` + `tgt` | Seq2seq |

## Common knobs

| Key | Default (approx) | Meaning |
|-----|------------------|---------|
| `d_model` | 8 | Model width |
| `d_ff` | derived | FFN width |
| `steps` | 80 | Training steps |
| `lr` | 0.08 | Learning rate |
| `seq` | 12 | Sequence length |
| `seed` | 0 | RNG |

Calls `aq_metrics.step` during training. Eval uses `evaluate()` (no tabular `predict`). Inspect dumps architecture notes and vocabulary maps (`itos` / `stoi`).

## Tests

`tests/transformers/{decoder,encoder,encoder-decoder}/`.
