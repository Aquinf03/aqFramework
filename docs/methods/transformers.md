# Transformers

File: `aq/kernel/methods/transformer.py`  
**`recipe.model` is required** (hub id or path).

| arch | Data | Model example |
|------|------|----------------|
| `decoder` | `text` | causal LM |
| `encoder` | `text` + `target` | sequence classification |
| `encoder-decoder` | `src` + `tgt` | seq2seq |

Runs on CUDA / MPS / ROCm / CPU via the same device layer as `hf_lm`.
