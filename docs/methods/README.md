# Methods

Built-in methods are thin wrappers over **real libraries**. Recipe is the API.

Install: `pip install -r aq/kernel/requirements.txt`

| method | Backend | Notes |
|--------|---------|--------|
| linear, logistic, ridge, lasso, elasticnet, tree, forest, gp | scikit-learn | |
| boosting | XGBoost → LightGBM → CatBoost → sklearn GBR | set `library:` to force |
| llm, lora, qlora | Hugging Face + PEFT | **`model:` required** |
| transformer | Hugging Face | **`model:` required**; arch encoder/decoder/enc-dec |

Custom override: `{train}/tools/<name>.py` with a `fit()` function wins over the kernel built-in.

- [Tabular](./tabular.md)
- [Transformers](./transformers.md)
- [LLM & LoRA](./llm.md)
- [Custom methods](./custom.md)

## Honesty

| Claim | Reality |
|-------|---------|
| QLoRA | CUDA + bitsandbytes only |
| GPTQ/AWQ/GGUF/EXL2 | `formats: true` or `formats: [gguf]` — see [recipe](../recipe.md) |
| Speculative / paged KV | Not implemented (errors if set) |
| MTP heads | Not implemented (`objective: mtp` errors) |
| `size:` | Label only; does not download Llama/Phi/etc. |
