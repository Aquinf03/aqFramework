# Methods

Built-in methods are selected in **`recipe.yaml`**. You usually do not write Python.

| method | What you need | Notes |
|--------|---------------|--------|
| `linear`, `logistic`, `ridge`, `lasso`, `elasticnet`, `tree`, `forest`, `gp` | table + `data.target` | Classic sklearn-style fits |
| `boosting` | table | Tries XGBoost → LightGBM → CatBoost → sklearn; set `library:` to force |
| `llm`, `lora`, `qlora` | text / SFT data + **`model:`** | Hugging Face + PEFT |
| `transformer` | text (+ labels) + **`model:`** | encoder / decoder / enc-dec |
| `cnn` | ImageFolder or path+label | aq-owned CNNs · `family: vision` |
| `vit` | same as CNN | aq ViT / Swin / DeiT / BEiT · `family: vision` |
| `clip` | image–text pairs jsonl | CLIP / SigLIP · `family: vlm` |
| `llava`, `flamingo` | chat jsonl + **`model:`** | Generative VLM · `family: vlm` |

Override any built-in: put `tools/<method>.py` with a `fit()` in **that train**.

- [Tabular](./tabular.md)  
- [Transformers](./transformers.md)  
- [LLM & LoRA](./llm.md)  
- [Vision & VLM](./vision.md)  
- [Custom methods](./custom.md)  

## Honesty (read once)

| Claim | Reality |
|-------|---------|
| QLoRA | CUDA + bitsandbytes only |
| Vision / VLM towers | aq-owned nets (not silent torchvision / timm wrappers) |
| `formats:` | Real export when the stack allows; otherwise fails closed |
| Speculative decode | Needs `draft_model:` at serve |
| `paged_kv` | Recorded; serve still uses a normal cache |
| `size:` | Label only — does not fetch a model |
