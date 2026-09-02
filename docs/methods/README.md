# Methods

Built-in methods are thin wrappers over **real libraries**. Recipe is the API.

Install: `pip install -r aq/kernel/requirements.txt`

| method | Backend | Notes |
|--------|---------|--------|
| linear, logistic, ridge, lasso, elasticnet, tree, forest, gp | scikit-learn | |
| boosting | XGBoost → LightGBM → CatBoost → sklearn GBR | set `library:` to force |
| llm, lora, qlora | Hugging Face + PEFT | **`model:` required** |
| transformer | Hugging Face | **`model:` required**; arch encoder/decoder/enc-dec |
| cnn | **aq `neural/cnn`** | `family: vision`; arch lenet/alexnet/vgg/resnet/inception/efficientnet/convnext |
| vit | **aq `neural/vit`** | `family: vision`; ViT / Swin / DeiT (CNN teacher) / BEiT (VQ + block MIM) |
| clip | **aq `neural/vlm`** | `family: vlm`; CLIP / SigLIP contrastive; pairs jsonl |
| llava / flamingo | **aq connector + HF LM** | `family: vlm`; LLaVA projector or Flamingo gated xattn; chat jsonl |

Custom override: `{train}/tools/<name>.py` with a `fit()` function wins over the kernel built-in.

- [Tabular](./tabular.md)
- [Transformers](./transformers.md)
- [LLM & LoRA](./llm.md)
- [Custom methods](./custom.md)

## Honesty

| Claim | Reality |
|-------|---------|
| QLoRA | CUDA + bitsandbytes only |
| Vision CNN | aq-owned modules (not torchvision.models); needs Pillow |
| Vision ViT | aq-owned ViT/Swin/DeiT/BEiT; DeiT uses a warmed CNN teacher; BEiT trains a discrete VAE then blockwise MIM |
| VLM CLIP/SigLIP | aq dual encoders; eval recall@1 |
| VLM LLaVA/Flamingo | aq vision+connector on HF/local causal LM; GPT-4V-style = LLaVA-class |
| GPTQ/AWQ/GGUF/EXL2 | `formats:` — see [recipe](../recipe.md) |
| Speculative | `draft_model:` + HF assisted decode |
| Paged KV | Recorded; serve still standard HF cache |
| MTP heads | `objective: mtp` + `n_predict` |
| `size:` | Label only; does not download Llama/Phi/etc. |
