# Recipe (`recipe.yaml`)

The recipe is the **full train API** for built-in methods. Change YAML, not the CLI. Custom fit code belongs in **`tools/<name>.py`** — see [Custom methods](./methods/custom.md).

Kernel packages: installed into the framework venv from `aq/kernel/requirements.txt` (you normally get this from the installer).

---

## Tabular

```yaml
family: tabular
method: linear   # logistic | ridge | lasso | elasticnet | tree | forest | boosting | gp
data:
  path: data.csv
  target: y
eval:
  metric: mse    # accuracy | mae | rmse | r2
# optional: lambda / l1_ratio / trees / depth
# boosting: library: auto | xgboost | lightgbm | catboost | sklearn
```

Uses scikit-learn, plus XGBoost / LightGBM / CatBoost when you pick boosting.

---

## LLM / LoRA / QLoRA

```yaml
family: llm
method: lora          # or llm | qlora
model: meta-llama/Llama-3.2-1B-Instruct   # REQUIRED
objective: lora       # next-token | sft | full-ft | lora | qlora | fim | mlm | span | continued-pretrain | mtp
# bits: 4             # QLoRA — CUDA + bitsandbytes only
rank: 16
alpha: 32
steps: 100
lr: 2.0e-4
# tokenizer: bpe      # optional local tokenizer: bpe | unigram | wordpiece | byte
# max_seq_len: 512
# prune: 0.1          # magnitude prune after train
# quant: int8         # aq weight dump (not GPTQ)
data:
  path: data.jsonl
  text: text
  # sft / full-ft:
  # prompt: prompt
  # completion: completion
eval:
  metric: loss
```

**Rules that save you time**

| Knob | Reality |
|------|---------|
| `model:` | **Required.** Hub id or local path. |
| `size:` | Label only (`llm` / `slm` / `edge`). Does not download weights. |
| SFT | Loss on completion tokens only. |
| full-ft | Loss on all non-pad tokens. |
| QLoRA | CUDA + bitsandbytes only. On Mac/AMD use LoRA. |
| `objective: mtp` | Multi-token heads; set `n_predict: 2` (or higher). |
| `speculative: true` | Serve-time assisted decode; needs `draft_model:`. |
| `formats: true` / `[gguf]` | Export packs after train (HF + GGUF; GPTQ/AWQ on CUDA). Prefer `[gguf]` on Mac. |
| `paged_kv: true` | Recorded today; serve still uses a normal HF cache. |

Unsupported export paths **fail closed**.

---

## Vision (CNN)

Aq-owned backbones (not torchvision wrappers). ImageFolder or a path+label table.

```yaml
family: vision
method: cnn
arch: resnet18    # lenet | alexnet | vgg16 | resnet18/34/50/101/152 |
                  # inception | efficientnet_b0/b1/b2 | convnext_tiny/small/base
epochs: 10
batch_size: 32
lr: 1.0e-3
image_size: 224   # lenet defaults smaller
val_frac: 0.1
data:
  path: data/images          # data/images/<class>/*.jpg
  # or table:
  # path: data/index.csv
  # image: path
  # target: label
eval:
  metric: accuracy           # or loss
```

Needs **Pillow**. Weights land under `artifacts/checkpoints/<n>/model.pt`.

---

## Vision (ViT / Swin / DeiT / BEiT)

Same data layout as CNN.

```yaml
family: vision
method: vit
arch: vit-b/16        # vit-t/16 | vit-s/16 | vit-b/16 | vit-b/32
                      # vit-l/14 | vit-l/16 | vit-h/14 | vit-g/14
                      # (OpenCLIP names like ViT-H-14 also resolve)
                      # swin-t | swin-s | swin-b   (image_size usually 224, % 32 == 0)
                      # deit-t | deit-s | deit-b
                      # beit-b | beit-l
epochs: 10
batch_size: 32
lr: 1.0e-3
weight_decay: 0.05
image_size: 224
# DeiT: teacher: resnet50 · teacher_epochs: 3 · distill_alpha / distill_temp
# BEiT: vocab_size · vq_epochs · mim_epochs · classify_epochs · mask_ratio
data:
  path: data/images
eval:
  metric: accuracy
```

---

## Vision–language (CLIP / SigLIP / LLaVA / Flamingo)

`family: vlm`.

### CLIP / SigLIP (contrastive)

```yaml
family: vlm
method: clip
arch: clip              # or siglip
vision: ViT-H-14        # any size: vit-b/16, vit-h/14, ViT-L-14, …
# vision_pretrained: laion2b_s32b_b79k   # OpenCLIP tag (needs open-clip-torch)
image_size: 224
embed_dim: 1024         # match tower / OpenCLIP embed when using pretrained
epochs: 10
batch_size: 64
lr: 5.0e-4
data:
  path: data/pairs.jsonl   # {image, text} — not ImageFolder
  image: image
  text: text
eval:
  metric: recall@1         # or loss
```

`vision:` accepts aq size strings (`vit-h/14`) **and** OpenCLIP ids (`ViT-H-14`). Set `vision_pretrained` (alias `pretrained`) to an OpenCLIP tag to load weights — `pip install open-clip-torch`. Without that tag, aq builds the matching ViT from scratch.

Optional: `text_tokenizer: openai/clip-vit-base-patch32` for a pretrained CLIP tokenizer; otherwise aq builds one from your captions.

### LLaVA / GPT-4V-style / Flamingo (generative)

```yaml
family: vlm
method: llava              # or flamingo
arch: llava                # llava | gpt4v-style | flamingo
vision: ViT-H-14
# vision_pretrained: laion2b_s32b_b79k
model: meta-llama/…        # HF id or local causal LM directory
image_size: 224
freeze_vision: true
freeze_lm: false           # flamingo defaults freeze_lm: true
epochs: 1
data:
  path: data/chat.jsonl    # {image, prompt, completion} or conversations[]
eval:
  metric: loss
```

`gpt4v-style` is a LLaVA-class alias. Flamingo adds a perceiver + gated cross-attention into the LM.

---

## Transformer (text)

```yaml
method: transformer
model: bert-base-uncased      # required
arch: encoder                 # decoder | encoder | encoder-decoder
steps: 50
data:
  path: data.csv
  text: text
  target: label               # encoder
  # src / tgt for encoder-decoder
```

---

## Guard (opt-in)

```yaml
guard:
  safety: true    # abort on exploding loss / NaNs
  leak: true      # abort if train data overlaps eval probes
```

See [Metrics & guard](./metrics-and-guard.md).
