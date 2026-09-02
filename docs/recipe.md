# Recipe (`recipe.yaml`)

The recipe is the **full train spec** for built-in methods. Custom fit code goes in **`tools/<name>.py`** (same folder as `aq tool` scripts).

Install backends: `pip install -r aq/kernel/requirements.txt`

## Tabular

```yaml
family: tabular
method: linear   # logistic | ridge | lasso | elasticnet | tree | forest | boosting | gp
data:
  path: data.csv
  target: y
eval:
  metric: mse    # accuracy | mae | rmse | r2
# lambda / l1_ratio / trees / depth / library (boosting: auto|xgboost|lightgbm|catboost|sklearn)
```

Backed by **scikit-learn**, **XGBoost**, **LightGBM**, and **CatBoost** (all installed with the kernel by default).

## LLM / LoRA / QLoRA

```yaml
family: llm
method: lora          # or llm | qlora
model: meta-llama/Llama-3.2-1B-Instruct   # REQUIRED
objective: lora       # next-token | sft | full-ft | lora | qlora | fim | mlm | span | continued-pretrain
# bits: 4             # QLoRA — CUDA + bitsandbytes only
rank: 16
alpha: 32
steps: 100
lr: 2.0e-4
tokenizer: bpe        # optional: train a local tokenizer (bpe|unigram|wordpiece|byte)
merges: 1000
max_seq_len: 512
# mixture: { web: 0.5, code: 0.5 }   # with data.source
# prune: 0.1                         # magnitude prune after train
# quant: int8                        # aq weight dump (not GPTQ)
data:
  path: data.jsonl
  text: text
  # or for sft/full-ft:
  # prompt: prompt
  # completion: completion
eval:
  metric: loss
```

Rules:

- **`model:` is required.** `size:` is only a label (`llm`/`slm`/`edge`).
- **SFT** masks prompt tokens (loss on completion only). **full-ft** trains all non-pad tokens.
- **mlm** needs a MaskedLM-capable model for true bidirectional MLM (e.g. BERT). Causal models (Llama, GPT) also work — aq uses masked-token loss on the causal backbone.
- **QLoRA** fails on MPS/CPU/ROCm without CUDA bitsandbytes.
- **`objective: mtp`** — multi-token prediction; set **`n_predict: 2`** (or higher). Auxiliary heads on a causal LM.
- **`speculative: true`** — serve-time assisted decode. Requires **`draft_model: <hub-id>`** (smaller causal LM, same tokenizer family). Train records the draft; `aq serve` / generate uses Hugging Face `assistant_model`.
- **`formats: true`** / **`formats: [gguf]`** — post-train weight packs under the checkpoint `formats/` dir. Always writes merged **HF** weights; **GGUF** via built-in Llama-family writer (`pip install gguf`) or `AQUIN_LLAMA_CPP`; **GPTQ/AWQ** need CUDA + optional packages; **EXL2** needs `AQUIN_EXL2_CONVERT`.
- **`paged_kv: true`** — recorded at train; `aq serve` still uses standard HF KV cache (not vLLM paged attention yet).
- These **fail closed** when a requested format cannot actually be produced.

## Vision (CNN)

Aq-owned backbones (not torchvision wrappers). Folder-of-folders or a path+label table.

```yaml
family: vision
method: cnn
arch: resnet18    # lenet | alexnet | vgg16 | resnet18/34/50/101/152 |
                  # inception | efficientnet_b0/b1/b2 | convnext_tiny/small/base
epochs: 10
batch_size: 32
lr: 1.0e-3
image_size: 224   # default depends on arch (lenet→32)
val_frac: 0.1
data:
  path: data/images          # ImageFolder: data/images/<class>/*.jpg
  # or a csv/jsonl:
  # path: data/index.csv
  # image: path
  # target: label
eval:
  metric: accuracy           # or loss
```

Requires **Pillow**. Checkpoints store `model.pt` under `artifacts/checkpoints/<n>/`.

## Transformer

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

## Guard (opt-in)

```yaml
guard:
  safety: true
  leak: true
```

## Custom methods

Only if you need something not built in: `tools/<name>.py` with `fit(src, rec)`.
