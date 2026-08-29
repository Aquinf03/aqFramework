# Recipe (`recipe.yaml`)

The recipe is the **full train spec**. For built-in methods, you do not write Python. `aq train` loads the method named in `recipe.method` and that module reads the recipe.

Install backends: `pip install -r aq/kernel/requirements.txt`

## Tabular

```yaml
family: tabular
method: linear   # logistic | ridge | lasso | elasticnet | tree | forest | boosting | gp
data:
  path: data.csv
  target: y
eval:
  metric: mse
  min_score: null
# lambda / l1_ratio / trees / depth / library (boosting: auto|xgboost|lightgbm|sklearn)
```

Backed by **scikit-learn** (XGBoost/LightGBM when installed for boosting).

## LLM / LoRA / QLoRA

```yaml
family: llm
method: lora          # or llm | qlora
model: meta-llama/Llama-3.2-1B-Instruct   # REQUIRED (hub id or local path)
objective: lora       # next-token | sft | full-ft | lora | qlora | continued-pretrain
bits: 4               # optional QLoRA (CUDA + bitsandbytes)
rank: 16
alpha: 32
steps: 100
lr: 2.0e-4
batch_size: 1
grad_accum: 8
max_seq_len: 512
data:
  path: data.jsonl
  text: text                    # or:
  # prompt: prompt
  # completion: completion
eval:
  metric: loss
```

There is **no toy fallback**. Missing `model` is an error.

Nested forms also work: `train.lr`, `lora.rank`, `quantization.load_in_4bit`, etc.

## Transformer

```yaml
method: transformer
model: bert-base-uncased      # required
arch: encoder                 # decoder | encoder | encoder-decoder
steps: 50
lr: 5.0e-5
data:
  path: data.csv
  text: text
  target: label               # encoder classify
  # src / tgt for encoder-decoder
```

## Guard (opt-in)

```yaml
guard:
  safety: true
  leak: true
```

## Custom methods

Only if you need something not built in: put `methods/<name>.py` with `fit(src, rec)` in the train. That overrides the kernel file of the same name.
