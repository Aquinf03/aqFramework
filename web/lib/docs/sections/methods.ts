import type { DocSection } from "../types";

export const METHODS_TABULAR: DocSection = {
  title: "Methods · Tabular",
  intro:
    "Classic / statistical fits via scikit-learn. Boosting prefers XGBoost → LightGBM → CatBoost → sklearn when installed. No custom Python required.",
  prerequisite: "family: tabular in recipe.yaml · data.csv with a target column",
  tools: [
    {
      command: "method: linear | logistic | ridge | lasso | elasticnet",
      description: "Linear models. Set lambda / l1_ratio as needed for ridge, lasso, elasticnet.",
      flags: [],
      example: `family: tabular
method: ridge
lambda: 1.0
data:
  path: data.csv
  target: y
eval:
  metric: mse`,
    },
    {
      command: "method: tree | forest | boosting | gp",
      description:
        "Trees, random forests, gradient boosting, and gaussian processes. Boosting library: auto | xgboost | lightgbm | catboost | sklearn.",
      flags: [],
      example: `family: tabular
method: boosting
library: auto
trees: 100
depth: 4
data:
  path: data.csv
  target: y
eval:
  metric: accuracy`,
    },
  ],
};

export const METHODS_TRANSFORMERS: DocSection = {
  title: "Methods · Transformers",
  intro:
    "Hugging Face transformers with arch encoder | decoder | encoder-decoder. model: is required. Runs on CUDA, MPS, ROCm, or CPU.",
  prerequisite: "method: transformer · recipe.model set to a hub id or path",
  tools: [
    {
      command: "method: transformer",
      description: "Train an encoder, decoder, or encoder-decoder architecture from recipe.model.",
      flags: [],
      example: `method: transformer
model: bert-base-uncased
arch: encoder
steps: 50
data:
  path: data.csv
  text: text
  target: label
eval:
  metric: accuracy`,
    },
  ],
};

export const METHODS_LLM: DocSection = {
  title: "Methods · LLM / LoRA",
  intro:
    "Foundation-model training with Hugging Face + PEFT. model: is required. Devices: CUDA, MPS, ROCm, CPU. QLoRA is CUDA + bitsandbytes only.",
  prerequisite: "family: llm · recipe.model set · kernel HF deps installed",
  tools: [
    {
      command: "method: lora | qlora | llm",
      description:
        "LoRA / QLoRA / full LLM training. Objectives: next-token, sft, full-ft, lora, qlora, fim, mlm, span, continued-pretrain.",
      flags: [],
      example: `family: llm
method: lora
model: meta-llama/Llama-3.2-1B-Instruct
objective: sft
rank: 16
alpha: 32
steps: 100
lr: 2.0e-4
data:
  path: data.jsonl
  prompt: prompt
  completion: completion
eval:
  metric: loss`,
      notes:
        "SFT masks prompt tokens (loss on completion). full-ft trains all non-pad tokens. mlm needs a MaskedLM-capable model. size: is a label only.",
    },
    {
      command: "aq serve",
      description: "Generate from the latest (or named) checkpoint after an LLM train.",
      flags: [
        { name: "--ckpt <name>", description: "Checkpoint name." },
        { name: "--max-tokens n", description: "Max tokens." },
      ],
      example: 'aq serve "hello" --max-tokens 32',
    },
  ],
};

export const METHODS_CUSTOM: DocSection = {
  title: "Methods · Custom",
  intro:
    "Drop methods/<name>.py with fit(src, rec) to override the kernel method of the same name. Convention over registration: the file existing is enough.",
  prerequisite: "A train folder · methods/<name>.py implementing fit(src, rec)",
  tools: [
    {
      command: "methods/<name>.py",
      description:
        "Train-local fit adapter. Wins over kernel/methods/<name>.py. Use only when a built-in is not enough.",
      flags: [],
      example: `# methods/foo.py
def fit(src, rec):
    # read recipe + data, write checkpoint artifacts
    ...`,
    },
  ],
};
