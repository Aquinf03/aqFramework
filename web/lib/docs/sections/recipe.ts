import type { DocSection } from "../types";

export const RECIPE: DocSection = {
  title: "Recipe",
  intro:
    "recipe.yaml is the full train API for built-in methods. You do not write Python unless you drop methods/<name>.py. The kernel reads the recipe; aq does not override it.",
  prerequisite: "A train folder with recipe.yaml · kernel deps: pip install -r aq/kernel/requirements.txt",
  tools: [
    {
      command: "recipe.yaml · tabular",
      description:
        "family: tabular with method linear | logistic | ridge | lasso | elasticnet | tree | forest | boosting | gp. Backed by scikit-learn (boosting prefers XGBoost → LightGBM → CatBoost → sklearn).",
      flags: [],
      example: `family: tabular
method: linear
data:
  path: data.csv
  target: y
eval:
  metric: mse
  min_score: null`,
    },
    {
      command: "recipe.yaml · llm / lora / qlora",
      description:
        "family: llm with method llm | lora | qlora. model: is required (hub id or path). size: is a label only. QLoRA needs CUDA + bitsandbytes.",
      flags: [],
      example: `family: llm
method: lora
model: meta-llama/Llama-3.2-1B-Instruct
objective: lora
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
        "Objectives that work: next-token, sft, full-ft, lora/qlora, fim, mlm, span, continued-pretrain. Fail closed (not faked): formats, speculative, paged_kv, objective: mtp.",
    },
    {
      command: "recipe.yaml · transformer",
      description: "method: transformer with arch encoder | decoder | encoder-decoder. model: required.",
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
    {
      command: "recipe.yaml · guard",
      description: "Opt-in fail-closed watches mid-train. See Metrics & guard.",
      flags: [],
      example: `guard:
  safety: true   # NaN / blow-up
  leak: true     # train vs evals overlap`,
    },
  ],
};
