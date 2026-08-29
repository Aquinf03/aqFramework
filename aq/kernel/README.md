# kernel

Python. Train, eval, serve, checkpoint, data hash. `aq` calls `run.py` with a train dir. **Spec is `recipe.yaml` only** for built-in methods. You do not write Python unless you add a custom `methods/<name>.py`.

```
run.py          worker. artifacts/request.json → result.json
backends/       real stacks (Hugging Face, scikit-learn, xgboost, …)
protocol/       recipe, revision, record, metrics, guard, method loader
engine/         train / eval / serve / checkpoint
methods/        thin adapters; filename = recipe.method
requirements.txt
```

Install ML deps (same Python as `aq train`):

```bash
pip install -r aq/kernel/requirements.txt
```

## Contract

- `recipe.model` is **required** for `llm` / `lora` / `qlora` / `transformer`. Hub id or local path. No toy fallback.
- Tabular methods use scikit-learn (boosting prefers xgboost when installed).
- Train-local `methods/<name>.py` still overrides the kernel if you need a custom fit.

## Example (QLoRA, recipe only)

```yaml
family: llm
method: lora
model: meta-llama/Llama-3.2-1B-Instruct
bits: 4
rank: 16
alpha: 32
steps: 200
lr: 2.0e-4
data:
  path: data.jsonl
  prompt: prompt
  completion: completion
eval:
  metric: loss
```

`bits: 4` needs CUDA + `bitsandbytes`. On Mac/CPU omit `bits` for LoRA.
