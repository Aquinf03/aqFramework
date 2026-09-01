# kernel

Python. Train, eval, serve, checkpoint, data hash. Spec is **`recipe.yaml`**. Built-ins need no custom Python.

```
run.py
backends/       real stacks (HF, sklearn, device, deploy, tok_train)
protocol/       recipe, revision, record, metrics, guard, method loader
engine/         train / eval / serve / checkpoint
methods/        thin adapters; filename = recipe.method
requirements.txt
```

```bash
pip install -r aq/kernel/requirements.txt
# aq prefers aq/kernel/.venv/bin/python when present
```

## Contract

- `recipe.model` **required** for llm / lora / qlora / transformer.
- `size:` is a label only.
- QLoRA = CUDA + bitsandbytes only (errors elsewhere).
- Unsupported: `formats` (errors, not faked). `objective: mtp` needs `n_predict >= 2`. `paged_kv` / `speculative` are recorded at train; serve is not paged/speculative yet.
- Devices: CUDA, MPS, ROCm, CPU.

See [docs/recipe.md](../../docs/recipe.md) and [internals/COMPLETED.md](../../internals/COMPLETED.md).
