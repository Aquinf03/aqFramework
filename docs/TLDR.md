# TLDR for beta testers

A **train** is a folder. **`aq`** is the CLI. For built-in methods, **`recipe.yaml` is the full train spec** (no custom Python).

## Install

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
# or from this repo:
cd aq && npm install && npm link
python3 -m venv aq/kernel/.venv
aq/kernel/.venv/bin/pip install -r aq/kernel/requirements.txt
```

Needs Node ≥ 18, npm, python3. Then: `aq help` · `aq doctor`

## Tabular

```bash
aq init my-train && cd my-train
# bare `aq init` → aq-experiment (or aq-experiment-new1, …)
# recipe: method: linear, data.path, data.target
aq train && aq eval
```

## LLM / LoRA (recipe only)

```yaml
family: llm
method: lora
model: meta-llama/Llama-3.2-1B-Instruct   # required hub id or path
rank: 16
alpha: 32
steps: 100
lr: 2.0e-4
data:
  path: data.jsonl
  prompt: prompt
  completion: completion
eval:
  metric: loss
```

- **QLoRA** (`bits: 4`): NVIDIA CUDA + `bitsandbytes` only. On Mac/AMD use LoRA without `bits`.
- **`size:`** is a label only. It does not pick weights.
- Devices: CUDA, MPS, ROCm (AMD), CPU.

## What is real vs not

See [COMPLETED.md](../internals/COMPLETED.md) for what works. `paged_kv` stays inactive until real paged attention lands — tracked in [TODO.md](../internals/TODO.md).

## What we want from you

Bugs + a repro train folder beat vibes. Did `init → train → eval → fork` feel obvious?

**Docs index:** [README.md](./README.md)
