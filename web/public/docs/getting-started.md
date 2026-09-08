# Getting started

A **train** is a folder. **`aq`** is the CLI. For built-in methods, **`recipe.yaml` is the full train spec** — you usually do not write Python.

## Install

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

Needs **Node ≥ 18**, **npm**, and **Python >= 3.10**. Then:

```bash
aq help
aq doctor
```

From a git checkout instead: see [Install](./install.md).

## First train (tabular)

```bash
aq init my-train && cd my-train
```

`aq init` always creates a **new folder** (`aq-experiment`, or `aq-experiment-new1` if that name is taken). It never dumps files into the current directory.

Edit `recipe.yaml` to something like:

```yaml
family: tabular
method: linear
data:
  path: data.csv
  target: y
eval:
  metric: mse
  min_score: null
```

Put your table at `data.csv` (or under `data/` and point `path` there). Then:

```bash
aq train
aq eval
aq status
```

Fork a variant without copying junk:

```bash
aq fork ../my-train-ridge
cd ../my-train-ridge
# change method: ridge (and lambda if you want)
aq train && aq eval
aq diff <run-a> <run-b>
```

## First LoRA (recipe only)

```yaml
family: llm
method: lora
model: meta-llama/Llama-3.2-1B-Instruct   # required hub id or local path
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

- **`model:` is required.** `size:` is only a label — it does not download weights.  
- **QLoRA** (`bits: 4`): NVIDIA CUDA + `bitsandbytes` only. On Mac / AMD use LoRA without `bits`.  
- Devices: CUDA, MPS (Apple), ROCm (AMD), CPU.

```bash
aq train
aq serve "hello" --max-tokens 32
```

## What “done” feels like

You can explain the train from `experiment.md` + `recipe.yaml`, reproduce with `aq train` / `aq eval`, and decide the gate yourself when `eval.min_score` is set.

## Next

- [Concepts](./concepts.md)  
- [Train folder](./train-folder.md)  
- [Recipe](./recipe.md)  
- [CLI](./cli.md)  

Bugs with a repro train folder beat vibes. If something fails closed, that is intentional — unsupported knobs should not silently pretend.
