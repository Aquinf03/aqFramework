# Philosophy

Aquin is a **training operating system** whose unit of work is a folder. You do not “submit jobs to a platform” as the primary act. You grow a directory, run `aq` verbs against it, and keep enough on disk that a stranger (or your future self) can fork, resume, and audit what happened.

## One-liners

- **A train is a directory.**
- **`aq` is the CLI** (TypeScript face).
- **The kernel is Python** (fit, eval, serve, hash).
- **The UI is a view.** The Unix is the product.
- **No preset zoo.** You write the data, the gate, and when needed the method.

## Naming

| Name | Meaning |
|------|---------|
| **Aquin** | The product / whole system |
| **aq** | The CLI you type, the way you type `git` |
| **Kernel** | Python under `aq/kernel/`: train, eval, inspect, numerics |
| **Train / handle** | A directory that is a real unit of work |
| **aq agent** | An fx-shaped agent that lives *inside* this Unix, not beside it |

## First principles

### 1. The unit of work is a directory

If it cannot be copied, forked, and resumed from disk, it is not real. Checkpoints, run records, metrics streams, eval results, chats, and job trees all live under the train. Cloud blobs and web sessions are optional mirrors - never the source of truth.

### 2. Do one thing. Compose the rest.

`aq train` fits. `aq eval` scores. `aq job run` runs a command with resource asks. Schedules compose those verbs. The agent composes them with tools. Nothing tries to be an all-in-one “experiment platform UI” that hides the folder.

### 3. The computer is the product

Laptop, rack, and cluster should share the same job model. Resources (`--cpu`, `--ram`, `--gpu`) are asks against a host, not a different product SKU.

### 4. CLI is Unix, not an IDE

Quiet. Current working directory is the workspace. Help is short. The agent (when you open it) is a resident of that Unix - same cwd, same files, same verbs - not a chat toy with a private filesystem.

### 5. Convention over registration

Drop a file in the right folder and it exists:

- `methods/foo.py` → method `foo`
- `tools/bar.py` → `aq tool bar`
- `evals/holdout.csv` → probe for `aq eval`
- `skills/…` → agent skills / MCP
- `schedules/nightly.yaml` → `aq schedule`

There is no central registry to update.

### 6. Python is the kernel. TypeScript is the face.

They talk through **files and JSON**, not a shared memory ABI:

1. CLI writes `artifacts/request.json`
2. Spawns `kernel/run.py <train>`
3. Kernel writes `artifacts/result.json` and appends `metrics.jsonl`
4. CLI prints result lines; live progress streams on stderr

### 7. Coverage is the kernel’s job

The same path (train, then eval, then inspect) must work across families (tabular, transformers, foundation models). Wrapping PyTorch or Hugging Face just to tick a box is not coverage. Kernel-owned `fit` (stdlib / our ops) is the bar until a family proves torch is the wall.

### 8. Interpretability is a capability, not a ritual

`artifacts/inspect.md` exists so humans can *see* what the model did (weights, trees, tokenizer notes). It is not a separate product. It is part of how you check a fit.

### 9. No preset zoo

Aquin does not ship “ImageNet but smaller” or a blessed benchmark suite as product. **You** put probes in `evals/`. Humans own the gate (`aq eval`). The agent must not invent scores.

### 10. Memory and skills

Drop notes in `memory/` and `skills/` when a train teaches you something. The agent can read and write them; they fork with the train. No separate “harness” CLI — just files.

## What “working” means

A stranger should be able to:

1. `aq init` (or clone a train)
2. Point `recipe.yaml` at *their* data
3. Set *their* `eval.min_score` (or leave it null and read numbers)
4. `aq train` → `aq eval`
5. Fail, inspect, edit one file, fork, run again
6. File long work as `aq job run`

If that path is awkward, the framework is unfinished - not “needs a dashboard.”

## What we will not do

- Invent a second protocol beside the train folder
- Ship dual CLIs with different semantics
- Make interpretability the product instead of a capability
- Build a generic web-builder agent that ignores the train
- Require a central tool registry
- Treat laptop and cluster as different products
- Tick coverage boxes by wrapping external ML libraries
- Ship an official eval zoo that pretends to be science

## Where this sits relative to PyTorch

PyTorch already won tensors, autograd, and CUDA. Aquin does not compete there. Aquin wins on **reproducible folders**, **gates that error instead of faking**, **hashed data and tokenizers**, **jobs you can checkout**, and **an agent that refuses to invent metrics**. Leaving the Hugging Face adapter for an aq-owned neural path is next (`internals/REPLACE_HF.md`).
