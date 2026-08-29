# TLDR for beta testers

A **train** is a folder. **`aq`** is the CLI. **`recipe.yaml` is the full train spec** for supported methods. You do not write Python for those.

## Install

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
# or from this repo:
cd aq && npm install && npm link
pip install -r aq/kernel/requirements.txt
```

Needs Node ≥ 18, npm, python3, and the kernel requirements (torch, transformers, peft, scikit-learn, …). Then: `aq help` · `aq doctor`

## 10-minute path (tabular)

```bash
aq init my-train && cd my-train
# recipe: method: linear, data.path, data.target
aq train
aq eval
aq status
cat artifacts/inspect.md
```

## LLM / LoRA (recipe only)

```yaml
family: llm
method: lora
model: meta-llama/Llama-3.2-1B-Instruct   # required. hub id or path. no toy fallback.
rank: 16
alpha: 32
steps: 100
lr: 2.0e-4
# bits: 4   # QLoRA on CUDA only
data:
  path: data.jsonl
  prompt: prompt
  completion: completion
eval:
  metric: loss
```

```bash
aq train
aq serve "Hello"
```

No `methods/lora.py` needed. If `model` is missing, train fails on purpose.

## Mental model

| Piece | Where |
|-------|--------|
| Intent | `instructions.md` |
| Spec | `recipe.yaml` (this is what trains) |
| Data / probes | `data/`, `evals/` |
| Outputs | `artifacts/` |
| Agent | `aq` on a TTY (same folder) |

## Optional

```bash
aq provider openai
aq login
aq job run -- aq train
# guard: { safety: true, leak: true }
```

## What we want from you

1. Does init, train, eval, fork feel obvious with recipe only?
2. Where did you get stuck?
3. Did live steps / status / inspect make the run auditable?
4. Real HF train on your GPU/Mac: what broke?

**Deeper docs:** [`docs/README.md`](./README.md)
