# LLM & LoRA

Recipe-only language-model trains. **`model:` is required** (Hugging Face id or local path).

```yaml
family: llm
method: lora
model: meta-llama/Llama-3.2-1B-Instruct
objective: lora
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

## Objectives

| objective | What it does |
|-----------|----------------|
| `next-token` | Causal LM |
| `sft` | Loss on completion tokens only |
| `full-ft` | Loss on all non-pad tokens |
| `lora` / `qlora` | PEFT adapters (`bits: 4` → QLoRA, CUDA only) |
| `fim` | Fill-in-the-middle |
| `mlm` / `span` | Masked / span corruption |
| `continued-pretrain` | Load prior aq weights via `init.checkpoint`, train more |
| `mtp` | Multi-token prediction heads (`n_predict`) |

## Devices

CUDA, MPS (Apple), ROCm (AMD), CPU. **QLoRA needs CUDA + bitsandbytes.**

```bash
aq train
aq serve "hello" --max-tokens 64
```

More knobs (`formats`, `speculative`, tokenizer): [Recipe — LLM](../recipe.md).
