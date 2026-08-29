# LLM and LoRA

Recipe-only. **`model` is required** (Hugging Face id or local path).

```yaml
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
```

## Objectives that work

| objective | What it does |
|-----------|----------------|
| `next-token` | Causal LM |
| `sft` | Causal LM; **loss on completion only** |
| `full-ft` | Causal LM; loss on all non-pad tokens |
| `lora` / `qlora` | PEFT LoRA; QLoRA needs CUDA+bnb |
| `fim` | PSM FIM formatting + causal LM |
| `mlm` | Masked LM (BERT-class `model`) |
| `span` | Span-corrupt input + reconstruct labels |
| `continued-pretrain` | Loads `init.checkpoint` weights, then trains |

## Devices

Detected automatically: `cuda`, `rocm` (AMD HIP), `mps` (Apple), `cpu`. Written into the checkpoint manifest as `device`.

Implementation: `aq/kernel/backends/hf_lm.py`.
