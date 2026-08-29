# LLM and LoRA

Recipe-only. **`model` is required** (Hugging Face id or local path). No toy weights.

```yaml
method: lora          # or llm | qlora
model: meta-llama/Llama-3.2-1B-Instruct
objective: lora       # sft | full-ft | qlora | next-token | …
bits: 4               # QLoRA; CUDA + bitsandbytes
rank: 16
alpha: 32
steps: 200
lr: 2.0e-4
data:
  path: data.jsonl
  prompt: prompt
  completion: completion
```

Implementation: `aq/kernel/backends/hf_lm.py` via `methods/lora.py`, `qlora.py`, `llm.py`.
