# PyAquin (later)

Full note: `internals/PYAQUIN.md`.

**Not next.** First Aquin-the-OS must be boring: folder, recipe, job, eval that can fail, one real train you can fork.

## Layers

| Layer | Analogy | Role |
|-------|---------|------|
| **Aquin** | Linux | Directory, `aq`, jobs, `evals/`, fork, checkout |
| **PyAquin** | JavaScript | What you write in the train (`train.aq`) - small, eager, one way |
| **PyTorch / CUDA / HF / tokenizers** | glibc | Hidden floor until a family proves it is the wall |

Do not compete with `nn.Linear`. Do not skip the framework to write a compiler.

## Why a runtime might exist someday

PyTorch already won tensors, autograd, CUDA, `torch.compile`. Wins worth a runtime for Aquin’s product (science, clinic, control):

1. **Repro is default** - same train + data hash + aq version = same numbers, or error  
2. **Tensors have lineage** - shape/dtype plus recipe, data hash, step  
3. **Shards and devices are syscalls** - you never write FSDP yourself; `aq train` places the job  
4. **Tokens and packs are types** - tokenizer lives in the train and is hashed like weights  
5. **NaN / overflow fail the job** in safety-critical mode (already started via `guard.safety`)  
6. **Science types** - fields, meshes, units, time - not only NCHW  

If those six are not worth a runtime, wrap torch forever and only build Aquin.

## Sketch (destination, not queue)

```
clinic-sft/
  instructions.md
  recipe.yaml
  train.aq
  data/notes.jsonl
  evals/leak.py
```

```python
# train.aq - you do not import torch
from aq import data, model, train, save

tok = data.tokenizer("clinic-bpe")
ds  = data.pack("data/notes.jsonl", tok, ctx=2048)
m = model.decoder(dim=512, layers=8, vocab=tok.vocab)
m = model.lora(m, rank=16)
train(m, ds, steps=2000, batch=8, lr=2e-4)
save("artifacts/checkpoints/last")
```

**Near term:** compile to generated PyTorch under `artifacts/gen/`.  
**Later:** PyAquin kernels; torch may remain FFI. Same user file.

## When to start

Not until:

- one foundation-model train folder is boring and correct  
- a family hits a wall that torch/HF/tokenizers cannot answer for this product  

First touchable PyAquin is not a GPU: a CPU tensor with `(value, recipe_id, data_hash)` that refuses to save if eval is missing. Then steal kernels.

Until then, the harness and coverage walk are the queue. This file is the destination.
