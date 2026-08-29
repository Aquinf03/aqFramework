# Coverage and roadmap

Internal sources of truth for builders: `internals/COMPLETED.md`, `internals/TODO.md`, `internals/ML_MODEL_TAXONOMY.md`, `internals/ML_DEVELOPMENT_METHODS.md`, `internals/PLAN.md`.

This page is the **user-facing** map of what the framework already walks vs what remains encyclopedia material.

## Done (harness)

Directory protocol, CLI verbs, jobs, kernel ops (hash/train/eval/checkpoint/serve), metrics stream + live steps, opt-in guard, tools/skills/schedules/stages, research tracking (`runs`, `diff`, `status`), agent chat/ask/spawn/doctor, auth login tokens, install/release.

## Done (coverage walk)

- Classic tabular: linear → GP (see [tests catalog](./tests-catalog.md))  
- Transformers: encoder / decoder / encoder-decoder  
- Foundation models: tokenizers, pack/mixture, AR / MLM+span / MTP / FIM, continued pretrain, SFT, full-ft, LoRA, QLoRA, serve  

## Remaining (TODO backlog themes)

Not stubs - only tick when a real train exists:

- Classic leftovers (SVM, kNN, …)  
- Architectures beyond the tiny transformer  
- Preference / RLHF / large-scale parallelism (when kernel-owned, not wraps)  
- Vision, multimodal, scientific ML, RL, unsupervised, embeddings, causal, robotics, 3D, security, federated, AutoML, …  
- Training infrastructure encyclopedia items (optimizers, schedules, compression, HPO, …) as **first-class folder workflows**

Taxonomy and methods encyclopedias under `internals/` list the landscape; they are not a promise that every row ships tomorrow.

## Build order reminder

Directory → CLI → Jobs → Kernel → Grow handle → Research tracking → Agent → Coverage → Own stack / [PyAquin](./pyaquin.md) only when reason-gated.

## What “coverage” means here

Same path every family:

```
init/fork → recipe + data → aq train → aq eval (your gate) → inspect → job/checkout → fork again
```

Wrapping an external library solely to claim a checkbox is explicitly **not** coverage.
