# Size / deploy

Recipe knobs on `method: llm`. Not new architectures. Kernel-owned.

| Train | Knob | Inspect |
| --- | --- | --- |
| [llms](../llms/) | `size: llm` | `class: llm` |
| [slms](../slms/) | `size: slm` | `class: slm` |
| [edge](../edge/) | `size: edge` | `class: edge` |
| [distilled](../distilled/) | `distill: true` | `distill: true` |
| [quantized](../quantized/) | `quant: int8` | `quant: int8` |
| [weight-formats](../weight-formats/) | `formats: true` | `model.gguf` + q8/gptq/awq/exl2 |
| [pruned](../pruned/) | `prune: 0.3` | `sparsity` |
| [speculative](../speculative/) | `speculative: true` | `draft_layers` |
| [paged-kv](../paged-kv/) | `paged_kv: true` | `page_size` |

```
cd tests/<name>
aq train
aq eval
cat artifacts/inspect.md
```
