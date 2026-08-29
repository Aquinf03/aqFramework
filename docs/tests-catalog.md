# Tests catalog

`/tests` holds **manual end-to-end trains**, not product `evals/` probes and not a unit-test runner. Each folder is a train (or a small tree of trains) plus a README. Pattern:

1. Build CLI (`cd aq && npm run build`) if needed  
2. `aq train` / `aq eval`  
3. Read `artifacts/inspect.md`  
4. Often also exercise the agent path  

> Tick a box in COMPLETED/TODO when it actually works, not when a stub exists.

Index: `tests/README.md`.

---

## Classic tabular

| Folder | Proves |
|--------|--------|
| `linear-regression/` | First family; simple line; MSE gate |
| `logistic-regression/` | Binary class; accuracy; weight sign |
| `ridge/` | L2; collinear features; no exact zeros |
| `lasso/` | L1; junk feature → 0 |
| `elastic-net/` | L1+L2 mix |
| `decision-trees/` | CART split; flowchart inspect |
| `random-forests/` | Bagging; feature-use |
| `gradient-boosting/` | Residual trees (`method: boosting`) |
| `gaussian-processes/` | RBF GP on `sin(x)`; posterior band |

## Neural / tokenizer / packing

| Folder | Proves |
|--------|--------|
| `transformers/` | decoder / encoder / enc-dec; kernel-owned |
| `tokenizer-in-train/` | Tokenizer hashed like weights; eval refuses drift |
| `tokenizer-algorithms/` | byte, BPE, WordPiece, Unigram |
| `pack-mixture/` | Packing + mixture weights + context |

## Foundation models

| Folder | Proves |
|--------|--------|
| `ar-pretrain/` | Causal / next-token |
| `masked-pretrain/` | `mlm/` + `span/` |
| `mtp-pretrain/` | Multi-token prediction |
| `fim-pretrain/` | Fill-in-the-middle |
| `continued-pretrain/` | Parent checkpoint continue |
| `sft/` | Loss on completion only |
| `full-ft/` | Loss on all tokens |
| `lora/` | Frozen parent + rank adapters |
| `qlora/` | 4-bit parent + LoRA |
| `serve-checkpoint/` | `aq serve` → `serve.json` |

## Safety / agent meta

| Folder | Proves |
|--------|--------|
| `guard-safety/` | Opt-in safety: `good/` settles, `bad/` aborts on blow-up |
| `aq-agent-internal-evals/` | Internal agent probes (follow skill, fork, tool, job, eval, stay-in-train, …) - **not** a user feature |

---

When adding a new family to the coverage walk: create a train under `tests/`, write a README with exact `aq` commands and what to look for in inspect/eval, then tick COMPLETED only after a human (or CI you trust) has run it.
