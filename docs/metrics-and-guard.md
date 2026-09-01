# Metrics and guard

## `artifacts/metrics.jsonl`

Append-only observability log. One JSON object per line. Same idea as a W&B/MLflow stream, but **the train folder is the source of truth**. CLI and web can both tail this file.

Session API (`aq/kernel/protocol/metrics.py`):

| Call | Event | When |
|------|-------|------|
| `begin(...)` | `start` | Start of train/eval/serve |
| `step(...)` | `step` | Each training step (iterative methods) |
| `end(...)` | `end` | Successful finish |
| `event(name, …)` | custom | e.g. `guard.leak`, `eval.probe` |
| (abort path) | `guard.abort` | Safety trip |

Every record includes `ts`, `event`, and usually `run_id`, `op`, `elapsed_ms`.

### Live stderr UI

Every `emit` also prints a short line to stderr (always on - not behind a flag):

```
  train  linear  tabular
  step    12/40   loss  0.887628   lr  0.05   690ms
  done   loss  0.266422   1988ms
```

`aq train` inherits stdio so these lines appear as the kernel runs, not only after `result.json` is written.

---

## Guard (opt-in, fail-closed)

Module: `aq/kernel/protocol/guard.py`  
**Off unless `recipe.guard` enables a watch.** Failures raise `GuardAbort` (subclass of `SystemExit`).

### `guard.safety`

On every `aq_metrics.step(..., loss=…)` when safety is enabled:

1. Non-finite loss → abort  
2. Optional `max_loss` absolute ceiling  
3. After `blowup_warmup` steps: if `loss > best * blowup_factor` → abort  
4. Near-zero best with a large absolute jump → abort  

Also: after fit, if the model dict has `train_loss`, one check runs even for closed-form methods.

Defaults (when enabled): `blowup_factor: 8`, `blowup_warmup: 2`.

### `guard.leak`

Before train (and before eval scoring):

- Fingerprint rows in train data vs all `evals/*.csv|jsonl`
- Fields used: recipe columns among `target`, `text`, `prompt`, `completion`, `instruction`, `output`, `src`, `tgt` (else all columns)
- Any overlap → abort with paths and counts

### Recipe example

```yaml
guard:
  safety: true
  leak: true
  max_loss: null
  blowup_factor: 8
  blowup_warmup: 2
```

### Manual proof

`tests/guard-safety/good` settles under safety.  
`tests/guard-safety/bad` aborts mid-run on blow-up.  
Both use a train-local `tools/watchdemo.py`.
