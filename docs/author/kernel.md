> **Author docs** — for framework builders. End-user guide: [../README.md](../README.md).

# Kernel

Python worker under `aq/kernel/`. Not a user-facing CLI. Spec is always `recipe.yaml` in the train.

```
run.py            dispatch worker
engine/step.py    do_train / do_eval / do_serve / do_checkpoint
protocol/         recipe, revision, record, tokenizer, method, metrics, guard
methods/          fit adapters (filename = method name)
```

Primary short README: `aq/kernel/README.md`.

---

## Invocation

```
aq (TS)  →  artifacts/request.json
         →  python3 run.py <trainAbs>
         →  artifacts/result.json   { ok, lines } | { ok: false, error }
         →  stderr live progress from protocol.metrics
```

Allowed request keys (everything else stripped):  
`op`, `snapshot`, `ckpt`, `keep`, `probe`, `prompt`, `max_tokens`, `temperature`.

| `op` | Handler | Summary |
|------|---------|---------|
| `hash` | `protocol.revision.hash_train` | Hash data path; optional snapshot |
| `train` | `engine.step.do_train` | Fit → checkpoint → metrics → run record |
| `eval` | `engine.step.do_eval` | Score probes vs checkpoint |
| `checkpoint` | `engine.step.do_checkpoint` | List or `--keep` |
| `serve` | `engine.step.do_serve` | `generate()` |

---

## `do_train` (detailed)

1. Load and validate recipe (`protocol.recipe`).
2. Resolve `data.path`; compute recipe hash + data hash.
3. `aq_metrics.begin(op="train", …)` - may print header on stderr; opens `metrics.jsonl` session.
4. If `guard.leak`: fingerprint train vs `evals/*`; abort on overlap.
5. `load_method` + `call_fit` - must return a **dict**; set `kind` from method name.
6. If `guard.safety` and model has `train_loss`: one-shot finite/blowup check.
7. If model has `tokenizer` dict: pin to `artifacts/tokenizer.json` (hashed).
8. Write `artifacts/checkpoints/{n}.json` and copy to `last.json` (`n` = 1 + existing numbered files).
9. Optional `write_inspect` → `artifacts/inspect.md`.
10. `aq_metrics.end(…)`; write run record under `artifacts/runs/`.
11. Return human-readable `lines` for the CLI summary.

Iterative methods should call `aq_metrics.step(step=…, loss=…, lr=…)` during fit so:

- JSONL gets a `step` event per iteration  
- stderr shows a live line like `step  12/40   loss  0.88   lr  0.05   690ms`  
- `guard.safety` can abort mid-run on NaN / blow-up  

---

## `do_eval`

1. Begin metrics session (`op="eval"`).
2. Optional leak check again.
3. Load checkpoint (`--ckpt` or `last.json`); reattach pinned tokenizer if needed.
4. Probe set:
 - named `evals/<probe>.csv|jsonl` if requested  
 - else all files under `evals/`  
 - else fall back to train `data.path`
5. Scoring:
 - methods with `predict` → tabular metrics (`mse`, `rmse`, `mae`, `r2`, `accuracy`)  
 - else method `evaluate(model, src, rec) → (score, n)`
6. Pass gate when `eval.min_score` is a number:
 - lower-is-better for `mse`, `rmse`, `mae`, `loss`  
 - higher-is-better otherwise (e.g. `accuracy`, `r2`)
7. Write `artifacts/eval.json`; emit `eval.probe` metrics events; return lines.

---

## `do_serve`

Requires a prompt from argv or `serve.prompt` (and/or `--image` / `serve.image` / `serve.features`). Method must implement:

```python
def generate(model, prompt, rec, max_tokens=None, temperature=None):
    ...
```

Built-ins cover LLM, VLM, vision classify, CLIP, and tabular. Writes `artifacts/serve.json` (completion/label/score, tokens, …).

---

## `do_checkpoint`

- No `keep`: list checkpoint filenames.  
- With `keep`: copy `last.json` → `artifacts/checkpoints/{name}.json`.

---

## Protocol modules

| Module | Role |
|--------|------|
| `recipe.py` | Minimal YAML load + required-key validation |
| `revision.py` | SHA-256 of file/tree; optional snapshots; `hash_train` |
| `record.py` | Run JSON/MD under `artifacts/runs/` with recipe/data/code hashes |
| `tokenizer.py` | Pin / verify `artifacts/tokenizer.json` |
| `method.py` | Resolve `tools/<name>.py` (train first, then kernel) |
| `metrics.py` | Append-only JSONL + live stderr formatting |
| `guard.py` | Opt-in safety + leak; raises `GuardAbort` |

### Method loader rules

1. Normalize name (`-` → `_`).
2. Reject reserved names (`connection`, `method`, `recipe`, …).
3. Prefer `{train}/tools/{name}.py`, else `{kernel}/methods/{name}.py`.
4. `call_fit`:
 - if `fit` has ≥3 parameters → `fit(src, target, metric)` (linear, logistic)
 - else → `fit(src, rec)`

### Method contract

| Symbol | Required? | Role |
|--------|-----------|------|
| `fit(...)` → `dict` | yes | Train |
| `predict(model, X)` | for tabular eval path | Vector predictions |
| `evaluate(model, src, rec)` | alternative to predict | `(score, n)` |
| `generate(...)` | for serve | Text generation |
| `write_inspect(train, model)` | optional | Relative path to markdown |

---

## Live metrics printing

`protocol.metrics.emit` always:

1. Appends one JSON object to `artifacts/metrics.jsonl`
2. Prints a short human line to **stderr** for `start`, `step`, `end`, `guard.abort`, `eval.probe`, `error`

Example train stream:

```
  train  watchdemo  tabular  guard:safety
  step     0/40   loss  2.35   lr  0.05   0ms
  step     1/40   loss  2.197685   lr  0.05   50ms
  …
  done   loss  0.266422   1988ms
```

Step totals come from `recipe.steps` (or `begin(..., steps=…)`) when present.

---

## Error model

Unhandled exceptions become `{ ok: false, error: msg }` and process exit 1. `GuardAbort` subclasses `SystemExit` and is fail-closed: the job stops; a `guard.abort` event is written first when safety trips during `step()`.
