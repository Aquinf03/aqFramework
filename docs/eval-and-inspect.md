# Eval and inspect

## User probes (`evals/`)

There is **no bundled benchmark zoo**. You place probes:

```
evals/
  holdout.csv
  leak_check.jsonl
```

```bash
aq eval                 # all probes
aq eval holdout         # one name
aq eval --ckpt 3        # specific checkpoint
```

Results: `artifacts/eval.json` plus metrics events (`eval.probe`). Pass/fail only when `recipe.eval.min_score` is set; otherwise scores are informational.

If `forecast.yaml` exists, `aq eval` also appends **predicted vs actual** to `artifacts/calibration.jsonl`, retunes `artifacts/budget.json` from how honest those intervals were, and writes `artifacts/eval-critique.json` (`n`, trust, ignore-delta). That calibration is the trust signal — not whether the run "passed," and not a fixed sequence of agent steps.

**Humans own the gate.** The agent prompt forbids inventing eval numbers. If you need a gate in CI, run `aq eval` and parse `eval.json` / exit behavior yourself.

## Inspect

Methods may implement `write_inspect(train, model)` → relative markdown path, usually `artifacts/inspect.md`.

This is the human-readable dump: coefficients, tree flowcharts, tokenizer notes, architecture. After `aq train`, read it. After a surprising eval, read it before changing five recipe knobs at once.

## Run records and diff

Each train writes `artifacts/runs/{id}.json` (+ `.md`) with recipe/data/code hashes, metrics summary, artifact paths. `last.json` points at the newest.

```bash
aq diff                 # list run ids
aq diff <a> <b>         # compare two records (files, not screenshots)
aq status               # jobs + last run + eval + recent metrics + schedules
```

Diff is for science: what changed between two fits? Hashes and scores - not chat logs.

## Data hash

```bash
aq data hash
aq data hash --snapshot
```

Pins `data/revision.json` so run records can cite an exact data digest. Snapshots copy content under `data/revisions/{digest}/` for audit.
