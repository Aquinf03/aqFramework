# Eval & inspect

## Probes (`evals/`)

There is **no bundled benchmark zoo**. You place the probes:

```
evals/
  holdout.csv
  leak_check.jsonl
```

```bash
aq eval                 # every probe
aq eval holdout         # one name
aq eval --ckpt 3        # specific checkpoint
```

Results land in `artifacts/eval.json` and in the metrics stream. Pass/fail only when `recipe.eval.min_score` is set; otherwise you get scores without an invented verdict.

**You own the gate.** The agent must not invent eval numbers. For CI, run `aq eval` and parse `eval.json` yourself.

On a real terminal, eval shows a **scoreboard** (metric, score, verdict, probes) — different from the train dashboard. `AQ_TUI=0` forces plain tables.

## Inspect

After train, many methods write `artifacts/inspect.md` — coefficients, architecture, tokenizer notes, whatever helps a human. Read it before thrashing five recipe knobs.

## Runs, status, diff

Each train/eval/serve appends `artifacts/runs/{id}.json` (and a short `.md`). `last.json` points at the newest.

```bash
aq status               # jobs, last run, eval, recent metrics
aq diff                 # list run ids
aq diff <a> <b>         # compare two records
```

Diff is for science: hashes and scores between two fits — not chat logs.

## Data hash

```bash
aq data hash
aq data hash --snapshot
```

Pins `data/revision.json` so runs can cite an exact data digest. Snapshots copy content under `data/revisions/{digest}/` for audit.
