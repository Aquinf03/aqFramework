# Train folder

A **train** is any directory that contains both:

- `experiment.md` - human intent / brief  
- `recipe.yaml` - kernel specification (family, method, data, eval, optional guard)

Authority: `aq/src/core/schema.ts` (`isTrain` / `assertTrain`).

`aq init` creates a **new folder** (`aq-experiment`, or `aq-experiment-new1` if that exists, then `-new2`, …). `aq init my-name` does the same with your name. It never dumps files into the current directory. Rename the folder anytime — a train is identified by `experiment.md` + `recipe.yaml` inside, not by the folder name.

`aq init` copies templates from `aq/templates/` and creates optional slots with `.keep` files. It refuses to initialize inside the `aq` package tree itself. `aq stage init <name>` still writes exactly under `stages/<name>/`.

## Canonical layout

```
my-train/
  experiment.md            REQUIRED - what this train is for
  recipe.yaml              REQUIRED - kernel spec (aq does not override)

  data/                    your datasets (path from recipe)
  evals/                   user probes (.csv / .jsonl) - no bundled zoo
  tools/                   scripts + optional custom fit (tools/<method>.py)
  skills/                  agent skills (+ optional MCP)
  stages/                  nested trains (each is itself a train)
  jobs/                    SYSTEM - runs + plans/ (cron/sweep yaml)
  artifacts/               SYSTEM - checkpoints, metrics, runs, …
```

## Required files

### `experiment.md`

Plain markdown for humans (and the agent). Say what the train is trying to prove, what success looks like, and any constraints. Template:

> What this train is for… Fork the folder to try a variant.

This file is part of the train identity. Forking copies it; changing it is a deliberate edit.

### `recipe.yaml`

See [Recipe](./recipe.md). Minimal shape:

```yaml
family: tabular
method: linear
data:
  path: data.csv
  target: label
eval:
  metric: mse
  min_score: null
```

The kernel reads this. The CLI does not invent or override method hyperparameters for you.

## Optional slots (convention)

| Slot | Who uses it | Notes |
|------|-------------|-------|
| `data/` | kernel | `recipe.data.path` often points here |
| `evals/` | `aq eval` | One file per probe; gate via `eval.min_score` |
| `tools/` | `aq tool`, `aq train`, agent | Helpers; **`tools/<method>.py` + `fit()`** overrides kernel fit |
| `skills/` | agent only | SKILL.md / run scripts / mcp.json |
| `jobs/plans/` | `aq job plan` | cron, sweep, pipeline, resume, agents |
| `stages/` | `aq stage` | nested full trains |
| `jobs/` | job system | runtime runs; do not hand-edit casually |
| `artifacts/` | everything | regenerable but valuable history |

## What artifacts contain

Everything under `artifacts/` is **system-owned output**. Forking a train **skips** `jobs/` and `artifacts/` (then recreates empty dirs) so the child starts clean.

Typical contents after real work:

```
artifacts/
  request.json             last kernel request (IPC)
  result.json              last kernel result (IPC)
  metrics.jsonl            append-only observability stream
  checkpoints/
    1.json … N.json
    last.json              always the newest fit
    named.json             from aq checkpoint --keep
  tokenizer.json           pinned when model carries a tokenizer
  inspect.md               human-readable model dump (if method supports it)
  runs/
    {id}.json / .md
    last.json / last.md
  eval.json                last eval summary
  serve.json               last serve output
  agents/<id>/             spawned worker agents
  schedules/               schedule run logs (as implemented)
```

Agent **chats** live in `~/.aq/chats/` (not in the train), so experiments stay lean.
## Data revision

`aq data hash [dir] [--snapshot]` asks the kernel to hash `recipe.data.path` and write `data/revision.json`. Optional `--snapshot` copies the hashed tree under `data/revisions/{digest}/`. Run records prefer this hash when present so “what data did this checkpoint see?” is answerable from disk.

## Fork and checkout

- **`aq fork <dest>`** - copy the train, omit runtime `jobs/` and `artifacts/`, recreate empty ones. Use this to try a variant without carrying old metrics.
- **`aq checkout <job-id>`** - restore that job’s captured `tree/` into cwd (or a new dest with the job record). This is how you time-travel a workspace that was snapshotted at enqueue.

## Nested trains (`stages/`)

Each `stages/<name>/` is a full train (its own `experiment.md` + `recipe.yaml`). `aq stage init <name>` scaffolds one; `aq stage <name>` trains it; `aq stage eval <name>` evaluates it. Use stages for multi-step pipelines that still want folder isolation.

## Mental tests for “is this a real train?”

1. Can I `cp -R` it to another machine and run `aq status`?
2. Can I fork, change one recipe key, and compare with `aq diff`?
3. Can a stranger read `experiment.md` and know the gate?
4. If `artifacts/` is deleted, can I retrain from recipe + data alone?

If yes, you are using the product as designed.
