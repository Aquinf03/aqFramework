# Train folder

A **train** is any directory that contains both:

- **`experiment.md`** — human intent / brief  
- **`recipe.yaml`** — what aq should train and how to eval  

`aq init` creates a **new folder** (`aq-experiment`, or `aq-experiment-new1` if taken). It never dumps files into the current directory. Rename the folder anytime — identity is the two files inside, not the name on the outside.

## Layout

```
my-train/
  experiment.md         required — what this train is for
  recipe.yaml           required — family, method, data, eval

  data/                 your datasets (path from recipe)
  evals/                user probes (.csv / .jsonl)
  tools/                scripts + optional custom fit
  skills/               agent skills (+ optional MCP)
  stages/               nested trains (each is a full train)
  jobs/                 system — runs + plans/
  artifacts/            system — checkpoints, metrics, runs, …
```

## Required files

### `experiment.md`

Plain markdown for humans and the agent. Say what you are trying to prove, what success looks like, and any constraints.

Forking copies this file. Changing it is a deliberate edit.

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

The kernel reads this. The CLI does not invent hyperparameters for you.

## Optional slots

| Slot | Who uses it | Notes |
|------|-------------|-------|
| `data/` | train / eval | `recipe.data.path` often points here |
| `evals/` | `aq eval` | One file per probe; gate with `eval.min_score` |
| `tools/` | `aq tool`, train, agent | Helpers; **`tools/<method>.py` + `fit()`** overrides the built-in |
| `skills/` | agent | Skill markdown / runners / MCP |
| `jobs/plans/` | `aq job plan` | cron, sweep, pipeline, resume, agents |
| `stages/` | `aq stage` | nested full trains |
| `jobs/` | job system | runtime; do not hand-edit casually |
| `artifacts/` | everything | regenerable but valuable history |

## Artifacts

Everything under `artifacts/` is **system-owned**. Forking **skips** `jobs/` and `artifacts/` so the child starts clean.

Typical contents after real work:

```
artifacts/
  metrics.jsonl            append-only live log
  checkpoints/
    1.json                 manifest for run 1
    last.json              pointer to latest
    1/                     weights / adapter / …
  runs/                    one JSON record per train/eval/serve
  eval.json                last eval summary
  inspect.md               human-readable model card from the method
  plots/                   from `aq plot`
  serve.json               last generate
```

## Fork, checkout, stages

```bash
aq fork ../variant-b          # copy train; skip jobs/ + artifacts/
aq checkout <job-id> recovered   # restore a detached job’s tree
aq stage init prep            # nested train under stages/prep/
aq stage prep                 # train that stage
```

Each stage is itself a train (its own `experiment.md` + `recipe.yaml`).

## Next

[Recipe](./recipe.md) · [Eval & inspect](./eval-and-inspect.md) · [Jobs](./jobs.md)
