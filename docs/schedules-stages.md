# Schedules and stages

## Schedules (`schedules/`)

YAML or JSON files describing automation over a train. CLI:

```bash
aq schedule [dir]           # list
aq schedule tick            # run whatever is due
aq schedule run <name>      # fire one now
```

Kinds supported by the scheduler include:

| Kind | Intent |
|------|--------|
| `cron` | Time-based fire |
| `sweep` | Parameter / variant sweeps |
| `resume` | Resume-on-fail patterns |
| `pipeline` | Ordered steps |
| `agents` | Spawn worker agents |

Schedules typically enqueue **jobs** or spawn workers so long work survives disconnects. Logs land under artifacts/schedule-related paths as implemented by `handle/schedule.ts`.

Keep schedules **in the train** so forking a train copies automation intent (you still skip old `jobs/` / `artifacts/`).

## Stages (`stages/`)

Nested trains for multi-step pipelines that need isolation:

```
stages/
  pretrain/
    instructions.md
    recipe.yaml
    data/ …
  sft/
    instructions.md
    recipe.yaml
    …
```

```bash
aq stage                 # list
aq stage init sft        # scaffold nested train
aq stage pretrain        # aq train that stage
aq stage eval sft        # aq eval that stage
```

Each stage is a **full train** (same schema). Parent `instructions.md` should explain how stages relate; do not hide the graph only in a web UI.
