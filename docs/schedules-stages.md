# Schedules and stages

## Job plans (was `schedules/`)

**Schedules are merged into jobs.** Put plan files in `jobs/plans/*.yaml` and use:

```bash
aq job plan [dir]
aq job plan tick [dir]
aq job plan run [dir] <name>
```

See [Jobs](./jobs.md) for plan kinds (cron, sweep, pipeline, resume, agents) and examples.

Legacy `schedules/*.yaml` migrates to `jobs/plans/` on first use.

## Stages (`stages/`)

Nested trains for multi-step pipelines that need isolation:

```
stages/
  pretrain/
    experiment.md
    recipe.yaml
    data/ …
  sft/
    …
```

```bash
aq stage                 # list
aq stage init sft        # scaffold nested train
aq stage pretrain        # aq train that stage
aq stage eval sft        # aq eval that stage
```

Each stage is a **full train** (same schema). Parent `experiment.md` should explain how stages relate.

Job plans can reference stages: `run: stage:sft` in a plan file.
