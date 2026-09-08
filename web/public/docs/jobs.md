# Jobs & plans

Run work in the background, ask for resources, and recover the tree later. Plans (cron, sweeps, pipelines) live under the same system.

## Immediate runs

```bash
aq job run -- aq train
aq job run --cpu 4 --ram 8G -- python scripts/sweep.py
aq job list
aq job log <id>
aq job cancel <id>
aq job resume <id>
aq job tree <id>
aq checkout <id> recovered-train
```

Each job keeps:

```
jobs/<id>/
  spec.json     status, command, resources, timestamps
  log           combined output
  tree/         snapshot of the train at enqueue
```

Status moves `queued` → `starting` → `running` → `exited` | `canceled` | `error`.

## Plans (`jobs/plans/`)

YAML/JSON plans — not a separate `schedules/` folder.

```bash
aq job plan                 # list
aq job plan tick            # run due cron / resume plans
aq job plan run <name>      # fire one now
aq schedule …               # alias for aq job plan …
```

Example (`jobs/plans/nightly.yaml`):

```yaml
kind: cron
cron: "0 2 * * *"
run: train
```

Other kinds: **sweep**, **pipeline**, **resume**, **agents**. Keep plans small and obvious; put heavy logic in `tools/` scripts the plan calls.

## Tips

- Request only the resources you need (`--cpu`, `--ram`, `--disk`, `--gpu`).  
- `checkout` restores the snapshot without dragging all of `artifacts/` history unless it was in the tree.  
- Prefer `aq job run -- aq train` over long interactive trains on a laptop you will close.
