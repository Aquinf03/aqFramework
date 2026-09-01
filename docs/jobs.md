# Jobs

Local process queue **and** automation plans. Implementation: `aq/src/job/job.ts`, `plans.ts`, `resources.ts`, `job-wait.mjs`.

## Runs (immediate work)

```
jobs/<id>/
  spec.json       status, cmd, resources, timestamps
  log             combined output
  tree/           snapshot of the train at enqueue
```

`tree/` skips `jobs`, `artifacts`, `node_modules`, `.git` so checkout stays useful without copying history blobs.

Status: `queued` → `starting` → `running` → `exited` | `canceled` | `error`

```bash
aq job run [dir] [--cpu N] [--ram SIZE] [--disk SIZE] [--gpu N] -- <cmd>
aq job list [dir]
aq job log [dir] <id>
aq job cancel [dir] <id>
aq job resume [dir] <id>
aq job tree [dir] <id>
```

Examples:

```bash
aq job run -- aq train
aq job run --cpu 4 --ram 8G -- python scripts/sweep.py
aq checkout <id>            # restore that job's tree into cwd
```

## Plans (cron, sweep, pipeline, resume, agents)

YAML/JSON under **`jobs/plans/`** — same job system, not a separate `schedules/` folder.

```bash
aq job plan [dir]              # list plans
aq job plan tick [dir]         # run due cron/resume plans
aq job plan run [dir] <name>   # fire a plan now
```

`aq schedule …` is an alias for `aq job plan …`.

Example plan (`jobs/plans/nightly.yaml`):

```yaml
kind: cron
run: train
every: 60
```

Plan kinds: **cron**, **sweep**, **resume**, **pipeline**, **agents**. Plans enqueue **job runs** (or spawn workers). State and logs: `artifacts/jobs/plans/`.

`jobs/plans/` **forks** with the train; runtime `jobs/<id>/` dirs do not.

## Other consumers

| Feature | How it uses jobs |
|---------|------------------|
| `aq spawn` | Worker agents = `aq ask -y --json …` as jobs |
| Agent `run` tool with `detach` | Long shell work as a job |

## Design notes

- Jobs are Unix process management on disk, not a cloud control plane.
- Surviving disconnects: detached waiter; `aq job list/log` reconnect to state.
- Forking skips runtime `jobs/<id>/` but copies `jobs/plans/`.
