# Jobs

Local process queue for arbitrary commands with optional resource asks. Implementation: `aq/src/job/job.ts`, `resources.ts`, `job-wait.mjs`.

Jobs live **inside the train**:

```
jobs/<id>/
  spec.json       status, cmd, resources, timestamps
  log             combined output
  tree/           snapshot of the train at enqueue
```

`tree/` skips `jobs`, `artifacts`, `node_modules`, `.git` so checkout stays useful without copying history blobs.

## Status machine

`queued` → `starting` → `running` → `exited` | `canceled` | `error`

A detached waiter (`job-wait.mjs --dispatch`) packs jobs onto the host using detected CPU/RAM/disk/GPU capacity. Asks are clamped to what the machine has.

## CLI

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
aq checkout <id> recovered  # new dir including the job record
```

## Consumers

| Feature | How it uses jobs |
|---------|------------------|
| Schedules | Enqueue + wait for train/eval/commands |
| `aq spawn` | Worker agents = `aq ask -y --json …` as jobs |
| Agent `run` tool with `detach` | Long shell work as a job |

## Design notes

- Jobs are **not** the cloud control plane; they are Unix process management that can later map to a cluster with the same resource asks.
- Surviving disconnects: the waiter is detached; `aq job list/log` reconnect to on-disk state.
- Forking a train skips `jobs/` so children do not inherit another train’s queue.
