# CLI reference

Binary: `aq` → `aq/bin/aq` → `dist/cli.js`. Router: `aq/src/cli.ts`. Help text: `aq/src/help.ts`.

On a **TTY**, bare `aq` (or `aq agent`) opens the agent chat. Without a TTY, bare `aq` prints help.

After TypeScript changes: `cd aq && npm run build`.

---

## Account & providers

| Command | Purpose |
|---------|---------|
| `aq version` | Framework version (`-v` / `--version`; `aq version --verbose` → node + install paths) |
| `aq login` | Open auth portal desktop/CLI handoff; paste code |
| `aq login --token aq-…` | Install an existing CLI token |
| `aq login --check` | Who is signed in |
| `aq logout [email]` | Drop active (or that) account |
| `aq switch [email]` | List / switch `~/.aquin` accounts |
| `aq provider` | List openai / anthropic / grok / ollama |
| `aq provider <name>` | Save key (hidden) and select |
| `aq provider use <name>` | Switch active provider |

Keys for models live in `~/.aq/config.json`. Aquin account tokens live in `~/.aquin/config.json`. Env fallbacks: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY` / `GROK_API_KEY`.

---

## Train commands

| Command | Purpose |
|---------|---------|
| `aq init` | New folder `aq-experiment` (or `aq-experiment-new1`, …) with skeleton |
| `aq init <name>` | Same with that name (also `-newN` if taken). Rename the folder anytime. |
| `aq fork <new-dir>` | Copy cwd train; skip `jobs/` + `artifacts/` |
| `aq fork <src> <dest>` | Copy that train |
| `aq checkout <id>` | Restore job tree into cwd |
| `aq checkout <id> <dest>` | New dir with tree + that job |
| `aq data hash [dir] [--snapshot]` | Hash `recipe.data.path` |
| `aq train [dir]` | Fit; checkpoints + `metrics.jsonl`; live steps on stderr |
| `aq eval [dir] [name] [--ckpt name]` | Score `evals/` (all if name omitted) |
| `aq checkpoint [dir] [--keep name]` | List or name a copy of `last` |
| `aq serve [dir] [prompt] [--ckpt name] [--max-tokens n] [--temperature t]` | Generate |
| `aq status [dir]` | Jobs, job plans, last run, eval, recent metrics |
| `aq diff [dir] [<a> <b>]` | List runs or diff two run records |
| `aq plot [dir] [metrics\|jobs\|runs\|all]` | Matplotlib charts from artifacts → `artifacts/plots/` |
| `aq status [dir]` | Jobs, job plans, last run, eval, recent metrics |
| `aq doctor [dir]` | Health: node, python, kernel, provider, train, skills, MCP |

### Train notes

- Opt-in fail-closed watches: set `guard.safety` / `guard.leak` in `recipe.yaml` (see [Metrics & guard](./metrics-and-guard.md)).
- `aq eval` is the **human gate**. When `eval.min_score` is set, results are pass/fail; otherwise scores are reported without inventing a verdict.
- Serve requires a method that implements `generate(...)`.
- `aq plot` reads `artifacts/metrics.jsonl`, `jobs/*/spec.json`, and `artifacts/runs/*.json`. Config: `recipe.yaml` `plot:` block and `~/.aq/config.json` → `plot`. Set `plot.auto: true` to chart after `aq train`.

---

## Tools, stages

| Command | Purpose |
|---------|---------|
| `aq tool [dir]` | List `tools/` |
| `aq tool [dir] <name> [-- args]` | Run `tools/<name>.{py,ts,js,sh}` |
| `aq stage [dir]` | List nested trains under `stages/` |
| `aq stage init <name>` | Scaffold nested train |
| `aq stage <name>` | Train that stage |
| `aq stage eval <name>` | Eval that stage |

Tools run with cwd = train and `AQ_TRAIN` set to the train absolute path.

---

## Agent & chat

| Command | Purpose |
|---------|---------|
| `aq` / `aq agent` | Interactive chat (TTY) |
| `aq ask [dir] <prompt>` | One-shot answer (no UI) |
| `aq ask -y …` / `--yes` | Auto-approve shell `run` |
| `aq ask --json …` | Emit `{ text, tools }` |
| `aq chat list` | Previous chats for this train under `~/.aq/chats/` (`--all` for every chat) |
| `aq chat last` | Resume latest |
| `aq chat <id>` | Resume that id |
| `aq spawn run [dir] -- <prompt>` | Background worker agent (as a job) |
| `aq spawn run --kill -- <prompt>` | Critic worker (cheapest disproof) |
| `aq spawn list [dir]` | Workers |
| `aq spawn log [dir] <id>` | Worker log |
| `aq spawn cancel [dir] <id>` | Cancel worker |

See [Agent](./agent.md).

---

## Jobs

```
aq job run [dir] [--cpu N] [--ram SIZE] [--disk SIZE] [--gpu N] -- <cmd>
aq job list [dir]
aq job log [dir] <id>
aq job cancel [dir] <id>
aq job resume [dir] <id>
aq job tree [dir] <id>
aq job plan [dir]              list plans in jobs/plans/
aq job plan tick [dir]         run due cron/resume plans
aq job plan run [dir] <name>   fire a plan now
aq schedule …                  alias for aq job plan …
```

See [Jobs](./jobs.md).

---

## Typical flows

### Tabular experiment

```bash
aq init clinic-linear
cd clinic-linear
# edit recipe.yaml + put data.csv + evals/holdout.csv
aq train
aq eval
aq status
```

### Fork a variant

```bash
aq fork ../clinic-ridge
cd ../clinic-ridge
# change method: ridge, set lambda
aq train && aq eval
aq diff <run-a> <run-b>
```

### Foundation model serve

```bash
aq train          # recipe method: llm, …
aq serve "hello" --max-tokens 32
cat artifacts/serve.json
```

### Detached heavy work

```bash
aq job run --cpu 4 --ram 8G -- aq train
aq job list
aq checkout <job-id> recovered-train
```

---

## Exit behavior

Kernel failures surface as thrown errors from `runKernel` (message from `result.error`). `aq doctor` exits 1 if any check fails. Job commands operate on `jobs/<id>/spec.json` status fields (`queued`, `running`, `exited`, …).
