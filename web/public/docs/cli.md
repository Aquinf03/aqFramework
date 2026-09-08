# CLI

On a **TTY**, bare `aq` (or `aq agent`) opens agent chat. Without a TTY, bare `aq` prints help.

```bash
aq help
aq version
aq update
aq doctor
```

---

## Account & providers

| Command | Purpose |
|---------|---------|
| `aq update` | Install the latest release (same as `curl …/install.sh \| bash`) |
| `aq login` | Open the auth portal; paste the code |
| `aq login --token aq-…` | Install an existing CLI token |
| `aq login --check` | Who is signed in |
| `aq logout [email]` | Drop the active (or that) account |
| `aq switch [email]` | List or switch `~/.aquin` accounts |
| `aq provider` | List openai / anthropic / grok / ollama |
| `aq provider <name>` | Save key (hidden) and select |
| `aq provider use <name>` | Switch active provider |

Model keys live in `~/.aq/config.json`. Aquin account tokens live in `~/.aquin/`. Env fallbacks: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY` / `GROK_API_KEY`.

---

## Trains

| Command | Purpose |
|---------|---------|
| `aq init` | New folder `aq-experiment` (or `-new1`, …) |
| `aq init <name>` | Same with your name |
| `aq fork <new-dir>` | Copy cwd train; skip `jobs/` + `artifacts/` |
| `aq fork <src> <dest>` | Copy that train |
| `aq checkout <id>` | Restore a job tree into cwd |
| `aq checkout <id> <dest>` | New dir with that tree + job |
| `aq data hash [dir] [--snapshot]` | Hash `recipe.data.path` |
| `aq train [dir]` | Fit; checkpoints + metrics |
| `aq eval [dir] [name] [--ckpt name]` | Score `evals/` (all if name omitted) |
| `aq checkpoint [dir] [--keep name]` | List or name a copy of `last` |
| `aq serve [dir] [prompt] [--image path] [--ckpt name] [--max-tokens n] [--temperature t]` | Run the last checkpoint |
| `aq status [dir]` | Jobs, plans, last run, eval, recent metrics |
| `aq diff [dir] [<a> <b>]` | List runs or diff two run records |
| `aq plot [dir] [metrics\|jobs\|runs\|all]` | Charts → `artifacts/plots/` |

### Notes

- Opt-in watches: `guard.safety` / `guard.leak` in the recipe — see [Metrics & guard](./metrics-and-guard.md).  
- `aq eval` is the **human gate**. With `eval.min_score` you get pass/fail; otherwise scores only.  
- `aq serve` works for **every** built-in method (LLM completion, VLM + `--image`, vision classify, CLIP score/embed, tabular predict). Writes `artifacts/serve.json`.  
- `plot.auto: true` in the recipe charts after train.

---

## Tools & stages

| Command | Purpose |
|---------|---------|
| `aq tool [dir]` | List `tools/` |
| `aq tool [dir] <name> [-- args]` | Run `tools/<name>.{py,ts,js,sh}` |
| `aq stage [dir]` | List nested trains under `stages/` |
| `aq stage init <name>` | Scaffold a nested train |
| `aq stage <name>` | Train that stage |
| `aq stage eval <name>` | Eval that stage |

Tools run with cwd = train and `AQ_TRAIN` set to the train path.

---

## Agent & chat

| Command | Purpose |
|---------|---------|
| `aq` / `aq agent` | Interactive chat (TTY) |
| `aq ask [dir] <prompt>` | One-shot answer (no UI) |
| `aq ask -y …` | Auto-approve shell `run` |
| `aq ask --json …` | Machine-readable reply |
| `aq chat list` | Previous chats |
| `aq chat last` | Resume latest |
| `aq chat <id>` | Resume that chat |
| `aq spawn run -- <prompt>` | Background agent worker |
| `aq spawn --kill -- …` | Critic worker that tries to break the idea |
| `aq spawn list` / `log` / `cancel` / `attach` | Manage workers |

Details: [Agent](./agent.md).

---

## Jobs

| Command | Purpose |
|---------|---------|
| `aq job run [--cpu N] [--ram Ng] [-- …]` | Detached command (default: `aq train`) |
| `aq job list` | Running / finished |
| `aq job log <id>` | Tail log |
| `aq job cancel <id>` | Stop |
| `aq job resume <id>` | Resume |
| `aq job tree <id>` | Show checkout tree |
| `aq job plan` / `tick` / `run <name>` | Plans under `jobs/plans/` |
| `aq schedule …` | Alias for `aq job plan …` |

Details: [Jobs & plans](./jobs.md).

---

## Typical flows

**Tabular loop**

```bash
aq init clinic && cd clinic
# edit recipe + data + evals/
aq train && aq eval && aq status
```

**Detached train**

```bash
aq job run --cpu 4 --ram 8G -- aq train
aq job list
aq checkout <id> recovered
```

**Compare forks**

```bash
aq fork ../clinic-ridge
cd ../clinic-ridge && aq train && aq eval
aq diff <run-a> <run-b>
```
