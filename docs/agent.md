# Agent

The agent is an **fx-shaped** resident of the train Unix: same cwd, same files, same `aq` verbs. It is not a separate product with a private filesystem.

Code: `aq/src/agent/` (`agent.ts`, `agent-loop.ts`, `agent-tools.ts`, `chat-ui.ts`, `prompt.ts`, `provider.ts`, `spawn.ts`, `doctor.ts`, …).

## Entry points

| Entry | Behavior |
|-------|----------|
| `aq` / `aq agent` (TTY) | Interactive chat UI |
| `aq ask [dir] <prompt>` | One-shot turn; prints answer |
| `aq ask --json` | `{ text, tools }` for scripts / spawn |
| `aq ask -y` | Auto-approve shell `run` |
| `aq chat list\|last\|<id>` | Resume sessions under `artifacts/chats/` |
| `aq spawn …` | Background workers (jobs + `artifacts/agents/`) |

## Tool rounds

`agent-loop.ts` runs up to a bounded number of tool rounds: stream a model turn, run tools, feed results back, until the model stops calling tools or the cap is hit.

Prompt (`prompt.ts`) emphasizes:

- The folder is the API  
- Cite real files (paths)  
- **Do not invent metrics or eval results** - humans own `aq eval`  
- Prefer `aq` verbs and train tools over hallucinated state  

## Providers

Catalog: **openai**, **anthropic**, **grok**, **ollama** (aliases: oai, claude, xai/x, local).

```bash
aq provider
aq provider openai          # paste key (hidden)
aq provider use anthropic
```

Config: `~/.aq/config.json` (mode `0600`). This is separate from `aq login` / `~/.aquin`.

## Builtin tools (high level)

| Group | Tools |
|-------|-------|
| Filesystem | `ls`, `find`, `glob`, `grep`, `read`, `write`, `edit`, `mkdir`, `mv`, `cp`, `rm` |
| Memory | `memory_search`, `memory_read`, `memory_write` → `memory/*.md` |
| Web | `web_search`, `web_fetch` (native search on some providers) |
| Skills | `skill_load`, `skill_run`, `skills_search` |
| Shell | `run` (needs approval unless `-y`; `detach` → job) |
| CLI | `aq`, `aq_*` wrappers (init, status, train, eval, …) |
| Workers | `spawn`, `spawn_list`, `spawn_log`, `spawn_cancel` |
| Train tools | `tool` → `tools/<name>.{py,ts,js,sh}` |

Path operations stay inside the train (path jail).

## Chat persistence

`artifacts/chats/<id>/{spec.json,messages.json}`  

Slash / menu commands in the UI cover provider, model, key, sound, compact, context, etc.

## Spawn

```bash
aq spawn run -- "investigate why eval failed"
aq spawn list
aq spawn log <id>
aq spawn cancel <id>
```

Each worker is essentially `aq ask -y --json` enqueued as a job, with agent metadata under `artifacts/agents/<id>/`.

## Doctor

```bash
aq doctor [dir]
```

Checks Node ≥ 18, Python, kernel presence, provider/key, train files, tools, skills, MCP probes, tsx if needed. Exit 1 on failure.

## Internal agent evals

`tests/aq-agent-internal-evals/` is **not** a user product feature. It probes whether the agent follows skills, forks instead of mutating, uses tools, files jobs, runs `aq eval`, stays in-train, recovers from failed tools, etc. Tick only when the harness says so.
