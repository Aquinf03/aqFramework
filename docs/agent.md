# Agent

The agent lives in your train the Unix way: same folder, same files, same `aq` verbs. It is not a separate product with a private filesystem.

## Entry points

| Entry | Behavior |
|-------|----------|
| `aq` / `aq agent` (TTY) | Interactive chat |
| `aq ask [dir] <prompt>` | One-shot answer |
| `aq ask --json …` | `{ text, tools }` for scripts |
| `aq ask -y …` | Auto-approve shell `run` |
| `aq chat list` / `last` / `<id>` | Resume sessions (`~/.aq/chats/`) |
| “make a chart” | Agent uses the plot tool → `artifacts/plots/` |
| `aq spawn run -- <prompt>` | Background worker |
| `aq spawn --kill -- …` | Critic that looks for the cheapest disproof |

## How it behaves

- A wish is not permission. Tools stay off until you clearly say go.  
- One heavy step at a time (create files **or** train **or** eval), then it should ask again.  
- Durable lessons go to **`~/.aq/memory/`** (not inside the train).  
- It must **cite real paths** and must **not invent metrics** — you own `aq eval`.

## Providers

```bash
aq provider
aq provider openai          # paste key (hidden)
aq provider use anthropic
```

Catalog: **openai**, **anthropic**, **grok**, **ollama** (aliases: oai, claude, xai, local).

Keys: `~/.aq/config.json`. This is separate from `aq login` / `~/.aquin/`.

## Spawn workers

```bash
aq spawn run -- "find a cheap way this recipe is wrong"
aq spawn list
aq spawn log <id>
aq spawn cancel <id>
aq spawn attach <id>
```

Workers are jobs plus records under `artifacts/agents/`. Use `--kill` when you want a dedicated critic.

## Doctor

```bash
aq doctor
aq doctor /path/to/train
```

Checks Node, Python, kernel, provider, and whether the path is a train.
