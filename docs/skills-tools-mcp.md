# Skills, tools, and MCP

Extensibility is **convention over registration**: drop files in the right folder.

## Tools (`tools/`)

User scripts the CLI and agent can run:

```
tools/
  summarize.py
  plot.ts
  nudge.sh
```

```bash
aq tool                 # list
aq tool summarize -- --limit 10
```

- Resolved as `tools/<name>.{py,ts,js,sh}`  
- cwd = train  
- env: `AQ_TRAIN` = absolute train path  
- Agent tool name: `tool`

Use tools for deterministic helpers you do not want the model to reimplement every turn.

## Skills (`skills/`)

Agent-only (no dedicated `aq skill` CLI). Layouts:

```
skills/
  hash-first.md                 # single-file skill
  leak-check/
    SKILL.md                    # or README.md
    run.py                      # optional runner
    mcp.json                    # optional MCP server config
```

Code: `aq/src/lib/skill.ts`, `skill-runtime.ts`.

Agent tools: `skills_search`, `skill_load`, `skill_run`.

Loading a skill may start an MCP server; its tools appear as `mcp_<skill>_<tool>`.

## MCP

Client: `aq/src/lib/mcp.ts` - stdio JSON-RPC with `Content-Length` framing (`tools/list`, `tools/call`).

Skills are the preferred way to attach MCP to a train: keep the server definition next to the skill that needs it. `aq doctor` can probe MCP health.

## Memory (`memory/`)

Markdown notes the agent can search/read/write. Persistent across chats in the same train. Good for “what we already tried” without polluting `instructions.md`.

## Connections (`connections/`)

Slot for connection definitions (external systems). Convention exists in `aq init`; grow it as the product needs without inventing a second protocol outside the folder.

## Design rule

If something should exist for every clone of the train, put it under the train. If it is an account secret, put it in `~/.aq` or `~/.aquin`. Do not invent a global registry of tools/skills.
