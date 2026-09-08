# Aquin docs

**How to use `aq`.** Live site: [https://aq.aquin.app/docs](https://aq.aquin.app/docs).  
This folder (`web/content/docs/`) is the **canonical markdown** source.

A train is a folder. The recipe is the train API. Humans own eval.

## Start here

1. [Getting started](./getting-started.md) — install, first train, first LoRA  
2. [Concepts](./concepts.md) — mental model in one page  
3. [Install](./install.md) — curl, checkout, env overrides  

## Everyday work

| Guide | What you learn |
|-------|----------------|
| [Train folder](./train-folder.md) | What lives in a train directory |
| [Recipe](./recipe.md) | Every important `recipe.yaml` knob by family |
| [CLI](./cli.md) | Every `aq` command you need |
| [Methods](./methods/README.md) | Tabular, transformers, LLM, vision, VLM, custom |
| [Eval & inspect](./eval-and-inspect.md) | Probes, gates, status, diff |
| [Metrics & guard](./metrics-and-guard.md) | Live output, JSONL, safety / leak watches |
| [Jobs & plans](./jobs.md) | Background runs, cron, sweeps |
| [Stages](./stages.md) | Nested trains |
| [Agent](./agent.md) | Chat, ask, spawn, providers |
| [Skills, tools & MCP](./skills-tools-mcp.md) | Extend a train without forking the CLI |
| [Login & accounts](./auth.md) | `aq login`, switch, logout |

## Mental model

```
train/                  ← a directory on disk
  experiment.md         ← why this train exists
  recipe.yaml           ← what aq should do
  data/  evals/  tools/
  artifacts/            ← checkpoints, metrics, runs (system-owned)
```

```bash
aq init my-train && cd my-train
# edit recipe.yaml + data
aq train && aq eval && aq status
```

## Authors / internals

Framework layout, kernel IPC, release pipeline, and coverage maps live under **[author/](./author/README.md)** — not required to use `aq`.
