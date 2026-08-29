# Aquin Framework docs

This folder is the deep reference for the whole framework: philosophy, architecture, CLI, Python kernel, methods, agent, jobs, auth portal, install, tests, and roadmap.

**Beta testers:** start with [TLDR](./TLDR.md).

**Start here if you are new:** [Philosophy](./philosophy.md), [Train folder](./train-folder.md), [CLI reference](./cli-reference.md), [Kernel](./kernel.md).

**A train is a directory. `aq` is the CLI. The Python kernel ships inside the `aq` package.**

---

## Contents

| Doc | What it covers |
|-----|----------------|
| [TLDR (beta)](./TLDR.md) | One-pager: install, 10-minute path, what to report |
| [Philosophy](./philosophy.md) | Why Aquin exists, first principles, naming, what “done” means |
| [Architecture](./architecture.md) | Surfaces, TypeScript ↔ Python bridge, package layout, build order |
| [Train folder](./train-folder.md) | Directory protocol, every slot, what artifacts contain |
| [CLI reference](./cli-reference.md) | Every `aq` verb, flags, and typical flows |
| [Kernel](./kernel.md) | `run.py`, ops, protocol modules, IPC, live metrics printing |
| [Recipe](./recipe.md) | `recipe.yaml` keys in depth, families, eval gates, guard |
| [Methods](./methods/README.md) | Built-in fit adapters + how to write your own |
| [Metrics & guard](./metrics-and-guard.md) | `metrics.jsonl`, live step UI, safety / leak watches |
| [Jobs](./jobs.md) | Local process queue, resources, checkout, trees |
| [Agent](./agent.md) | Chat, ask, tools, providers, spawn, doctor |
| [Skills, tools & MCP](./skills-tools-mcp.md) | Extensibility: `skills/`, `tools/`, MCP stdio |
| [Schedules & stages](./schedules-stages.md) | Cron/sweeps/pipelines and nested trains |
| [Eval & inspect](./eval-and-inspect.md) | User probes, gates, `inspect.md`, `diff` / `status` |
| [Web & auth](./web-auth.md) | Auth portal, CLI tokens, desktop handoff, SDK API |
| [Install & release](./install-release.md) | curl install, git, checkout, R2 releases |
| [Tests catalog](./tests-catalog.md) | Every `tests/` family and what it proves |
| [Coverage & roadmap](./coverage-roadmap.md) | What actually works vs unfinished (matches COMPLETED) |

---

## Mental model (one screen)

```
cwd = train/
  instructions.md + recipe.yaml     identity; recipe is the train API
  data/  evals/  methods/  tools/   your science (methods/ only if custom)
  artifacts/                        checkpoints, metrics, runs

aq (TypeScript) → kernel (Python: HF / sklearn)
  Devices: CUDA | MPS | ROCm | CPU
  Unsupported knobs error out (see TODO.md)
```

---

## Related trees in the repo

| Path | Role |
|------|------|
| `aq/` | Shipable CLI + kernel package |
| `tests/` | Manual end-to-end trains (not product evals) |
| `web/` | Auth portal + keyed SDK API |
| `internals/` | Working notes, taxonomy, TODO / COMPLETED |
| `scripts/` | Release + Cloudflare worker |

`internals/` is for builders of the framework. **`docs/` is for anyone using or extending Aquin.** When they disagree, prefer the code and these docs; update `internals` separately.
