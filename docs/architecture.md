# Architecture

## Surfaces

```
┌─────────────────────────────────────────────────────────────┐
│  Human / Agent / Scripts                                    │
│    aq init | train | eval | job | agent | schedule …        │
└───────────────────────────┬─────────────────────────────────┘
                            │ cwd = train/
┌───────────────────────────▼─────────────────────────────────┐
│  TypeScript CLI  (aq/src)                                   │
│    handle/*  agent/*  job/*  lib/*                          │
│    writes artifacts/request.json                            │
│    spawnSync python3 kernel/run.py                          │
│    streams stderr (live steps)                              │
│    prints artifacts/result.json lines                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Python kernel  (aq/kernel)                                 │
│    run.py → engine/step.py                                  │
│    protocol/{recipe,revision,record,metrics,guard,method…}  │
│    methods/{linear,llm,transformer,…}                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Disk (source of truth)                                     │
│    recipe.yaml  data/  evals/  methods/                     │
│    artifacts/{checkpoints,metrics.jsonl,runs,eval,serve…}   │
│    jobs/  skills/  tools/  schedules/  stages/              │
└─────────────────────────────────────────────────────────────┘

Optional: web/ auth portal (identity, CLI tokens) - does not own trains.
```

## Package layout

```
aqfw/
  aq/                         shipable npm package
    bin/aq                    CLI entry
    src/                      TypeScript
      cli.ts                  verb router
      help.ts
      core/                   schema, paths, python bridge, package root
      handle/                 train verbs (init, fork, status, …)
      agent/                  chat, ask, tools, provider
      job/                    process queue
      lib/                    skills, mcp, memory, files, web
    kernel/                   Python (cwd when spawned)
      run.py
      engine/step.py
      protocol/
      methods/
    templates/                aq init copies these
  docs/                       this documentation
  tests/                      manual E2E trains
  web/                        auth portal + SDK API
  internals/                  builder notes / taxonomy / TODO
  scripts/                    release + Cloudflare worker
  install.sh
```

## TypeScript ↔ Python bridge

Implementation: `aq/src/core/python.ts` + `aq/kernel/run.py`.

1. Resolve train with `assertTrain` (`instructions.md` + `recipe.yaml` required).
2. Ensure `artifacts/` exists.
3. Write `artifacts/request.json` with allowed keys only:
 - `op`: `hash` | `train` | `eval` | `checkpoint` | `serve`
 - optional: `snapshot`, `ckpt`, `keep`, `probe`, `prompt`, `max_tokens`, `temperature`
4. `spawnSync(python3, [kernel/run.py, trainAbsPath], { cwd: kernelRoot, stdio: inherit })`
5. Kernel dispatches, writes `artifacts/result.json`:
 - success: `{ "ok": true, "lines": ["…"] }`
 - failure: `{ "ok": false, "error": "…" }` and exit 1
6. CLI throws on `!ok`; otherwise prints `lines` to stdout.

**Live progress** does not wait for `result.json`. The metrics layer prints step lines to **stderr** during train/eval; inherited stdio shows them immediately. Structured history still appends to `artifacts/metrics.jsonl`.

## How the package finds itself

`aq/src/core/root.ts` walks upward from the compiled file until it finds a directory with both `package.json` and `bin/`. `kernelRoot()` is `aqRoot()/kernel`. After `npm link`, that is your checkout’s `aq/`; after release install, it is under `$AQUIN_INSTALL_DIR` (default `~/.local/share/aquin-framework`).

## Path jail

Agent and many tools use `insideTrain` (`aq/src/core/paths.ts`) so file operations stay under the train directory. Jobs snapshot a `tree/` of the train (skipping `jobs`, `artifacts`, `node_modules`, `.git`) so `aq checkout` can restore a frozen workspace.

## Build order (historical spine)

The framework was built in this order; it is still the right mental stack:

0. **Directory protocol** - schema, init, fork  
1. **CLI** - help, verbs, provider  
2. **Jobs** - queue, resources, checkout  
3. **Kernel** - hash, train, eval, checkpoint, serve  
4. **Grow the handle** - tools, skills, schedules, stages, methods override  
5. **Research tracking** - run records, metrics, diff, status  
6. **Agent** - chat, tools, MCP, spawn, internal agent evals  
7. **Coverage walk** - tabular → transformers → foundation models (see [Coverage](./coverage-roadmap.md))  
8. **Own neural stack (drop HF)** - see `internals/REPLACE_HF.md` 

## Dual configs (do not confuse them)

| File | Purpose |
|------|---------|
| `~/.aq/config.json` | LLM **provider** keys (openai, anthropic, grok, ollama) |
| `~/.aquin/config.json` | **Aquin account** / CLI tokens (`aq-…`) from login |

`aq provider` ≠ `aq login`.
