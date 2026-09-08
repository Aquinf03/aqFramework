import type { DocSection } from "../types";

export const TRAIN_FOLDER: DocSection = {
  title: "Train folder",
  intro:
    "A train is a directory. If it has instructions.md and recipe.yaml, aq treats it as a real unit of work. aq init scaffolds the layout; disk is the source of truth.",
  prerequisite: "aq installed · run aq init from a writable directory (not inside the aq package tree)",
  tools: [
    {
      command: "aq init [dir]",
      description:
        "Create a skeleton train from aq/templates/. Writes instructions.md, recipe.yaml, optional train.ts, and empty slots (data, evals, methods, tools, skills, memory, sandbox, connections, schedules, stages, jobs, artifacts) with .keep files.",
      flags: [{ name: "[dir]", description: "Target directory. Defaults to cwd." }],
      example: ["aq init my-train", "cd my-train"],
      notes:
        "Creates instructions.md, recipe.yaml, optional train.ts, and empty slots (data, evals, methods, tools, skills, memory, sandbox, connections, schedules, stages, jobs, artifacts) with .keep files. Default template recipe is LLM LoRA. Change recipe.yaml for tabular or transformers. Refuses to init inside the aq package tree.",
    },
    {
      command: "aq fork <dest>",
      description:
        "Copy a train to try a variant. Skips jobs/ and artifacts/, then recreates empty ones so the child starts clean.",
      flags: [
        { name: "aq fork <src> <dest>", description: "Copy a train that is not cwd." },
      ],
      example: ["aq fork ../clinic-ridge", "cd ../clinic-ridge", "aq train", "aq eval"],
    },
    {
      command: "aq checkout <id> [dest]",
      description:
        "Restore a job's captured tree/ into cwd (or a new dest). Time-travel a workspace snapshotted at enqueue.",
      flags: [],
      example: [
        "aq job run --cpu 4 --ram 8G -- aq train",
        "aq checkout <job-id> recovered-train",
      ],
    },
    {
      command: "aq data hash [--snapshot]",
      description:
        "Hash recipe.data.path into data/revision.json. Optional --snapshot copies the hashed tree under data/revisions/{digest}/ so run records can answer what data a checkpoint saw.",
      flags: [{ name: "--snapshot", description: "Copy the hashed data tree under data/revisions/." }],
      example: "aq data hash --snapshot",
    },
    {
      command: "aq status",
      description: "Show jobs, last run, last eval, recent metrics, and schedules for this train.",
      flags: [],
      example: "aq status",
    },
  ],
};

/** Structure trees shown on the train folder page. */
export const TRAIN_LAYOUT = `my-train/
  instructions.md          REQUIRED - what this train is for
  recipe.yaml              REQUIRED - kernel spec (aq does not override)
  train.ts                 OPTIONAL - placeholder; aq does not read yet

  data/                    your datasets (path from recipe)
  evals/                   user probes (.csv / .jsonl) - no bundled zoo
  methods/                 optional train-local fit adapters (override kernel)
  tools/                   scripts: tools/<name>.{py,ts,js,sh}
  skills/                  agent skills (+ optional MCP)
  memory/                  agent memory markdown
  sandbox/                 scratch for agent / tools
  connections/             connection defs (slot)
  schedules/               yaml/json schedules
  stages/                  nested trains (each is itself a train)
  jobs/                    SYSTEM - process queue state
  artifacts/               SYSTEM - checkpoints, metrics, runs, chats, …`;

export const ARTIFACTS_LAYOUT = `artifacts/
  request.json             last kernel request (IPC)
  result.json              last kernel result (IPC)
  metrics.jsonl            append-only observability stream
  checkpoints/
    1.json … N.json
    last.json              always the newest fit
    named.json             from aq checkpoint --keep
  tokenizer.json           pinned when model carries a tokenizer
  inspect.md               human-readable model dump (if method supports it)
  runs/
    {id}.json / .md
    last.json / last.md
  eval.json                last eval summary
  serve.json               last serve output
  chats/<id>/              agent chat sessions
  agents/<id>/             spawned worker agents
  schedules/               schedule run logs`;

export const TRAIN_SLOT_ROWS: { slot: string; who: string; notes: string }[] = [
  { slot: "instructions.md", who: "human / agent", notes: "Brief: what this train proves and what success looks like." },
  { slot: "recipe.yaml", who: "kernel", notes: "Full train API for built-ins. CLI does not invent hyperparameters." },
  { slot: "data/", who: "kernel", notes: "Datasets; recipe.data.path often points here." },
  { slot: "evals/", who: "aq eval", notes: "One file per probe. Gate via eval.min_score." },
  { slot: "methods/", who: "kernel loader", notes: "{name}.py wins over kernel/methods/{name}.py." },
  { slot: "tools/", who: "aq tool / agent", notes: "Executable helpers with AQ_TRAIN set." },
  { slot: "skills/", who: "agent", notes: "SKILL.md / run scripts / mcp.json." },
  { slot: "memory/", who: "agent", notes: "Searchable markdown notes." },
  { slot: "schedules/", who: "aq schedule", notes: "cron, sweep, resume, pipeline, agents." },
  { slot: "stages/", who: "aq stage", notes: "Nested full trains." },
  { slot: "jobs/", who: "job system", notes: "Do not hand-edit casually. Fork skips this." },
  { slot: "artifacts/", who: "everything", notes: "System output. Fork skips this; regenerable but valuable." },
];
