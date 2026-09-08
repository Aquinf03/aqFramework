import type { DocSection } from "../types";

export const EVAL_INSPECT: DocSection = {
  title: "Eval & inspect",
  intro:
    "No preset zoo. You put probes in evals/. aq eval is the human gate. Interpretability is a capability after fit: artifacts/inspect.md when the method supports it, not a separate product.",
  prerequisite: "Trained checkpoint under artifacts/checkpoints/ · probes in evals/",
  tools: [
    {
      command: "aq eval",
      description:
        "Score every probe in evals/ (or one named file). Writes artifacts/eval.json. Pass/fail only when recipe.eval.min_score is set.",
      flags: [
        { name: "[name]", description: "Probe basename." },
        { name: "--ckpt <name>", description: "Named checkpoint." },
      ],
      example: ["aq eval", "cat artifacts/eval.json"],
    },
    {
      command: "artifacts/inspect.md",
      description:
        "Optional human-readable dump after fit (coefficients, trees, tokenizer notes). Part of how you check a fit, not SAE / activation tracing.",
      flags: [],
      example: "cat artifacts/inspect.md",
    },
    {
      command: "aq diff",
      description: "Compare run records after forking a variant or retraining.",
      flags: [],
      example: "aq diff <run-a> <run-b>",
    },
  ],
};

export const METRICS_GUARD: DocSection = {
  title: "Metrics & guard",
  intro:
    "Live metrics stream on stderr during train and append to artifacts/metrics.jsonl. Guards are opt-in and fail-closed.",
  prerequisite: "recipe.yaml · aq train",
  tools: [
    {
      command: "artifacts/metrics.jsonl",
      description: "Append-only observability stream from the kernel. Pair with aq status for a quick read.",
      flags: [],
      example: ["aq train", "tail -f artifacts/metrics.jsonl", "aq status"],
    },
    {
      command: "guard.safety / guard.leak",
      description:
        "Opt-in in recipe.yaml. safety watches for NaN / blow-up mid-train. leak checks train vs evals overlap.",
      flags: [],
      example: `guard:
  safety: true
  leak: true`,
    },
  ],
};

export const JOBS: DocSection = {
  title: "Jobs",
  intro:
    "Local process queue under jobs/. Resource asks (--cpu, --ram, --disk, --gpu) are against the host, not a different product SKU. Survives disconnects; checkout restores the snapshotted tree.",
  prerequisite: "A train folder",
  tools: [
    {
      command: "aq job run -- <cmd>",
      description: "Enqueue a command (often aq train) with resource asks.",
      flags: [
        { name: "--cpu N", description: "CPU ask." },
        { name: "--ram SIZE", description: "RAM ask (e.g. 8G)." },
        { name: "--disk SIZE", description: "Disk ask." },
        { name: "--gpu N", description: "GPU ask." },
      ],
      example: [
        "aq job run --cpu 4 --ram 8G -- aq train",
        "aq job list",
        "aq job log <id>",
        "aq job cancel <id>",
        "aq job resume <id>",
        "aq job tree <id>",
      ],
    },
  ],
};

export const SKILLS_TOOLS: DocSection = {
  title: "Skills, tools & MCP",
  intro:
    "Convention over registration. Drop a file in the right folder and it exists. Tools are scripts; skills are agent playbooks (optional MCP); memory is markdown the agent can search.",
  prerequisite: "A train folder",
  tools: [
    {
      command: "aq tool <name>",
      description: "Run tools/<name>.{py,ts,js,sh} with cwd = train and AQ_TRAIN set.",
      flags: [],
      example: ["aq tool", "aq tool summarize -- --limit 10"],
    },
    {
      command: "skills/",
      description:
        "Agent skills: .md or a directory with SKILL.md, optional run.py, optional mcp.json. Agent tools: skills_search, skill_load, skill_run. MCP servers become mcp_<skill>_<tool>.",
      flags: [],
      example: `# skills/hash-first/SKILL.md
# skills/hash-first/run.py
# skills/hash-first/mcp.json`,
    },
    {
      command: "memory/",
      description: "Agent-searchable markdown notes for this train.",
      flags: [],
      example: "memory/notes.md",
    },
  ],
};

export const SCHEDULES_STAGES: DocSection = {
  title: "Schedules & stages",
  intro:
    "Schedules compose aq verbs (cron, sweep, resume, pipeline, agents). Stages are nested full trains under stages/<name>/ - each with its own instructions.md and recipe.yaml.",
  prerequisite: "A train folder",
  tools: [
    {
      command: "aq schedule",
      description: "List schedules under schedules/. tick runs due cron/resume entries; run fires one now.",
      flags: [],
      example: ["aq schedule", "aq schedule tick", "aq schedule run nightly"],
    },
    {
      command: "aq stage",
      description: "List, scaffold, train, or eval nested trains under stages/.",
      flags: [],
      example: ["aq stage init prep", "aq stage prep", "aq stage eval prep", "aq stage"],
    },
  ],
};
