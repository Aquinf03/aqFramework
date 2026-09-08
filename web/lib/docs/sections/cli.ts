import type { DocSection } from "../types";

export const CLI: DocSection = {
  title: "CLI reference",
  intro:
    "Binary: aq. On a TTY, bare aq (or aq agent) opens the agent chat. Without a TTY, bare aq prints help. After TypeScript changes in a checkout: cd aq && npm run build.",
  prerequisite: "aq installed · cwd is usually a train folder",
  tools: [
    {
      command: "aq train",
      description:
        "Fit the recipe. Writes checkpoints under artifacts/checkpoints/, appends artifacts/metrics.jsonl, and records a run. Live steps stream on stderr. Kernel IPC: artifacts/request.json → result.json.",
      flags: [{ name: "[dir]", description: "Train directory (default cwd)." }],
      example: "aq train",
    },
    {
      command: "aq eval [name]",
      description:
        "Score probes in evals/. Humans own the gate. When recipe.eval.min_score is set, results are pass/fail; otherwise scores are reported without inventing a verdict. Writes artifacts/eval.json.",
      flags: [
        { name: "[name]", description: "Probe name. Omit to run all." },
        { name: "--ckpt <name>", description: "Named checkpoint instead of last." },
      ],
      example: ["aq eval", "aq eval holdout --ckpt last"],
    },
    {
      command: "aq checkpoint [--keep name]",
      description: "List checkpoints, or name a copy of last under artifacts/checkpoints/.",
      flags: [{ name: "--keep <name>", description: "Save a named copy of last." }],
      example: "aq checkpoint --keep best",
    },
    {
      command: "aq serve [prompt]",
      description: "Generate from a checkpoint (methods that implement generate). Writes artifacts/serve.json.",
      flags: [
        { name: "--ckpt <name>", description: "Checkpoint to serve." },
        { name: "--max-tokens n", description: "Max new tokens." },
        { name: "--temperature t", description: "Sampling temperature." },
      ],
      example: ['aq serve "hello" --max-tokens 32', "cat artifacts/serve.json"],
    },
    {
      command: "aq diff [<a> <b>]",
      description: "List run records under artifacts/runs/, or diff two run records.",
      flags: [],
      example: "aq diff <run-a> <run-b>",
    },
    {
      command: "aq tool [name]",
      description: "List or run tools/<name>.{py,ts,js,sh}. Runs with cwd = train and AQ_TRAIN set.",
      flags: [{ name: "-- args", description: "Pass-through args after --." }],
      example: ["aq tool", "aq tool summarize -- --limit 10"],
    },
    {
      command: "aq help",
      description: "Short Unix-style help for the aq CLI.",
      flags: [],
      example: "aq help",
    },
  ],
};
