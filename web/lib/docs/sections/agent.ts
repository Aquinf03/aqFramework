import type { DocSection } from "../types";

export const AGENT: DocSection = {
  title: "Agent",
  intro:
    "The aq agent lives inside the train Unix: same cwd, same files, same verbs. On a TTY, bare aq opens chat. It is not a chat toy with a private filesystem.",
  prerequisite: "aq provider configured · cwd is a train folder",
  tools: [
    {
      command: "aq / aq agent",
      description: "Interactive agent chat (TTY). Uses tools that wrap aq verbs against this train.",
      flags: [],
      example: "aq",
    },
    {
      command: "aq ask <prompt>",
      description: "One-shot answer without the interactive UI.",
      flags: [
        { name: "-y / --yes", description: "Auto-approve shell run." },
        { name: "--json", description: "Emit { text, tools }." },
      ],
      example: [
        'aq ask "hash the data then train"',
        'aq ask -y "run eval and summarize"',
        'aq ask --json "what failed last run?"',
      ],
    },
    {
      command: "aq chat list | last | <id>",
      description: "List or resume chats under artifacts/chats/.",
      flags: [],
      example: ["aq chat list", "aq chat last", "aq chat <id>"],
    },
    {
      command: "aq spawn run -- <prompt>",
      description: "Background worker agent as a job. list / log / cancel manage workers under artifacts/agents/.",
      flags: [],
      example: [
        'aq spawn run -- "train then eval"',
        "aq spawn list",
        "aq spawn log <id>",
        "aq spawn cancel <id>",
      ],
    },
  ],
};
