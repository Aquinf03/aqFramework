import type { NavItem } from "./types";

/** Home (`/`) is the account / install hub; Getting started is `/docs`. */
export const DOCS_NAV: NavItem[] = [
  { label: "Home", href: "/", icon: "rocket" },
  { label: "Getting started", href: "/docs", icon: "rocket" },
  { label: "Install", href: "/docs/install", icon: "terminal" },
  { label: "Train folder", href: "/docs/train", icon: "cube" },
  { label: "Recipe", href: "/docs/recipe", icon: "clipboard" },
  { label: "CLI reference", href: "/docs/cli", icon: "terminal" },
  {
    label: "Methods",
    icon: "brain",
    children: [
      { label: "Tabular", href: "/docs/methods/tabular" },
      { label: "Transformers", href: "/docs/methods/transformers" },
      { label: "LLM / LoRA", href: "/docs/methods/llm" },
      { label: "Custom", href: "/docs/methods/custom" },
    ],
  },
  { label: "Eval & inspect", href: "/docs/eval", icon: "magnifyingGlass" },
  { label: "Metrics & guard", href: "/docs/metrics", icon: "chartLine" },
  { label: "Jobs", href: "/docs/jobs", icon: "circlesThree" },
  { label: "Agent", href: "/docs/agent", icon: "terminal" },
  { label: "Skills, tools & MCP", href: "/docs/skills", icon: "cube" },
  { label: "Schedules & stages", href: "/docs/schedules", icon: "chartLine" },
];

/** Common flags that appear across aq commands. */
export const SHARED_FLAGS = [
  {
    name: "[dir]",
    description: "Optional train directory. Defaults to the current working directory.",
  },
  {
    name: "--ckpt <name>",
    description: "Use a named checkpoint under artifacts/checkpoints/ instead of last.",
  },
];
