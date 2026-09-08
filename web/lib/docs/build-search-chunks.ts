import { createHash } from "crypto";
import type { DocSection, ToolDoc } from "./types";
import { AGENT } from "./sections/agent";
import { INSTALL } from "./sections/install";
import { TRAIN_FOLDER, TRAIN_LAYOUT, ARTIFACTS_LAYOUT, TRAIN_SLOT_ROWS } from "./sections/train";
import { RECIPE } from "./sections/recipe";
import { CLI } from "./sections/cli";
import {
  METHODS_TABULAR,
  METHODS_TRANSFORMERS,
  METHODS_LLM,
  METHODS_CUSTOM,
} from "./sections/methods";
import {
  EVAL_INSPECT,
  METRICS_GUARD,
  JOBS,
  SKILLS_TOOLS,
  SCHEDULES_STAGES,
} from "./sections/ops";
import { DOCS_PAGES } from "./metadata";
import { SHARED_FLAGS } from "./nav";
import { commandAnchor } from "./command-anchor";

export type DocsSearchChunk = {
  id: string;
  group: string;
  title: string;
  description: string;
  href: string;
  text: string;
};

const DOC_SECTIONS: { section: DocSection; href: string; group: string }[] = [
  { section: INSTALL, href: "/docs/install", group: "Install" },
  { section: TRAIN_FOLDER, href: "/docs/train", group: "Train folder" },
  { section: RECIPE, href: "/docs/recipe", group: "Recipe" },
  { section: CLI, href: "/docs/cli", group: "CLI" },
  { section: METHODS_TABULAR, href: "/docs/methods/tabular", group: "Methods · Tabular" },
  { section: METHODS_TRANSFORMERS, href: "/docs/methods/transformers", group: "Methods · Transformers" },
  { section: METHODS_LLM, href: "/docs/methods/llm", group: "Methods · LLM" },
  { section: METHODS_CUSTOM, href: "/docs/methods/custom", group: "Methods · Custom" },
  { section: EVAL_INSPECT, href: "/docs/eval", group: "Eval & inspect" },
  { section: METRICS_GUARD, href: "/docs/metrics", group: "Metrics & guard" },
  { section: JOBS, href: "/docs/jobs", group: "Jobs" },
  { section: AGENT, href: "/docs/agent", group: "Agent" },
  { section: SKILLS_TOOLS, href: "/docs/skills", group: "Skills & tools" },
  { section: SCHEDULES_STAGES, href: "/docs/schedules", group: "Schedules & stages" },
];

function joinParts(parts: (string | undefined)[]): string {
  return parts.filter(Boolean).join("\n");
}

function toolText(tool: ToolDoc, section: DocSection, href: string): string {
  const flags = tool.flags
    .map((f) => `${f.name}${f.required ? " (required)" : ""}: ${f.description}`)
    .join("\n");
  const flagNames = tool.flags.map((f) => f.name).join(" ");
  const examples = Array.isArray(tool.example)
    ? tool.example
    : tool.example
      ? [tool.example]
      : [];
  return joinParts([
    `Section: ${section.title}`,
    `Command: ${tool.command}`,
    tool.agentTool ? `Agent tool: ${tool.agentTool}` : undefined,
    tool.description,
    flags ? `Flags:\n${flags}` : undefined,
    flagNames ? `CLI flags: ${flagNames}` : undefined,
    examples.length ? `Examples:\n${examples.join("\n")}` : undefined,
    tool.notes ? `Notes: ${tool.notes}` : undefined,
    `Documentation: ${href}#${commandAnchor(tool.command)}`,
  ]);
}

function sectionChunks(section: DocSection, href: string, group: string): DocsSearchChunk[] {
  const chunks: DocsSearchChunk[] = [
    {
      id: `page:${href}`,
      group: "Pages",
      title: section.title,
      description: section.intro.slice(0, 220),
      href,
      text: joinParts([
        section.title,
        section.intro,
        section.prerequisite ? `Prerequisite: ${section.prerequisite}` : undefined,
        `Page: ${href}`,
      ]),
    },
  ];

  if (section.prerequisite) {
    chunks.push({
      id: `prereq:${href}`,
      group,
      title: `${section.title}: prerequisites`,
      description: section.prerequisite,
      href,
      text: joinParts([section.title, "Prerequisites", section.prerequisite, href]),
    });
  }

  for (const tool of section.tools) {
    chunks.push({
      id: `cmd:${tool.command}:${href}`,
      group,
      title: tool.command,
      description: tool.description.slice(0, 220),
      href: `${href}#${commandAnchor(tool.command)}`,
      text: toolText(tool, section, href),
    });
  }

  return chunks;
}

function homeChunks(): DocsSearchChunk[] {
  return [
    {
      id: "page:/",
      group: "Pages",
      title: "Home — install, sign in, start a train",
      description:
        "Install the aq CLI, sign in, run aq doctor, and start a train. Account hub for Aquin.",
      href: "/",
      text: joinParts([
        "Home install sign in account",
        "curl install.sh bash",
        "aq login aq doctor aq init my-train",
        "CLI token API key profile",
        "Page: /",
      ]),
    },
    {
      id: "page:/docs",
      group: "Pages",
      title: "Getting started with aq",
      description:
        "What Aquin is: developer environment and framework. Train folder, CLI, kernel, agent.",
      href: "/docs",
      text: joinParts([
        "Getting started with aq",
        "Aquin is a developer environment and framework. A train is a directory.",
        "aq CLI TypeScript face, Python kernel, in-train agent",
        "recipe.yaml is the train API, disk is source of truth",
        "Directory system: instructions.md recipe.yaml data evals methods tools skills artifacts jobs",
        "CLI verbs: train eval serve status diff fork job",
        "Kernel IPC via artifacts/request.json and result.json",
        "Agent same cwd same verbs skills tools MCP",
        "Page: /docs",
      ]),
    },
  ];
}

function trainStructureChunks(): DocsSearchChunk[] {
  return [
    {
      id: "structure:/docs/train:layout",
      group: "Train folder",
      title: "Train folder layout after aq init",
      description: "Canonical my-train/ tree with instructions.md, recipe.yaml, and slots.",
      href: "/docs/train",
      text: joinParts([
        "aq init train folder structure",
        TRAIN_LAYOUT,
        ...TRAIN_SLOT_ROWS.map((r) => `${r.slot}: ${r.who} - ${r.notes}`),
      ]),
    },
    {
      id: "structure:/docs/train:artifacts",
      group: "Train folder",
      title: "artifacts/ after train",
      description: "checkpoints, metrics.jsonl, inspect.md, runs, eval.json, chats.",
      href: "/docs/train",
      text: joinParts(["artifacts layout", ARTIFACTS_LAYOUT]),
    },
  ];
}

function sharedFlagChunks(): DocsSearchChunk[] {
  return SHARED_FLAGS.map((f) => ({
    id: `flag:${f.name}`,
    group: "CLI flags",
    title: f.name,
    description: f.description,
    href: "/docs/cli",
    text: joinParts([`Flag ${f.name}`, f.description, "aq CLI"]),
  }));
}

function metaPageChunks(): DocsSearchChunk[] {
  return DOCS_PAGES.map((p) => ({
    id: `meta:${p.path}`,
    group: "Pages",
    title: p.title,
    description: p.description,
    href: p.path,
    text: joinParts([p.title, p.description, ...(p.keywords ?? []), `Page: ${p.path}`]),
  }));
}

export function buildDocsSearchChunks(): DocsSearchChunk[] {
  const chunks: DocsSearchChunk[] = [
    ...homeChunks(),
    ...metaPageChunks(),
    ...trainStructureChunks(),
    ...sharedFlagChunks(),
  ];

  for (const { section, href, group } of DOC_SECTIONS) {
    chunks.push(...sectionChunks(section, href, group));
  }

  return chunks;
}

export function chunkContentHash(chunks: DocsSearchChunk[]): string {
  const payload = chunks
    .map((c) => `${c.id}\n${c.title}\n${c.description}\n${c.href}\n${c.text}`)
    .join("\n---\n");
  return createHash("sha256").update(payload).digest("hex").slice(0, 16);
}

export function docsContentHash(): string {
  return chunkContentHash(buildDocsSearchChunks());
}
