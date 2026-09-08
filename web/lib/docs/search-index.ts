import type { DocsSearchChunk } from "./build-search-chunks";
import { buildDocsSearchChunks } from "./build-search-chunks";
export { commandAnchor } from "./command-anchor";

export type DocsSearchItem = {
  id: string;
  group: string;
  title: string;
  description?: string;
  href: string;
};

export function chunkToSearchItem(chunk: DocsSearchChunk): DocsSearchItem {
  return {
    id: chunk.id,
    group: chunk.group,
    title: chunk.title,
    description: chunk.description,
    href: chunk.href,
  };
}

export const DOCS_SEARCH_INDEX: DocsSearchItem[] = buildDocsSearchChunks().map(chunkToSearchItem);

export const DOCS_SEARCH_GROUPS = [
  "Pages",
  "Install",
  "Train folder",
  "Recipe",
  "CLI",
  "CLI flags",
  "Methods · Tabular",
  "Methods · Transformers",
  "Methods · LLM",
  "Methods · Custom",
  "Eval & inspect",
  "Metrics & guard",
  "Jobs",
  "Agent",
  "Skills & tools",
  "Schedules & stages",
] as const;
