import { readFileSync, existsSync } from "fs";
import path from "path";
import OpenAI from "openai";
import { buildDocsSearchChunks } from "./build-search-chunks";
import { chunkToSearchItem, type DocsSearchItem } from "./search-index";

export const DOCS_EMBEDDING_MODEL = "text-embedding-3-small";

export type DocsSearchEmbeddingRecord = {
  id: string;
  group: string;
  title: string;
  description: string;
  href: string;
  text: string;
  embedding: number[];
};

export type DocsSearchEmbeddingsFile = {
  version: 1;
  model: string;
  contentHash: string;
  builtAt: string;
  records: DocsSearchEmbeddingRecord[];
};

const EMBEDDINGS_PATH = path.join(process.cwd(), "lib/docs/search-embeddings.json");

let cached: DocsSearchEmbeddingsFile | null = null;

export function loadDocsEmbeddings(): DocsSearchEmbeddingsFile | null {
  if (cached) return cached;
  if (!existsSync(EMBEDDINGS_PATH)) return null;
  try {
    const raw = readFileSync(EMBEDDINGS_PATH, "utf8");
    cached = JSON.parse(raw) as DocsSearchEmbeddingsFile;
    return cached;
  } catch {
    return null;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export function keywordSearchChunks(
  chunks: ReturnType<typeof buildDocsSearchChunks>,
  query: string,
  limit = 20,
): DocsSearchItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return chunks.map(chunkToSearchItem);

  const terms = q.split(/\s+/).filter(Boolean);
  const scored = chunks
    .map(chunk => {
      const hay = `${chunk.title} ${chunk.description} ${chunk.text} ${chunk.group}`.toLowerCase();
      let score = 0;
      if (hay.includes(q)) score += 10;
      for (const term of terms) {
        if (chunk.title.toLowerCase().includes(term)) score += 4;
        if (chunk.description.toLowerCase().includes(term)) score += 2;
        if (hay.includes(term)) score += 1;
      }
      // Boost exact CLI flag matches (e.g. --check, --save).
      if (q.startsWith("--") && hay.includes(q)) score += 8;
      return { chunk, score };
    })
    .filter(x => x.score > 0)
    .sort((a, b) => b.score - a.score);

  return scored.slice(0, limit).map(x => chunkToSearchItem(x.chunk));
}

export async function embedQuery(query: string): Promise<number[]> {
  const openaiKey = process.env.OPENAI_API_KEY;
  if (!openaiKey) throw new Error("OPENAI_API_KEY not set");

  const openai = new OpenAI({ apiKey: openaiKey });
  const res = await openai.embeddings.create({
    model: DOCS_EMBEDDING_MODEL,
    input: query.trim(),
  });
  return res.data[0]?.embedding ?? [];
}

export async function semanticSearchDocs(
  query: string,
  limit = 12,
): Promise<{ items: DocsSearchItem[]; mode: "semantic" | "keyword" }> {
  const chunks = buildDocsSearchChunks();
  const trimmed = query.trim();
  if (trimmed.length < 2) {
    return { items: keywordSearchChunks(chunks, trimmed, limit), mode: "keyword" };
  }

  const index = loadDocsEmbeddings();
  if (!index?.records?.length) {
    return { items: keywordSearchChunks(chunks, trimmed, limit), mode: "keyword" };
  }

  const queryEmbedding = await embedQuery(trimmed);
  if (!queryEmbedding.length) {
    return { items: keywordSearchChunks(chunks, trimmed, limit), mode: "keyword" };
  }

  const ranked = index.records
    .map(record => ({
      record,
      score: cosineSimilarity(queryEmbedding, record.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  const semanticItems = ranked.map(x => chunkToSearchItem(x.record));

  const seen = new Set(semanticItems.map(i => i.id));
  const keywordExtras = keywordSearchChunks(chunks, trimmed, limit).filter(i => !seen.has(i.id));

  return {
    items: [...semanticItems, ...keywordExtras].slice(0, limit + 8),
    mode: "semantic",
  };
}

// ── AI reranking ──────────────────────────────────────────────────────────────

type RerankCandidate = {
  id: string;
  group: string;
  title: string;
  description: string;
  href: string;
  text: string;
};

/** Top candidate chunks for AI reranking: embeddings if available, else keyword. */
async function rerankCandidates(query: string, k: number): Promise<RerankCandidate[]> {
  const chunks = buildDocsSearchChunks();
  const index = loadDocsEmbeddings();

  if (index?.records?.length) {
    const queryEmbedding = await embedQuery(query);
    if (queryEmbedding.length) {
      return index.records
        .map(record => ({ record, score: cosineSimilarity(queryEmbedding, record.embedding) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k)
        .map(x => ({
          id: x.record.id,
          group: x.record.group,
          title: x.record.title,
          description: x.record.description,
          href: x.record.href,
          text: x.record.text,
        }));
    }
  }

  const byId = new Map(chunks.map(c => [c.id, c]));
  return keywordSearchChunks(chunks, query, k)
    .map(item => byId.get(item.id))
    .filter((c): c is (typeof chunks)[number] => !!c)
    .map(c => ({
      id: c.id,
      group: c.group,
      title: c.title,
      description: c.description,
      href: c.href,
      text: c.text,
    }));
}

const RERANK_SYSTEM = [
  "You are the search intelligence for the aq (Aquin) documentation.",
  "The user typed an intent: it may be a keyword, a goal (\"save compute\"), or a question (\"how do I use this to ...\").",
  "From the candidate documentation entries, select ONLY the ones that genuinely help with that intent, ordered most useful first.",
  "Think about what the user is trying to accomplish, not just literal word matches. Infer related commands, flags, and pages.",
  "Drop candidates that are not actually relevant. Prefer 3 to 8 strong results over a long list.",
  "For each pick, write a reason of at most 8 words explaining why it helps.",
  "Respond ONLY as JSON: {\"results\":[{\"id\":\"<candidate id>\",\"reason\":\"<short reason>\"}]}.",
  "Use the exact id strings from the candidates. Do not invent ids."].join(" ");

export async function aiRerankDocs(
  query: string,
  limit = 8,
): Promise<{ items: DocsSearchItem[]; reasons: Record<string, string>; mode: "ai" | "semantic" | "keyword" }> {
  const trimmed = query.trim();
  if (trimmed.length < 2) {
    const fallback = await semanticSearchDocs(trimmed, limit);
    return { items: fallback.items, reasons: {}, mode: fallback.mode };
  }

  const openaiKey = process.env.OPENAI_API_KEY;
  if (!openaiKey) {
    const fallback = await semanticSearchDocs(trimmed, limit);
    return { items: fallback.items, reasons: {}, mode: fallback.mode };
  }

  const candidates = await rerankCandidates(trimmed, 24);
  if (!candidates.length) {
    return { items: [], reasons: {}, mode: "semantic" };
  }

  const candidateList = candidates
    .map(c => `- id: ${c.id}\n  title: ${c.title}\n  group: ${c.group}\n  about: ${c.description}`)
    .join("\n");

  try {
    const openai = new OpenAI({ apiKey: openaiKey });
    const res = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.1,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: RERANK_SYSTEM },
        { role: "user", content: `Intent: "${trimmed}"\n\nCandidates:\n${candidateList}` }],
    });

    const raw = res.choices[0]?.message?.content ?? "{}";
    const parsed = JSON.parse(raw) as { results?: { id?: string; reason?: string }[] };
    const byId = new Map(candidates.map(c => [c.id, c]));

    const items: DocsSearchItem[] = [];
    const reasons: Record<string, string> = {};
    for (const r of parsed.results ?? []) {
      if (!r?.id) continue;
      const c = byId.get(r.id);
      if (!c || reasons[c.id]) continue;
      items.push({
        id: c.id,
        group: c.group,
        title: c.title,
        description: c.description,
        href: c.href,
      });
      if (r.reason) reasons[c.id] = r.reason.trim();
      if (items.length >= limit) break;
    }

    if (!items.length) {
      const fallback = await semanticSearchDocs(trimmed, limit);
      return { items: fallback.items, reasons: {}, mode: fallback.mode };
    }

    return { items, reasons, mode: "ai" };
  } catch {
    const fallback = await semanticSearchDocs(trimmed, limit);
    return { items: fallback.items, reasons: {}, mode: fallback.mode };
  }
}
