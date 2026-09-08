import { NextRequest, NextResponse } from "next/server";
import { buildDocsSearchChunks } from "@/lib/docs/build-search-chunks";
import {
  aiRerankDocs,
  keywordSearchChunks,
  loadDocsEmbeddings,
  semanticSearchDocs,
} from "@/lib/docs/search-semantic";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const body = (await req.json().catch(() => ({}))) as {
    query?: string;
    limit?: number;
    ai?: boolean;
  };
  const query = typeof body.query === "string" ? body.query : "";
  const limit = typeof body.limit === "number" ? Math.min(Math.max(body.limit, 1), 30) : 16;
  const ai = body.ai === true;

  if (!query.trim()) {
    const chunks = buildDocsSearchChunks();
    return NextResponse.json({
      items: keywordSearchChunks(chunks, "", 200),
      mode: "browse",
      reasons: {},
      indexed: !!loadDocsEmbeddings(),
    });
  }

  try {
    if (ai) {
      const { items, reasons, mode } = await aiRerankDocs(query, limit);
      return NextResponse.json({
        items,
        reasons,
        mode,
        indexed: !!loadDocsEmbeddings(),
      });
    }

    const { items, mode } = await semanticSearchDocs(query, limit);
    return NextResponse.json({
      items,
      mode,
      reasons: {},
      indexed: !!loadDocsEmbeddings(),
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Search failed";
    const chunks = buildDocsSearchChunks();
    return NextResponse.json({
      items: keywordSearchChunks(chunks, query, limit),
      mode: "keyword",
      reasons: {},
      indexed: false,
      error: message,
    });
  }
}
