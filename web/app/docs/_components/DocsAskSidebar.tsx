"use client";

/* eslint-disable @next/next/no-img-element */
import { usePathname } from "next/navigation";
import { buildAskPrompt, getDocsPageMeta } from "@/lib/docs/metadata";
import { ArrowUpRight, Check, Copy } from "@phosphor-icons/react";
import { useState } from "react";

const AI_PROVIDERS = [
  {
    id: "chatgpt",
    label: "ChatGPT",
    img: "/icons/gpt.png",
    buildUrl: (q: string) => `https://chat.openai.com/?q=${encodeURIComponent(q)}`,
  },
  {
    id: "claude",
    label: "Claude",
    img: "/icons/claude.png",
    buildUrl: (q: string) => `https://claude.ai/new?q=${encodeURIComponent(q)}`,
  },
  {
    id: "perplexity",
    label: "Perplexity",
    img: "/icons/perple.png",
    buildUrl: (q: string) => `https://www.perplexity.ai/?q=${encodeURIComponent(q)}`,
  },
] as const;

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };
  return (
    <button
      type="button"
      onClick={copy}
      className="rounded-md p-1 text-stone-400 transition-colors hover:bg-stone-100 hover:text-stone-700"
      aria-label="Copy page context"
    >
      {copied ? (
        <Check className="h-3.5 w-3.5 text-emerald-500" weight="bold" />
      ) : (
        <Copy className="h-3.5 w-3.5" />
      )}
    </button>
  );
}

export function DocsAskSidebar() {
  const pathname = usePathname();
  const page = getDocsPageMeta(pathname);
  const prompt = buildAskPrompt(pathname);

  return (
    <aside className="sticky top-[var(--docs-chrome-top)] hidden min-w-0 self-start overflow-y-auto overscroll-contain border-l border-stone-200 pb-8 pl-4 xl:block max-h-[calc(100vh-var(--docs-chrome-top)-2rem)]">
      <p className="mb-3 px-3 text-[13px] font-semibold leading-snug text-stone-800">
        Ask your AI about aq
      </p>

      <nav className="mb-6 space-y-0.5">
        {AI_PROVIDERS.map((p) => (
          <button
            key={p.id}
            type="button"
            onClick={() => window.open(p.buildUrl(prompt), "_blank", "noopener,noreferrer")}
            className="group flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[13px] text-stone-600 transition-colors hover:bg-stone-100 hover:text-stone-900"
          >
            <img src={p.img} alt="" className="h-4 w-4 shrink-0 rounded-sm object-contain" />
            <span className="flex-1">{p.label}</span>
            <ArrowUpRight className="h-3.5 w-3.5 shrink-0 text-stone-300 group-hover:text-stone-500" />
          </button>
        ))}
      </nav>

      <div className="border-t border-stone-200 px-1 pt-4">
        <div className="mb-2 flex items-center justify-between gap-1 px-2">
          <p className="font-mono text-[10px] uppercase tracking-widest text-stone-400">This page</p>
          <CopyBtn text={prompt} />
        </div>
        <p className="px-2 text-[12px] leading-relaxed text-stone-500">{page.description}</p>
      </div>
    </aside>
  );
}
