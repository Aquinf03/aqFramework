"use client";

import { useMemo, useState } from "react";
import { Check, Copy } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

export type CodeLanguage = "bash" | "python" | "json" | "javascript" | "yaml" | "text";

function detectLanguage(code: string): CodeLanguage {
  const trimmed = code.trim();
  if (/^\s*[[{]/.test(trimmed) && /[:{}[\]]/.test(trimmed) && !/^family:|^method:|^data:/m.test(trimmed)) {
    return "json";
  }
  if (/^(family|method|model|data|eval|guard|objective|rank|alpha|steps|lr|arch):/m.test(trimmed)) {
    return "yaml";
  }
  if (/^(aq |curl |pip |cd |npm |npx |python |export |git |sudo |source |# |\.\/)/m.test(trimmed)) {
    return "bash";
  }
  if (/\b(def |import |class |print\(|lambda )\b/.test(trimmed)) return "python";
  if (/my-train\/|artifacts\/|instructions\.md|recipe\.yaml/.test(trimmed)) return "text";
  return "bash";
}

function isCommandLanguage(lang: CodeLanguage): boolean {
  return lang === "bash";
}

/** Lines that are real shell commands get a leading `$`. Comments / blanks do not. */
function shellLines(text: string): { prompt: boolean; text: string }[] {
  return text.split("\n").map((line) => {
    const trimmed = line.trimStart();
    if (!trimmed || trimmed.startsWith("#")) {
      return { prompt: false, text: line };
    }
    return { prompt: true, text: line };
  });
}

export function CodeBlock({
  code,
  label,
  language,
}: {
  code: string;
  label?: string;
  language?: CodeLanguage;
}) {
  const [copied, setCopied] = useState(false);
  const text = code.replace(/^\n+|\n+$/g, "");
  const lang = language ?? detectLanguage(text);
  const isShell = isCommandLanguage(lang);
  const lines = useMemo(
    () => (isShell ? shellLines(text) : text.split("\n").map((line) => ({ prompt: false, text: line }))),
    [isShell, text],
  );

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      // clipboard unavailable
    }
  }

  return (
    <div
      className={cn(
        "group relative mb-4 overflow-hidden rounded-lg border border-stone-200 bg-transparent font-mono text-[12px] text-stone-700",
      )}
    >
      <div className="flex items-center justify-between gap-3 border-b border-stone-200 px-3 py-2">
        <span className="text-[10px] uppercase tracking-widest text-stone-400">
          {label ?? (isShell ? "shell" : lang)}
        </span>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium text-stone-400 transition-colors hover:text-stone-700"
          title={copied ? "Copied" : "Copy"}
          aria-label="Copy code"
        >
          {copied ? <Check className="size-3" weight="bold" /> : <Copy className="size-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="m-0 overflow-x-auto px-3 py-3 leading-[1.7]">
        <code className="block whitespace-pre">
          {lines.map((line, i) => (
            <span key={i} className="flex gap-2">
              {isShell && (
                <span className="w-3 shrink-0 select-none text-stone-400">
                  {line.prompt ? "$" : " "}
                </span>
              )}
              <span
                className={cn(
                  "min-w-0 flex-1",
                  !line.prompt && line.text.trimStart().startsWith("#") && "text-stone-400",
                )}
              >
                {line.text || " "}
              </span>
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}
