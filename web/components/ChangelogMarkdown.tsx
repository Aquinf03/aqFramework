import type { ReactNode } from "react";

function inlineFormat(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return (
        <strong key={i} className="font-semibold text-stone-900">
          {part.slice(2, -2)}
        </strong>
      );
    if (part.startsWith("`") && part.endsWith("`"))
      return (
        <code key={i} className="rounded bg-stone-100 px-1 py-0.5 font-mono text-[13px] text-stone-800">
          {part.slice(1, -1)}
        </code>
      );
    return part;
  });
}

function lineToNode(line: string, key: number) {
  if (line.startsWith("## "))
    return (
      <h2 key={key} className="mt-8 mb-3 text-lg font-semibold text-stone-900">
        {line.slice(3)}
      </h2>
    );
  if (line.startsWith("### "))
    return (
      <h3 key={key} className="mt-5 mb-2 text-sm font-semibold text-stone-800">
        {line.slice(4)}
      </h3>
    );
  if (line.startsWith("- "))
    return (
      <li key={key} className="ml-4 list-disc text-stone-700">
        {inlineFormat(line.slice(2))}
      </li>
    );
  if (line.startsWith("```")) return null;
  if (!line.trim()) return <div key={key} className="h-2" />;
  return (
    <p key={key} className="text-sm leading-relaxed text-stone-700">
      {inlineFormat(line)}
    </p>
  );
}

export function ChangelogMarkdown({ source }: { source: string }) {
  const lines = source.split("\n");
  const nodes: ReactNode[] = [];
  let inCode = false;
  let code: string[] = [];

  lines.forEach((line, i) => {
    if (line.startsWith("```")) {
      if (inCode) {
        nodes.push(
          <pre
            key={`code-${i}`}
            className="my-3 overflow-x-auto rounded-xl border border-stone-200 bg-white px-4 py-3 font-mono text-[13px] text-stone-800"
          >
            {code.join("\n")}
          </pre>,
        );
        code = [];
        inCode = false;
      } else {
        inCode = true;
      }
      return;
    }
    if (inCode) {
      code.push(line);
      return;
    }
    const node = lineToNode(line, i);
    if (node) nodes.push(node);
  });

  return <div className="space-y-1">{nodes}</div>;
}
