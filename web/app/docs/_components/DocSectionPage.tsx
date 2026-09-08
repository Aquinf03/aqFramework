import type { ToolDoc } from "@/lib/docs/types";
import { commandAnchor } from "@/lib/docs/command-anchor";
import { CodeBlock } from "./CodeBlock";

function FlagTable({ flags }: { flags: ToolDoc["flags"] }) {
  if (!flags.length) return null;
  return (
    <div className="overflow-x-auto rounded-lg border border-stone-200 mb-4">
      <table className="w-full text-[12px]">
        <thead>
          <tr className="border-b border-stone-200">
            <th className="text-left font-mono font-semibold text-stone-600 px-3 py-2 w-[32%]">
              Flag
            </th>
            <th className="text-left text-stone-500 px-3 py-2">Description</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200">
          {flags.map((f) => (
            <tr key={f.name}>
              <td className="font-mono text-[11px] text-stone-800 px-3 py-2 align-top whitespace-nowrap">
                {f.name}
                {f.required && <span className="text-red-500 ml-1">*</span>}
              </td>
              <td className="text-stone-600 px-3 py-2 align-top leading-relaxed">{f.description}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ToolDocBlock({ tool }: { tool: ToolDoc }) {
  return (
    <section
      id={commandAnchor(tool.command)}
      className="mb-10 scroll-mt-28 pb-10 border-b border-stone-200 last:border-0"
    >
      <h3 className="font-mono text-[14px] font-semibold text-stone-900 mb-2">{tool.command}</h3>
      {tool.agentTool && (
        <p className="font-mono text-[10px] text-stone-400 mb-3">
          agent tool: <span className="text-stone-500">{tool.agentTool}</span>
        </p>
      )}
      <p className="text-[14px] text-stone-600 leading-relaxed mb-4">{tool.description}</p>
      <FlagTable flags={tool.flags} />
      {(Array.isArray(tool.example) ? tool.example : tool.example ? [tool.example] : []).map(
        (ex, i) => (
          <CodeBlock key={i} code={ex} label="example" />
        ),
      )}
      {tool.notes && (
        <p className="text-[13px] text-stone-500 leading-relaxed border border-stone-200 rounded-lg px-3 py-2">
          {tool.notes}
        </p>
      )}
    </section>
  );
}

export function DocSectionPage({
  title,
  intro,
  prerequisite,
  tools,
  footer,
}: {
  title: string;
  intro: string;
  prerequisite?: string;
  tools: ToolDoc[];
  footer?: React.ReactNode;
}) {
  return (
    <>
      <h1
        className="text-3xl font-normal tracking-tight mb-3"
        style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
      >
        {title}
      </h1>
      <p className="text-[15px] text-stone-600 leading-relaxed mb-4 max-w-2xl">{intro}</p>
      {prerequisite && (
        <div className="text-[13px] text-stone-500 border border-stone-200 rounded-lg px-4 py-3 mb-8 max-w-2xl">
          <span className="text-stone-400 uppercase tracking-widest text-[10px] block mb-1">
            Prerequisite
          </span>
          {prerequisite}
        </div>
      )}
      {tools.map((tool) => (
        <ToolDocBlock key={tool.command} tool={tool} />
      ))}
      {footer}
    </>
  );
}
