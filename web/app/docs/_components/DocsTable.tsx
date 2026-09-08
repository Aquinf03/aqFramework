import type { ReactNode } from "react";

/**
 * General docs table with a header row and arbitrary columns. Use for
 * comparisons or reference grids that need more than the two-column
 * key/description layout of DocsReferenceTable.
 */
export function DocsTable({
  headers,
  rows,
  monoFirstCol = false,
}: {
  headers: string[];
  rows: ReactNode[][];
  /** Render the first column in monospace (useful for command/flag names). */
  monoFirstCol?: boolean;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-stone-200 mb-8 bg-stone-100">
      <table className="w-full text-[12px] bg-stone-100">
        <thead>
          <tr className="border-b border-stone-200 bg-stone-50">
            {headers.map((h, i) => (
              <th
                key={i}
                className="text-left font-mono text-[10px] uppercase tracking-widest font-semibold text-stone-500 px-4 py-2.5"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 bg-stone-100">
          {rows.map((row, r) => (
            <tr key={r} className="bg-stone-100 align-top">
              {row.map((cell, c) => (
                <td
                  key={c}
                  className={`px-4 py-2.5 leading-relaxed ${
                    monoFirstCol && c === 0
                      ? "font-mono text-[11px] text-stone-800 whitespace-nowrap"
                      : "text-stone-600"
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
