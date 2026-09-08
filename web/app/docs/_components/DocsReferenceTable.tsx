export function DocsReferenceTable({ rows }: { rows: [string, string][] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-stone-200 mb-8 bg-stone-100">
      <table className="w-full text-[12px] bg-stone-100">
        <tbody className="divide-y divide-stone-200 bg-stone-100">
          {rows.map(([key, desc]) => (
            <tr key={key} className="bg-stone-100">
              <td className="font-mono text-[11px] text-stone-800 px-4 py-2.5 align-top w-[40%]">{key}</td>
              <td className="text-stone-600 px-4 py-2.5 align-top leading-relaxed">{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
