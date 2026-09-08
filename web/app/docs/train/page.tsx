import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { CodeBlock } from "@/app/docs/_components/CodeBlock";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import {
  TRAIN_FOLDER,
  TRAIN_LAYOUT,
  ARTIFACTS_LAYOUT,
  TRAIN_SLOT_ROWS,
} from "@/lib/docs/sections/train";

export const metadata = buildDocsMetadata("/docs/train");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...TRAIN_FOLDER} />

      <h2
        className="text-xl font-semibold mb-3 mt-12"
        style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
      >
        Canonical layout after aq init
      </h2>
      <p className="text-[15px] text-stone-600 leading-relaxed mb-4 max-w-4xl">
        aq init copies templates and creates optional slots with .keep files. A directory is a train
        when it has both instructions.md and recipe.yaml.
      </p>
      <CodeBlock code={TRAIN_LAYOUT} language="text" label="layout" />

      <h2
        className="text-xl font-semibold mb-3 mt-10"
        style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
      >
        What each slot is for
      </h2>
      <div className="overflow-x-auto mb-10">
        <table className="w-full text-left text-[14px] border-collapse">
          <thead>
            <tr className="border-b border-stone-300">
              <th className="py-2 pr-4 font-semibold text-stone-800">Slot</th>
              <th className="py-2 pr-4 font-semibold text-stone-800">Who uses it</th>
              <th className="py-2 font-semibold text-stone-800">Notes</th>
            </tr>
          </thead>
          <tbody>
            {TRAIN_SLOT_ROWS.map((row) => (
              <tr key={row.slot} className="border-b border-stone-200/80 align-top">
                <td className="py-2.5 pr-4 font-mono text-[13px] text-stone-800 whitespace-nowrap">
                  {row.slot}
                </td>
                <td className="py-2.5 pr-4 text-stone-600 whitespace-nowrap">{row.who}</td>
                <td className="py-2.5 text-stone-600 leading-relaxed">{row.notes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2
        className="text-xl font-semibold mb-3 mt-10"
        style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
      >
        Artifacts after real work
      </h2>
      <p className="text-[15px] text-stone-600 leading-relaxed mb-4 max-w-4xl">
        Everything under artifacts/ is system-owned output. Forking skips jobs/ and artifacts/ so the
        child starts clean. If artifacts/ is deleted, you can retrain from recipe + data alone.
      </p>
      <CodeBlock code={ARTIFACTS_LAYOUT} language="text" label="artifacts" />

      <h2
        className="text-xl font-semibold mb-3 mt-10"
        style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}
      >
        Mental tests
      </h2>
      <ol className="list-decimal pl-5 text-[15px] text-stone-600 leading-relaxed space-y-2 mb-8 max-w-4xl">
        <li>Can I copy the folder to another machine and run aq status?</li>
        <li>Can I fork, change one recipe key, and compare with aq diff?</li>
        <li>Can a stranger read instructions.md and know the gate?</li>
        <li>If artifacts/ is deleted, can I retrain from recipe + data alone?</li>
      </ol>
    </DocsShell>
  );
}
