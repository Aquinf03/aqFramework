import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { EVAL_INSPECT } from "@/lib/docs/sections/ops";

export const metadata = buildDocsMetadata("/docs/eval");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...EVAL_INSPECT} />
    </DocsShell>
  );
}
