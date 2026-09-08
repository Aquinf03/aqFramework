import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { JOBS } from "@/lib/docs/sections/ops";

export const metadata = buildDocsMetadata("/docs/jobs");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...JOBS} />
    </DocsShell>
  );
}
