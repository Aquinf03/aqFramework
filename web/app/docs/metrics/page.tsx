import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { METRICS_GUARD } from "@/lib/docs/sections/ops";

export const metadata = buildDocsMetadata("/docs/metrics");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...METRICS_GUARD} />
    </DocsShell>
  );
}
