import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { AGENT } from "@/lib/docs/sections/agent";

export const metadata = buildDocsMetadata("/docs/agent");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...AGENT} />
    </DocsShell>
  );
}
