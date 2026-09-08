import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { CLI } from "@/lib/docs/sections/cli";

export const metadata = buildDocsMetadata("/docs/cli");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...CLI} />
    </DocsShell>
  );
}
