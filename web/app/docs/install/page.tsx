import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { INSTALL } from "@/lib/docs/sections/install";

export const metadata = buildDocsMetadata("/docs/install");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...INSTALL} />
    </DocsShell>
  );
}
