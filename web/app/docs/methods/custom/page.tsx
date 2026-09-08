import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { METHODS_CUSTOM } from "@/lib/docs/sections/methods";

export const metadata = buildDocsMetadata("/docs/methods/custom");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...METHODS_CUSTOM} />
    </DocsShell>
  );
}
