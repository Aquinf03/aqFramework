import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { METHODS_TRANSFORMERS } from "@/lib/docs/sections/methods";

export const metadata = buildDocsMetadata("/docs/methods/transformers");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...METHODS_TRANSFORMERS} />
    </DocsShell>
  );
}
