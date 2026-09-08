import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { SKILLS_TOOLS } from "@/lib/docs/sections/ops";

export const metadata = buildDocsMetadata("/docs/skills");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...SKILLS_TOOLS} />
    </DocsShell>
  );
}
