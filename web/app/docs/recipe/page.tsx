import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { RECIPE } from "@/lib/docs/sections/recipe";

export const metadata = buildDocsMetadata("/docs/recipe");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...RECIPE} />
    </DocsShell>
  );
}
