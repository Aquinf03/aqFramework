import { DocsShell } from "@/app/docs/_components/DocsShell";
import { DocSectionPage } from "@/app/docs/_components/DocSectionPage";
import { buildDocsMetadata } from "@/lib/docs/metadata";
import { SCHEDULES_STAGES } from "@/lib/docs/sections/ops";

export const metadata = buildDocsMetadata("/docs/schedules");

export default function Page() {
  return (
    <DocsShell>
      <DocSectionPage {...SCHEDULES_STAGES} />
    </DocsShell>
  );
}
