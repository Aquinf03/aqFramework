import { DocsShell } from "@/app/docs/_components/DocsShell";
import AuthPortal from "@/components/AuthPortal";
import { buildDocsMetadata } from "@/lib/docs/metadata";

export const metadata = buildDocsMetadata("/");

export default function HomePage() {
  return (
    <DocsShell>
      <AuthPortal embedded />
    </DocsShell>
  );
}
