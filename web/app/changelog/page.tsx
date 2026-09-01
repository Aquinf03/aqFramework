import type { Metadata } from "next";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { AquinBrand } from "@/components/ui/AquinBrand";
import { ChangelogMarkdown } from "@/components/ChangelogMarkdown";
import { siteConfig } from "@/lib/config";
import { listChangelog } from "@/lib/changelog";
import { constructMetadata } from "@/lib/utils";

export const metadata: Metadata = constructMetadata({
  title: "Changelog",
  description: "Release notes for Aquin and the aq CLI.",
});

export default function ChangelogPage() {
  const entries = listChangelog();

  return (
    <div className="min-h-screen bg-[#f5f5f3] text-stone-900">
      <header className="border-b border-stone-200/80 bg-[#f5f5f3]/95 px-6 py-4 backdrop-blur-sm">
        <div className="mx-auto flex max-w-2xl items-center justify-between gap-4">
          <AquinBrand size="sm" href={siteConfig.links.mainSite} />
          <Link
            href={siteConfig.links.docs}
            className="text-sm font-medium text-stone-600 underline decoration-stone-300 underline-offset-4 hover:text-stone-900"
          >
            Documentation
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-2xl px-6 py-10 pb-16">
        <Link
          href={siteConfig.links.mainSite}
          className="mb-6 inline-flex items-center gap-1.5 text-sm text-stone-500 transition-colors hover:text-stone-800"
        >
          <ArrowLeft className="h-4 w-4" />
          aquin.app
        </Link>

        <h1 className="font-host-grotesk text-3xl font-semibold tracking-[-0.03em] text-stone-900">
          Changelog
        </h1>
        <p className="mt-2 text-sm text-stone-500">
          Release notes for <span className="font-mono text-stone-700">aq</span>. Run{" "}
          <span className="font-mono text-stone-700">aq version</span> to see what you have installed.
        </p>

        <div className="mt-10 space-y-14">
          {entries.length === 0 ? (
            <p className="text-sm text-stone-500">No release notes yet.</p>
          ) : (
            entries.map((entry) => (
              <section key={entry.slug} id={entry.slug} className="scroll-mt-8">
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <h2 className="font-host-grotesk text-xl font-semibold tracking-[-0.02em] text-stone-900">
                    {entry.title}
                  </h2>
                  {entry.unreleased ? (
                    <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide text-amber-800">
                      In progress
                    </span>
                  ) : null}
                </div>
                <div className="mt-4">
                  <ChangelogMarkdown source={entry.body} />
                </div>
              </section>
            ))
          )}
        </div>
      </main>
    </div>
  );
}
