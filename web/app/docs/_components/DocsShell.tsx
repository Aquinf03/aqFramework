import { DocHeader } from "./DocHeader";
import { DocsSidebar } from "./DocsSidebar";
import { DocsAskSidebar } from "./DocsAskSidebar";

export function DocsShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen w-full bg-[#f5f5f3] font-sans text-stone-900">
      <DocHeader />
      <div className="mx-auto max-w-7xl px-6 pb-24 pt-[var(--docs-chrome-top)]">
        <div className="block lg:grid lg:grid-cols-[11.5rem_minmax(0,1fr)] lg:gap-10 xl:grid-cols-[11.5rem_minmax(0,48rem)_11.5rem]">
          <DocsSidebar />
          <main className="min-w-0">{children}</main>
          <DocsAskSidebar />
        </div>
      </div>
    </div>
  );
}
