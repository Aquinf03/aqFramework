"use client";

/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import { PoliciesDropdown } from "@/components/PoliciesDropdown";
import { CliTokenDropdown } from "@/components/account/CliTokenSection";
import ProfileChip from "@/components/account/ProfileChip";
import { useAuth } from "@/contexts/AuthContext";
import { DocsCommandSearch } from "./DocsCommandSearch";

export function DocHeader() {
  const { user, loading } = useAuth();

  return (
    <header className="docs-chrome-header shrink-0 border-b border-stone-200 bg-[#f5f5f3]/95 backdrop-blur-sm">
      <div className="mx-auto flex h-[var(--docs-header-h)] max-w-7xl items-center gap-3 px-6 sm:gap-4">
        <DocsMobileNav signedIn={Boolean(user)} />
        <div className="flex min-w-0 shrink-0 items-center gap-4">
          <Link href="/" className="flex min-w-0 items-center" title="Aquin Labs">
            <img src="/mainlogo2.png" alt="Aquin Labs" className="mr-1.5 h-[22px]" />
            <span className="text-lg font-semibold tracking-tighter text-stone-900">
              Aquin
              <span className="ml-[0.25ch]">Labs</span>
            </span>
          </Link>
          <PoliciesDropdown />
        </div>
        <div className="flex min-w-0 flex-1 justify-end overflow-visible px-1 sm:justify-center sm:px-6">
          <DocsCommandSearch />
        </div>
        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
          {!loading && user ? (
            <div className="flex shrink-0 items-center gap-2 sm:gap-3">
              <CliTokenDropdown />
              <ProfileChip />
            </div>
          ) : !loading ? (
            <Link
              href="/"
              className="inline-flex h-8 items-center rounded-lg bg-stone-900 px-3 text-[12px] font-semibold text-white transition-colors hover:bg-stone-800 sm:h-9 sm:px-3.5 sm:text-[13px]"
            >
              Sign in
            </Link>
          ) : null}
        </div>
      </div>
    </header>
  );
}

function DocsMobileNav({ signedIn }: { signedIn: boolean }) {
  return (
    <details className="relative lg:hidden">
      <summary className="flex size-9 cursor-pointer list-none items-center justify-center rounded-md text-stone-600 hover:bg-stone-100 hover:text-stone-900 [&::-webkit-details-marker]:hidden">
        <span className="sr-only">Open docs menu</span>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path
            d="M4 7h16M4 12h16M4 17h16"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
          />
        </svg>
      </summary>
      <nav className="absolute left-0 top-full z-50 mt-2 w-56 rounded-xl border border-stone-200 bg-[#f5f5f3] p-2 shadow-lg">
        {[
          ["/", "Home"],
          ["/docs", "Getting started"],
          ["/docs/install", "Install"],
          ["/docs/train", "Train folder"],
          ["/docs/cli", "CLI reference"],
          ["/docs/agent", "Agent"],
          ["https://aquin.app/changelog", "Changelog"],
          ...(signedIn ? [] : [["/", "Sign in"]]),
        ].map(([href, label]) => (
          <Link
            key={`${href}-${label}`}
            href={href}
            className="block rounded-md px-3 py-2 text-[13px] text-stone-700 hover:bg-stone-100 hover:text-stone-900"
          >
            {label}
          </Link>
        ))}
      </nav>
    </details>
  );
}
