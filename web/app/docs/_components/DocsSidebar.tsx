"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { DOCS_NAV } from "@/lib/docs/nav";
import type { NavItem } from "@/lib/docs/types";

function navLinkClass(active: boolean, nested?: boolean) {
  return [
    "block rounded-md py-1.5 text-[13px] leading-snug transition-colors",
    nested ? "pl-5 pr-2" : "px-3",
    active
      ? "bg-stone-100 text-stone-900 font-medium"
      : "text-stone-600 hover:bg-stone-100/80 hover:text-stone-900"].join(" ");
}

function NavLink({
  href,
  label,
  nested,
}: {
  href: string;
  label: string;
  nested?: boolean;
}) {
  const pathname = usePathname();
  const active = pathname === href;
  return (
    <Link href={href} className={navLinkClass(active, nested)}>
      {label}
    </Link>
  );
}

function NavSection({ item }: { item: NavItem }) {
  const pathname = usePathname();
  const childActive = item.children?.some(c => c.href === pathname) ?? false;

  return (
    <div className="mt-6 first:mt-0">
      <p
        className={`mb-1.5 px-3 text-[10px] font-mono uppercase tracking-widest ${
          childActive ? "text-stone-500" : "text-stone-400"
        }`}
      >
        {item.label}
      </p>
      <div className="space-y-0.5">
        {item.children?.map(child =>
          child.href ? (
            <NavLink key={child.href} href={child.href} label={child.label} nested />
          ) : null,
        )}
      </div>
    </div>
  );
}

export function DocsSidebar() {
  return (
    <aside
      className="hidden lg:block sticky top-[var(--docs-chrome-top)] self-start max-h-[calc(100vh-var(--docs-chrome-top)-2rem)] overflow-y-auto overscroll-contain min-w-0 pb-8 pr-4 border-r border-stone-200"
      aria-label="Documentation"
    >
      <nav>
        <div className="space-y-0.5">
          {DOCS_NAV.map(item =>
            item.href ? (
              <NavLink key={item.href} href={item.href} label={item.label} />
            ) : (
              <NavSection key={item.label} item={item} />
            ),
          )}
        </div>
      </nav>
    </aside>
  );
}
