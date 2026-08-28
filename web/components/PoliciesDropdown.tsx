"use client";

import { CaretDown } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { siteConfig } from "@/lib/config";
import { POLICY_LINKS } from "@/lib/policies";
import { cn } from "@/lib/utils";

export function PoliciesDropdown() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const mainSite = siteConfig.links.mainSite.replace(/\/$/, "");

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="menu"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1 text-sm font-medium font-host-grotesk text-stone-600 hover:text-stone-900 transition-colors"
      >
        Policies
        <CaretDown
          className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")}
          weight="bold"
        />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute left-0 top-full z-50 mt-2 min-w-[15rem] rounded-xl border border-stone-200 bg-white py-1.5 shadow-lg"
        >
          {POLICY_LINKS.map((link) => (
            <a
              key={link.href}
              role="menuitem"
              href={`${mainSite}${link.href}`}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setOpen(false)}
              className="block px-3.5 py-2 text-sm font-host-grotesk text-stone-700 hover:bg-stone-50 hover:text-stone-900 transition-colors"
            >
              {link.label}
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
