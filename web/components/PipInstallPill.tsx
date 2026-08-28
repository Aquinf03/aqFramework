"use client";

import { useState } from "react";
import Link from "next/link";
import { Copy, Check } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

type PipInstallPillProps = {
  className?: string;
  /** Slightly larger for hero / marketing CTAs */
  size?: "sm" | "md";
  /**
   * docs — PyPI icon + divided copy (docs header)
   * hero — undivided copy, trailing docs link
   */
  variant?: "docs" | "hero";
  /** Stretch to parent width (e.g. login stack). Does not change height styling. */
  fullWidth?: boolean;
};

export function PipInstallPill({
  className,
  size = "sm",
  variant = "docs",
  fullWidth = false,
}: PipInstallPillProps) {
  const [copied, setCopied] = useState(false);
  const cmd = "pip install aquin";
  const copy = () => {
    void navigator.clipboard.writeText(cmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const isMd = size === "md";
  const isHero = variant === "hero";
  const iconSize = isMd ? "w-4 h-4" : "w-3.5 h-3.5";

  return (
    <div
      className={cn(
        "flex items-stretch border overflow-hidden font-mono text-stone-700",
        fullWidth
          ? "w-full rounded-2xl border-2 border-black/10 bg-transparent"
          : cn(
              "shrink-0 rounded-lg border",
              isHero
                ? "border-stone-300 bg-transparent"
                : "border-stone-200 bg-[#fafaf9]",
            ),
        isMd ? "text-[14px]" : "text-[12px]",
        className,
      )}
    >
      {!isHero && (
        <div
          className={cn(
            "flex items-center justify-center border-r border-stone-200 bg-stone-50",
            isMd ? "px-3" : "px-2.5",
          )}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/icons/pypi.png"
            alt="PyPI"
            className={cn("object-contain", isMd ? "h-5 w-5" : "h-4 w-4")}
          />
        </div>
      )}
      <span
        className={cn(
          "flex items-center whitespace-nowrap",
          isMd ? "pl-4 py-2.5" : "pl-3 py-2",
          fullWidth && "flex-1 min-w-0",
          isHero ? (isMd ? "pr-2" : "pr-1.5") : isMd ? "pr-4" : "pr-3",
        )}
      >
        <span className={cn(fullWidth && "truncate")}>{cmd}</span>
      </span>
      <button
        type="button"
        onClick={copy}
        className={cn(
          "flex items-center justify-center text-stone-400 hover:text-stone-700 transition-colors",
          fullWidth
            ? "px-3 hover:bg-stone-100"
            : isHero
              ? cn("hover:bg-black/5", isMd ? "pr-3.5 pl-1" : "pr-2.5 pl-0.5")
              : cn(
                  "border-l border-stone-200 hover:bg-stone-100",
                  isMd ? "px-3" : "px-2.5",
                ),
        )}
        aria-label="Copy pip install aquin"
      >
        {copied ? (
          <Check className={cn(iconSize, "text-emerald-500")} weight="bold" />
        ) : (
          <Copy className={iconSize} />
        )}
      </button>
      {isHero && (
        <Link
          href="https://www.aquin.app/docs"
          className={cn(
            "flex items-center font-semibold transition-colors",
            fullWidth
              ? "border-l border-black/10 px-3.5 text-[13px] text-stone-600 hover:text-stone-900 hover:bg-stone-100"
              : cn(
                  "border-l border-stone-300 text-stone-800 hover:bg-black/5",
                  isMd ? "px-3.5 text-[13px]" : "px-3 text-[12px]",
                ),
          )}
          style={{ fontFamily: "inherit" }}
        >
          <span className="font-sans">Docs</span>
        </Link>
      )}
    </div>
  );
}
