"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { CircleNotch, MagnifyingGlass, Sparkle, Terminal, X } from "@phosphor-icons/react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { Command as CommandPrimitive } from "cmdk";
import { cn } from "@/lib/utils";
import {
  DOCS_SEARCH_GROUPS,
  DOCS_SEARCH_INDEX,
  type DocsSearchItem,
} from "@/lib/docs/search-index";

type PanelRect = { top: number; left: number; width: number };

function searchValue(item: DocsSearchItem) {
  return [item.title, item.description ?? "", item.group].join(" ").toLowerCase();
}

function keywordFilter(items: DocsSearchItem[], query: string): DocsSearchItem[] {
  const q = query.trim().toLowerCase();
  if (!q) return items;
  const terms = q.split(/\s+/).filter(Boolean);
  return items
    .map(item => {
      const hay = searchValue(item);
      let score = 0;
      if (hay.includes(q)) score += 10;
      for (const term of terms) {
        if (item.title.toLowerCase().includes(term)) score += 4;
        if ((item.description ?? "").toLowerCase().includes(term)) score += 2;
        if (hay.includes(term)) score += 1;
      }
      return { item, score };
    })
    .filter(x => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .map(x => x.item);
}

function groupItems(items: DocsSearchItem[]): { group: string; items: DocsSearchItem[] }[] {
  const byGroup = new Map<string, DocsSearchItem[]>();
  for (const item of items) {
    const list = byGroup.get(item.group) ?? [];
    list.push(item);
    byGroup.set(item.group, list);
  }

  const ordered: { group: string; items: DocsSearchItem[] }[] = [];
  for (const group of DOCS_SEARCH_GROUPS) {
    const list = byGroup.get(group);
    if (list?.length) ordered.push({ group, items: list });
    byGroup.delete(group);
  }
  for (const [group, list] of byGroup) {
    if (list.length) ordered.push({ group, items: list });
  }
  return ordered;
}

function shortcutLabel() {
  if (typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform)) {
    return "⌘K";
  }
  return "Ctrl+K";
}

function useIsMobile() {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(max-width: 639px)");
    const update = () => setIsMobile(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  return isMobile;
}

type SearchMode = "browse" | "keyword" | "semantic" | "ai";

/** Heuristic: treat multi-word or longer queries as intent worth AI reasoning. */
function looksLikeIntent(q: string): boolean {
  return /\s/.test(q) || q.length >= 8;
}

function useDocsSearch(query: string, open: boolean) {
  const [items, setItems] = useState<DocsSearchItem[]>(DOCS_SEARCH_INDEX);
  const [reasons, setReasons] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [mode, setMode] = useState<SearchMode>("browse");
  const reqIdRef = useRef(0);

  useEffect(() => {
    if (!open) return;

    const q = query.trim();
    if (!q) {
      setItems(DOCS_SEARCH_INDEX);
      setReasons({});
      setMode("browse");
      setLoading(false);
      setThinking(false);
      return;
    }

    if (q.length === 1) {
      setItems(keywordFilter(DOCS_SEARCH_INDEX, q));
      setReasons({});
      setMode("keyword");
      setLoading(false);
      setThinking(false);
      return;
    }

    const myId = ++reqIdRef.current;
    const controller = new AbortController();

    // Phase 1: fast embedding/keyword results, shown immediately.
    const fastTimer = window.setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/docs/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, limit: 20 }),
          signal: controller.signal,
        });
        if (!res.ok) throw new Error("Search failed");
        const data = (await res.json()) as { items: DocsSearchItem[]; mode?: string };
        if (reqIdRef.current !== myId) return;
        setItems(data.items ?? []);
        setReasons({});
        setMode(data.mode === "semantic" ? "semantic" : "keyword");
      } catch (err) {
        if ((err as Error).name !== "AbortError" && reqIdRef.current === myId) {
          setItems(keywordFilter(DOCS_SEARCH_INDEX, q));
          setMode("keyword");
        }
      } finally {
        if (reqIdRef.current === myId) setLoading(false);
      }
    }, 150);

    // Phase 2: AI rerank for natural-language intent, refines the list a beat later.
    let aiTimer: number | undefined;
    if (looksLikeIntent(q)) {
      aiTimer = window.setTimeout(async () => {
        setThinking(true);
        try {
          const res = await fetch("/api/docs/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: q, limit: 8, ai: true }),
            signal: controller.signal,
          });
          if (!res.ok) throw new Error("AI search failed");
          const data = (await res.json()) as {
            items: DocsSearchItem[];
            mode?: string;
            reasons?: Record<string, string>;
          };
          if (reqIdRef.current !== myId) return;
          if (data.mode === "ai" && data.items?.length) {
            setItems(data.items);
            setReasons(data.reasons ?? {});
            setMode("ai");
          }
        } catch {
          // keep phase 1 results on failure
        } finally {
          if (reqIdRef.current === myId) setThinking(false);
        }
      }, 500);
    }

    return () => {
      controller.abort();
      window.clearTimeout(fastTimer);
      if (aiTimer) window.clearTimeout(aiTimer);
    };
  }, [query, open]);

  return { items, loading, thinking, mode, reasons };
}

function ResultsList({
  items,
  loading,
  thinking,
  mode,
  reasons,
  onSelect,
}: {
  items: DocsSearchItem[];
  loading: boolean;
  thinking: boolean;
  mode: SearchMode;
  reasons: Record<string, string>;
  onSelect: (href: string) => void;
}) {
  const groups = useMemo(() => groupItems(items), [items]);
  const isAi = mode === "ai";

  if (thinking) {
    return (
      <div className="flex items-center justify-center gap-2 py-10 text-sm text-yellow-600">
        <CircleNotch className="h-4 w-4 animate-spin" weight="bold" />
        Thinking about what you need…
      </div>
    );
  }

  if (loading && !groups.length) {
    return (
      <div className="flex items-center justify-center gap-2 py-10 text-sm text-stone-500">
        <CircleNotch className="h-4 w-4 animate-spin" weight="bold" />
        Searching docs…
      </div>
    );
  }

  if (!groups.length) {
    return <CommandEmpty className="py-8 text-sm text-stone-500">No results found.</CommandEmpty>;
  }

  return (
    <>
      {isAi ? (
        <div className="flex items-center gap-2 border-b border-stone-200/70 px-3 py-2 text-[11px] text-stone-500">
          <Sparkle className="h-3.5 w-3.5 text-yellow-500" weight="fill" />
          Ranked by relevance to your intent
        </div>
      ) : null}
      {groups.map(({ group, items: groupItems }, i) => (
        <div key={group}>
          {i > 0 ? <CommandSeparator className="bg-stone-200" /> : null}
          <CommandGroup heading={group} className="text-stone-500">
            {groupItems.map(item => {
              const reason = reasons[item.id];
              return (
                <CommandItem
                  key={item.id}
                  value={item.id}
                  onSelect={() => onSelect(item.href)}
                  className="cursor-pointer aria-selected:bg-stone-200/50"
                >
                  {item.id.startsWith("cmd:") ? (
                    <Terminal className="h-4 w-4 shrink-0 text-stone-400" weight="bold" />
                  ) : (
                    <MagnifyingGlass className="h-4 w-4 shrink-0 text-stone-400" weight="bold" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-mono text-[12px] text-stone-900">{item.title}</p>
                    {reason ? (
                      <p className="flex items-center gap-1 truncate text-[11px] text-yellow-700/90">
                        <Sparkle className="h-3 w-3 shrink-0 text-yellow-500" weight="fill" />
                        <span className="truncate">{reason}</span>
                      </p>
                    ) : item.description ? (
                      <p className="truncate text-[11px] text-stone-500">{item.description}</p>
                    ) : null}
                  </div>
                </CommandItem>
              );
            })}
          </CommandGroup>
        </div>
      ))}
    </>
  );
}

export function DocsCommandSearch() {
  const router = useRouter();
  const isMobile = useIsMobile();
  const anchorRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [mounted, setMounted] = useState(false);
  const [modKey, setModKey] = useState("⌘K");
  const [panelRect, setPanelRect] = useState<PanelRect | null>(null);

  const { items, loading, thinking, mode, reasons } = useDocsSearch(query, open);

  useEffect(() => {
    setMounted(true);
    setModKey(shortcutLabel());
  }, []);

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
  }, []);

  const updatePanelRect = useCallback(() => {
    const el = anchorRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const margin = 12;
    const maxWidth = window.innerWidth - margin * 2;
    const width = Math.min(Math.max(rect.width, 280), maxWidth);
    const left = Math.max(margin, Math.min(rect.left, window.innerWidth - width - margin));
    setPanelRect({
      top: rect.bottom - 1,
      left,
      width,
    });
  }, []);

  const openSearch = useCallback(() => {
    setOpen(true);
    requestAnimationFrame(() => {
      if (!isMobile) updatePanelRect();
      inputRef.current?.focus();
    });
  }, [isMobile, updatePanelRect]);

  const onSelect = useCallback(
    (href: string) => {
      close();
      router.push(href);
    },
    [close, router],
  );

  useEffect(() => {
    if (!open || isMobile) return;
    updatePanelRect();
    const onLayout = () => updatePanelRect();
    window.addEventListener("resize", onLayout);
    window.addEventListener("scroll", onLayout, true);
    window.visualViewport?.addEventListener("resize", onLayout);
    return () => {
      window.removeEventListener("resize", onLayout);
      window.removeEventListener("scroll", onLayout, true);
      window.visualViewport?.removeEventListener("resize", onLayout);
    };
  }, [open, isMobile, updatePanelRect]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        openSearch();
      }
      if (e.key === "Escape") close();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [close, openSearch]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (anchorRef.current?.contains(target)) return;
      if (panelRef.current?.contains(target)) return;
      close();
    };
    const id = window.setTimeout(() => {
      document.addEventListener("pointerdown", onPointerDown);
    }, 0);
    return () => {
      window.clearTimeout(id);
      document.removeEventListener("pointerdown", onPointerDown);
    };
  }, [open, close]);

  const desktopPanel =
    !isMobile && open && panelRect && mounted
      ? createPortal(
          <div
            ref={panelRef}
            style={{
              position: "fixed",
              top: panelRect.top,
              left: panelRect.left,
              width: panelRect.width,
            }}
            className={cn(
              "z-[100] overflow-hidden rounded-b-lg border border-stone-300 border-t-0 bg-[#f5f5f3]/95 shadow-sm backdrop-blur-sm",
              "animate-in fade-in-0 slide-in-from-top-1 duration-150",
            )}
          >
            <CommandList className="max-h-[min(420px,calc(100vh-5.5rem))] scroll-py-1">
              <ResultsList
                items={items}
                loading={loading}
                thinking={thinking}
                mode={mode}
                reasons={reasons}
                onSelect={onSelect}
              />
            </CommandList>
          </div>,
          document.body,
        )
      : null;

  const mobilePanel =
    isMobile && open && mounted
      ? createPortal(
          <>
            <button
              type="button"
              aria-label="Close search"
              className="fixed inset-0 top-[var(--docs-header-h)] z-[99] bg-stone-900/20"
              onClick={close}
            />
            <div
              ref={panelRef}
              className={cn(
                "fixed inset-x-0 top-[var(--docs-header-h)] z-[100] flex max-h-[calc(100dvh-var(--docs-header-h))] flex-col border-b border-stone-200 bg-[#f5f5f3]/95 shadow-sm backdrop-blur-sm",
                "animate-in fade-in-0 slide-in-from-top-2 duration-150",
              )}
            >
              <div className="flex h-12 shrink-0 items-center gap-2 border-b border-stone-200 px-3">
                <MagnifyingGlass className="h-4 w-4 shrink-0 text-stone-400" weight="bold" />
                <CommandPrimitive.Input
                  ref={inputRef}
                  value={query}
                  onValueChange={setQuery}
                  placeholder="Search commands & docs…"
                  className="flex h-10 w-full flex-1 border-0 bg-transparent text-[15px] text-stone-800 shadow-none outline-none ring-0 placeholder:text-stone-400 focus:outline-none focus:ring-0"
                />
                <button
                  type="button"
                  onClick={close}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-stone-500 hover:bg-stone-200/50"
                  aria-label="Close search"
                >
                  <X className="h-4 w-4" weight="bold" />
                </button>
              </div>
              <CommandList className="min-h-0 flex-1 overflow-y-auto scroll-py-1">
                <ResultsList
                  items={items}
                  loading={loading}
                  thinking={thinking}
                  mode={mode}
                  reasons={reasons}
                  onSelect={onSelect}
                />
              </CommandList>
            </div>
          </>,
          document.body,
        )
      : null;

  return (
    <Command
      shouldFilter={false}
      className="w-full overflow-visible bg-transparent"
    >
      {isMobile ? (
        !open ? (
          <button
            type="button"
            onClick={openSearch}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-stone-200/80 bg-transparent text-stone-500 transition-colors hover:border-stone-300 active:border-stone-300"
            aria-label="Search documentation"
          >
            <MagnifyingGlass className="h-4 w-4" weight="bold" />
          </button>
        ) : null
      ) : (
        <div
          ref={anchorRef}
          role="search"
          onClick={openSearch}
          className={cn(
            "flex h-10 cursor-text items-center gap-2 rounded-lg border border-stone-200/80 bg-transparent px-3 transition-[border-color,background-color,border-radius] duration-200",
            !open && "hover:border-stone-300",
            open && "relative z-[101] rounded-b-none border-stone-300 bg-[#f5f5f3]/95 backdrop-blur-sm",
          )}
        >
          <MagnifyingGlass className="h-4 w-4 shrink-0 text-stone-400 pointer-events-none" weight="bold" />
          <CommandPrimitive.Input
            ref={inputRef}
            value={query}
            onValueChange={setQuery}
            placeholder="Search commands & docs…"
            onFocus={openSearch}
            onClick={e => {
              e.stopPropagation();
              openSearch();
            }}
            className="flex h-9 w-full min-w-0 flex-1 border-0 bg-transparent text-[13px] text-stone-800 shadow-none outline-none ring-0 placeholder:text-stone-400 focus:outline-none focus:ring-0"
          />
          {!open ? (
            <kbd className="hidden sm:inline-flex h-5 shrink-0 items-center rounded border border-stone-200/80 bg-stone-200/30 px-1.5 font-mono text-[10px] font-medium text-stone-500 pointer-events-none">
              {modKey}
            </kbd>
          ) : thinking ? (
            <span className="hidden sm:inline-flex items-center gap-1 text-[9px] text-yellow-600 shrink-0">
              <CircleNotch className="h-3 w-3 animate-spin" weight="bold" />
              thinking
            </span>
          ) : mode === "ai" ? (
            <span className="hidden sm:inline-flex items-center gap-1 text-[9px] text-yellow-600 shrink-0">
              <Sparkle className="h-3 w-3" weight="fill" />
              ai
            </span>
          ) : mode === "semantic" ? (
            <span className="hidden sm:inline text-[9px] text-stone-400 shrink-0">semantic</span>
          ) : null}
        </div>
      )}
      {desktopPanel}
      {mobilePanel}
    </Command>
  );
}
