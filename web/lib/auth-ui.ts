/** Shared Aquin auth portal styling (matches aquin.app marketing site). */

export const inputCls =
  "w-full px-4 py-3 rounded-xl border-2 border-black/10 bg-transparent text-[15px] text-stone-900 placeholder-stone-400 focus:outline-none focus:ring-0 focus:border-2 focus:border-black/20 transition-colors";

/** Email row: bordered shell with borderless field + inset Continue CTA. */
export const inputShellCls =
  "flex items-center gap-2 rounded-xl border-2 border-black/10 bg-transparent p-1.5 pl-4 transition-colors focus-within:border-black/20";

export const inputInnerCls =
  "min-w-0 flex-1 bg-transparent text-[15px] text-stone-900 placeholder-stone-400 focus:outline-none";

export const continueInInputBtnCls =
  "inline-flex h-10 shrink-0 items-center rounded-xl bg-black px-4 text-[15px] font-semibold tracking-tight text-white transition-colors hover:bg-black/90 disabled:cursor-not-allowed disabled:opacity-50";

export const labelCls =
  "block text-xs font-host-grotesk uppercase tracking-widest text-stone-500 mb-2";

export const headingLg =
  "font-host-grotesk text-3xl font-semibold tracking-[-0.03em] text-stone-900 leading-tight";

export const headingMd =
  "font-host-grotesk text-2xl font-semibold tracking-[-0.03em] text-stone-900";

export const panelCls =
  "rounded-xl border border-stone-200 bg-white/80 px-4 py-3";

export const messageErrorCls =
  "text-xs font-host-grotesk text-stone-500 border border-stone-200 rounded-xl px-4 py-3 bg-white/80";

export const messageSuccessCls =
  "text-xs font-host-grotesk text-stone-600 border border-stone-200 rounded-xl px-4 py-3 bg-[#ffee91]/40";

export const primaryBtnCls =
  "w-full inline-flex justify-center items-center gap-2 py-3 px-4 rounded-xl text-sm font-semibold bg-black text-white hover:bg-black/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors";

export const secondaryBtnCls =
  "w-full inline-flex justify-center items-center gap-1.5 rounded-xl border border-stone-300 px-3 py-2.5 text-xs font-medium text-stone-800 hover:bg-[#d6d3d1]/40 transition-colors";

export const ghostBtnCls =
  "w-full py-2.5 px-4 rounded-xl text-xs font-medium border border-stone-200 text-stone-600 hover:bg-white/60 transition-colors";
