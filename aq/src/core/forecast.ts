/** Predict-before-run loop. Files on disk are the science, not chat. */

import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs"
import path from "node:path"
import { isTrain } from "./schema.js"

export const LEARN_EVERY = 10
export const BUDGET_START = 3
export const BUDGET_MIN = 1
export const BUDGET_MAX = 12
export const CAL_MIN_N = 3

export const NOVELTY = ["hyperparam", "method", "data", "architecture"] as const
export type Novelty = (typeof NOVELTY)[number]

export type Forecast = {
  metric: string
  lo: number
  hi: number
  why: string
  novelty: Novelty
  eval_n?: number
  delta_ignore?: number
  eval_note?: string
  parent?: string
  actual?: number
  hit?: boolean
  resolved_at?: string
}

export type CalEvent = {
  at: string
  metric: string
  lo: number
  hi: number
  actual: number
  hit: boolean
  points: number
  why: string
  n?: number
  dest?: string
}

export type Budget = {
  start: number
  current: number
  min: number
  max: number
  forks_used: number
}

export type SearchEvent = {
  at: string
  dest: string
  metric: string
  lo: number
  hi: number
  why: string
  novelty: Novelty
}

function art(train: string, name: string): string {
  return path.join(train, "artifacts", name)
}

function yamlQuote(s: string): string {
  const t = String(s)
  if (t === "") return '""'
  if (/[:#\n"'\\]|^\s|\s$/.test(t)) return JSON.stringify(t)
  return t
}

function asNum(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v
  if (typeof v === "string" && v.trim() && Number.isFinite(Number(v))) return Number(v)
  return undefined
}

function asStr(v: unknown): string {
  return typeof v === "string" ? v : v == null ? "" : String(v)
}

function parseSimpleYaml(text: string): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const line of text.split("\n")) {
    const t = line.split("#", 1)[0]!.trimEnd()
    if (!t.trim()) continue
    const m = t.match(/^([A-Za-z_][\w-]*)\s*:\s*(.*)$/)
    if (!m) continue
    const rest = m[2]!.trim()
    if (!rest) continue
    if ((rest.startsWith('"') && rest.endsWith('"')) || (rest.startsWith("'") && rest.endsWith("'"))) {
      try {
        out[m[1]!] = JSON.parse(rest.startsWith("'") ? `"${rest.slice(1, -1)}"` : rest)
      } catch {
        out[m[1]!] = rest.slice(1, -1)
      }
      continue
    }
    if (rest === "true") out[m[1]!] = true
    else if (rest === "false") out[m[1]!] = false
    else if (rest === "null" || rest === "~") out[m[1]!] = null
    else if (Number.isFinite(Number(rest)) && rest !== "") out[m[1]!] = Number(rest)
    else out[m[1]!] = rest
  }
  return out
}

export function forecastPath(train: string): string {
  return path.join(train, "forecast.yaml")
}

export function readForecast(train: string): Forecast | null {
  const p = forecastPath(train)
  if (!existsSync(p)) return null
  const raw = parseSimpleYaml(readFileSync(p, "utf8"))
  const lo = asNum(raw.lo)
  const hi = asNum(raw.hi)
  const why = asStr(raw.why).trim()
  if (lo == null || hi == null || !why) return null
  const nov = asStr(raw.novelty)
  const novelty: Novelty = (NOVELTY as readonly string[]).includes(nov) ? (nov as Novelty) : "hyperparam"
  const f: Forecast = {
    metric: asStr(raw.metric) || "loss",
    lo,
    hi,
    why,
    novelty,
  }
  const n = asNum(raw.eval_n)
  if (n != null) f.eval_n = n
  const d = asNum(raw.delta_ignore)
  if (d != null) f.delta_ignore = d
  const note = asStr(raw.eval_note).trim()
  if (note) f.eval_note = note
  const parent = asStr(raw.parent).trim()
  if (parent) f.parent = parent
  const actual = asNum(raw.actual)
  if (actual != null) f.actual = actual
  if (typeof raw.hit === "boolean") f.hit = raw.hit
  const resolved = asStr(raw.resolved_at).trim()
  if (resolved) f.resolved_at = resolved
  return f
}

export function writeForecast(train: string, f: Forecast): void {
  if (!(f.hi >= f.lo)) throw new Error("forecast hi must be >= lo")
  if (!f.why.trim()) throw new Error("forecast needs --why (one sentence, before you run)")
  const lines = [
    `# Predicted metric range. Write this before train/fork. aq eval fills actual.`,
    `metric: ${yamlQuote(f.metric)}`,
    `lo: ${f.lo}`,
    `hi: ${f.hi}`,
    `why: ${yamlQuote(f.why.trim())}`,
    `novelty: ${f.novelty}`,
  ]
  if (f.eval_n != null) lines.push(`eval_n: ${f.eval_n}`)
  if (f.delta_ignore != null) lines.push(`delta_ignore: ${f.delta_ignore}`)
  if (f.eval_note) lines.push(`eval_note: ${yamlQuote(f.eval_note)}`)
  if (f.parent) lines.push(`parent: ${yamlQuote(f.parent)}`)
  if (f.actual != null) lines.push(`actual: ${f.actual}`)
  if (f.hit != null) lines.push(`hit: ${f.hit}`)
  if (f.resolved_at) lines.push(`resolved_at: ${yamlQuote(f.resolved_at)}`)
  writeFileSync(forecastPath(train), lines.join("\n") + "\n")
}

export function countEvalRows(train: string): number {
  const d = path.join(train, "evals")
  if (!existsSync(d)) return 0
  let n = 0
  for (const name of readdirSync(d)) {
    if (name.startsWith(".") || name === ".keep") continue
    const p = path.join(d, name)
    try {
      if (!statSync(p).isFile()) continue
    } catch {
      continue
    }
    const ext = path.extname(name).toLowerCase()
    if (ext !== ".jsonl" && ext !== ".csv" && ext !== ".txt") continue
    const body = readFileSync(p, "utf8")
    let csvHeader = ext === ".csv"
    for (const line of body.split("\n")) {
      const t = line.trim()
      if (!t || t.startsWith("#")) continue
      if (csvHeader) {
        csvHeader = false
        continue
      }
      n += 1
    }
  }
  return Math.max(0, n)
}

export function deltaIgnore(n: number, score = 1): number {
  const rows = Math.max(1, n)
  const s = Math.max(Math.abs(score), 1e-6)
  return Math.round(Math.max(0.02, (1.96 * 0.5 * s) / Math.sqrt(rows)) * 1000) / 1000
}

export function evalTrust(n: number): "low" | "medium" | "ok" {
  if (n < 20) return "low"
  if (n < 100) return "medium"
  return "ok"
}

export function intervalPoints(lo: number, hi: number, actual: number): { hit: boolean; points: number } {
  const hit = actual >= lo && actual <= hi
  const width = Math.max(0, hi - lo)
  const scale = Math.max(Math.abs(actual), 1)
  const rel = width / scale
  const sharpness = 1 / (1 + Math.max(0, rel - 0.5))
  return { hit, points: (hit ? 1 : 0) * sharpness }
}

function readJsonl<T>(file: string): T[] {
  if (!existsSync(file)) return []
  const out: T[] = []
  for (const line of readFileSync(file, "utf8").split("\n")) {
    const t = line.trim()
    if (!t) continue
    try {
      out.push(JSON.parse(t) as T)
    } catch {
      /* skip */
    }
  }
  return out
}

function appendJsonl(file: string, row: unknown): void {
  mkdirSync(path.dirname(file), { recursive: true })
  writeFileSync(file, (existsSync(file) ? readFileSync(file, "utf8") : "") + JSON.stringify(row) + "\n")
}

export function readCalibration(train: string): CalEvent[] {
  return readJsonl<CalEvent>(art(train, "calibration.jsonl"))
}

export function readSearch(train: string): SearchEvent[] {
  return readJsonl<SearchEvent>(art(train, "search.jsonl"))
}

export function readBudget(train: string): Budget {
  const p = art(train, "budget.json")
  if (!existsSync(p)) {
    return {
      start: BUDGET_START,
      current: BUDGET_START,
      min: BUDGET_MIN,
      max: BUDGET_MAX,
      forks_used: 0,
    }
  }
  try {
    const j = JSON.parse(readFileSync(p, "utf8")) as Partial<Budget>
    return {
      start: Number(j.start) || BUDGET_START,
      current: Number.isFinite(Number(j.current)) ? Number(j.current) : BUDGET_START,
      min: Number(j.min) || BUDGET_MIN,
      max: Number(j.max) || BUDGET_MAX,
      forks_used: Number(j.forks_used) || 0,
    }
  } catch {
    return {
      start: BUDGET_START,
      current: BUDGET_START,
      min: BUDGET_MIN,
      max: BUDGET_MAX,
      forks_used: 0,
    }
  }
}

function writeBudget(train: string, b: Budget): void {
  mkdirSync(path.join(train, "artifacts"), { recursive: true })
  writeFileSync(art(train, "budget.json"), JSON.stringify(b, null, 2) + "\n")
}

export function calibrationScore(events: CalEvent[]): { n: number; hits: number; coverage: number; score: number } {
  const n = events.length
  if (!n) return { n: 0, hits: 0, coverage: 0, score: 0 }
  const hits = events.filter((e) => e.hit).length
  const score = events.reduce((s, e) => s + e.points, 0) / n
  return { n, hits, coverage: hits / n, score }
}

function retuneBudget(train: string): Budget {
  const b = readBudget(train)
  const { n, score } = calibrationScore(readCalibration(train))
  if (n < CAL_MIN_N) {
    writeBudget(train, b)
    return b
  }
  if (score >= 0.7) b.current = Math.min(b.max, b.current + 1)
  else if (score < 0.4) b.current = Math.max(b.min, b.current - 1)
  writeBudget(train, b)
  return b
}

export function spendForkBudget(train: string): Budget {
  const b = readBudget(train)
  if (b.current < 1) {
    throw new Error(
      `fork budget empty (calibration ${calibrationScore(readCalibration(train)).score.toFixed(2)}). Improve predictions vs actual, or aq fork --force`,
    )
  }
  b.current -= 1
  b.forks_used += 1
  writeBudget(train, b)
  return b
}

export function learnDue(train: string): boolean {
  const forks = readSearch(train).length
  if (forks < LEARN_EVERY) return false
  const p = art(train, "learn.json")
  let last = 0
  if (existsSync(p)) {
    try {
      last = Number((JSON.parse(readFileSync(p, "utf8")) as { forks?: number }).forks) || 0
    } catch {
      last = 0
    }
  }
  return Math.floor(forks / LEARN_EVERY) > Math.floor(last / LEARN_EVERY)
}

export function assertHeuristics(train: string): string {
  const p = path.join(train, "memory", "heuristics.md")
  if (!existsSync(p)) {
    throw new Error("learn due: write memory/heuristics.md (what failed, what not to retry) then aq learn")
  }
  const body = readFileSync(p, "utf8").trim()
  if (body.length < 40) {
    throw new Error("memory/heuristics.md is too thin — write what the last forks taught you, then aq learn")
  }
  return body
}

export function ackLearn(train: string): { forks: number; path: string } {
  const body = assertHeuristics(train)
  const forks = readSearch(train).length
  mkdirSync(path.join(train, "artifacts"), { recursive: true })
  const rec = { at: new Date().toISOString(), forks, chars: body.length }
  writeFileSync(art(train, "learn.json"), JSON.stringify(rec, null, 2) + "\n")
  return { forks, path: "memory/heuristics.md" }
}

export function lastNoveltyStreak(train: string): Novelty[] {
  return readSearch(train)
    .slice(-5)
    .map((e) => e.novelty)
}

export function taxBudget(train: string): Budget {
  const b = readBudget(train)
  if (b.current > 0) {
    b.current = Math.max(b.min, b.current - 1)
    writeBudget(train, b)
  }
  return b
}

export function requireDiverseNovelty(train: string, novelty: string): void {
  const n: Novelty = (NOVELTY as readonly string[]).includes(novelty) ? (novelty as Novelty) : "hyperparam"
  const streak = lastNoveltyStreak(train)
  if (streak.length < 5) return
  if (streak.every((x) => x === "hyperparam") && n === "hyperparam") taxBudget(train)
}

export type ForecastInput = {
  metric?: string
  lo: number
  hi: number
  why: string
  novelty?: string
  n?: number
  note?: string
}

export function inputToForecast(train: string, input: ForecastInput, parent?: string): Forecast {
  const n = input.n ?? countEvalRows(train)
  const metric = input.metric?.trim() || "loss"
  const noveltyRaw = (input.novelty || "hyperparam").trim()
  const novelty: Novelty = (NOVELTY as readonly string[]).includes(noveltyRaw)
    ? (noveltyRaw as Novelty)
    : "hyperparam"
  const note = input.note?.trim()
  const f: Forecast = {
    metric,
    lo: input.lo,
    hi: input.hi,
    why: input.why.trim(),
    novelty,
    eval_n: n,
    delta_ignore: deltaIgnore(n),
  }
  if (note) f.eval_note = note
  else if (n < 20) {
    f.eval_note = `${n} eval rows — ${evalTrust(n)} trust. Ignore deltas under ${f.delta_ignore}.`
  }
  if (parent) f.parent = parent
  return f
}

export function recordSearch(src: string, dest: string, f: Forecast): void {
  appendJsonl(art(src, "search.jsonl"), {
    at: new Date().toISOString(),
    dest,
    metric: f.metric,
    lo: f.lo,
    hi: f.hi,
    why: f.why,
    novelty: f.novelty,
  } satisfies SearchEvent)
}

function writeCritique(train: string, n: number, score: number, metric: string): void {
  const ignore = deltaIgnore(n, score)
  const trust = evalTrust(n)
  mkdirSync(path.join(train, "artifacts"), { recursive: true })
  writeFileSync(
    art(train, "eval-critique.json"),
    JSON.stringify(
      {
        n,
        metric,
        score,
        trust,
        ignore_delta_under: ignore,
        note:
          trust === "ok"
            ? "eval size is enough for coarse deltas"
            : `only ${n} eval rows (${trust} trust). Do not treat deltas under ${ignore} as real.`,
      },
      null,
      2,
    ) + "\n",
  )
}

function resolveParent(train: string, rel: string): string | null {
  const p = path.resolve(train, rel)
  return isTrain(p) ? p : null
}

export function recordEval(train: string): string[] {
  const evPath = art(train, "eval.json")
  if (!existsSync(evPath)) return []
  let ev: { metric?: string; score?: number; n?: number }
  try {
    ev = JSON.parse(readFileSync(evPath, "utf8")) as { metric?: string; score?: number; n?: number }
  } catch {
    return []
  }
  const actual = asNum(ev.score)
  if (actual == null) return []
  const n = ev.n ?? countEvalRows(train)
  const metric = ev.metric || "score"
  writeCritique(train, n, actual, metric)

  const f = readForecast(train)
  if (!f) {
    return [
      "forecast",
      "  (none)  write forecast.yaml before the next fork — predicted range vs this actual",
      "critique",
      `  n ${n}  ${evalTrust(n)}  ignore deltas under ${deltaIgnore(n, actual)}`,
    ]
  }

  const { hit, points } = intervalPoints(f.lo, f.hi, actual)
  const at = new Date().toISOString()
  f.actual = actual
  f.hit = hit
  f.resolved_at = at
  if (f.eval_n == null) f.eval_n = n
  writeForecast(train, f)

  const row: CalEvent = {
    at,
    metric: f.metric || metric,
    lo: f.lo,
    hi: f.hi,
    actual,
    hit,
    points,
    why: f.why,
    n,
  }
  appendJsonl(art(train, "calibration.jsonl"), row)
  retuneBudget(train)

  const parent = f.parent ? resolveParent(train, f.parent) : null
  if (parent) {
    appendJsonl(art(parent, "calibration.jsonl"), { ...row, dest: train })
    retuneBudget(parent)
  }

  const cal = calibrationScore(readCalibration(parent || train))
  const b = readBudget(parent || train)
  return [
    "forecast",
    `  predicted ${f.lo}–${f.hi}  actual ${actual}  ${hit ? "hit" : "miss"}  why ${f.why}`,
    "calibration",
    `  ${cal.hits}/${cal.n} hit  sharpness ${cal.score.toFixed(2)}  budget ${b.current}/${b.max}`,
    "critique",
    `  n ${n}  ${evalTrust(n)}  ignore deltas under ${deltaIgnore(n, actual)}`,
  ]
}

export function labDigest(train: string): string {
  if (!isTrain(train)) return ""
  const lines: string[] = []
  const f = readForecast(train)
  if (f) {
    const pending = f.actual == null ? "pending" : f.hit ? `hit actual ${f.actual}` : `miss actual ${f.actual}`
    lines.push(`forecast: ${f.metric} ${f.lo}–${f.hi} (${pending})`)
    lines.push(`  why: ${f.why}`)
  }
  const cal = calibrationScore(readCalibration(train))
  const b = readBudget(train)
  if (cal.n || existsSync(art(train, "budget.json")) || readSearch(train).length) {
    lines.push(
      `calibration: ${cal.hits}/${cal.n} hit  sharpness ${cal.n ? cal.score.toFixed(2) : "—"}  budget ${b.current} left (${b.forks_used} forks)`,
    )
  }
  if (learnDue(train)) {
    lines.push("heuristics overdue: writing memory/heuristics.md + aq learn clears the tax on future forecast-forks")
  }
  const crit = art(train, "eval-critique.json")
  if (existsSync(crit)) {
    try {
      const c = JSON.parse(readFileSync(crit, "utf8")) as { note?: string }
      if (c.note) lines.push(`eval critique: ${c.note}`)
    } catch {
      /* skip */
    }
  }
  return lines.join("\n")
}

export function formatLab(train: string): string {
  const digest = labDigest(train)
  const f = readForecast(train)
  const cal = calibrationScore(readCalibration(train))
  const b = readBudget(train)
  const search = readSearch(train)
  const out = ["lab"]
  if (digest) for (const line of digest.split("\n")) out.push("  " + line)
  else out.push("  no forecast yet")
  out.push("  forks " + String(search.length) + (learnDue(train) ? "  learn due" : ""))
  if (!f && !cal.n) {
    out.push("next:  aq forecast --lo <n> --hi <n> --why \"one sentence\"")
  }
  if (cal.n) out.push(`  coverage ${cal.coverage.toFixed(2)}  sharpness ${cal.score.toFixed(2)}  budget ${b.current}/${b.max}`)
  return out.join("\n")
}

export function parseLabFlags(argv: string[]): {
  rest: string[]
  force: boolean
  kill: boolean
  input: Partial<ForecastInput> & { name?: string }
} {
  const rest: string[] = []
  const input: Partial<ForecastInput> & { name?: string } = {}
  let force = false
  let kill = false
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!
    if (a === "--force") {
      force = true
      continue
    }
    if (a === "--kill") {
      kill = true
      continue
    }
    const take = (key: keyof ForecastInput | "name") => {
      const v = argv[i + 1]
      if (v === undefined) throw new Error(`missing value after ${a}`)
      i += 1
      return v
    }
    if (a === "--lo") {
      input.lo = Number(take("lo"))
      continue
    }
    if (a === "--hi") {
      input.hi = Number(take("hi"))
      continue
    }
    if (a === "--why") {
      input.why = take("why")
      continue
    }
    if (a === "--metric") {
      input.metric = take("metric")
      continue
    }
    if (a === "--novelty") {
      input.novelty = take("novelty")
      continue
    }
    if (a === "--n") {
      input.n = Number(take("n"))
      continue
    }
    if (a === "--note") {
      input.note = take("note")
      continue
    }
    if (a === "--name") {
      input.name = take("name")
      continue
    }
    rest.push(a)
  }
  return { rest, force, kill, input }
}

export function completeInput(input: Partial<ForecastInput>): ForecastInput {
  const lo = input.lo
  const hi = input.hi
  const why = input.why?.trim()
  if (lo == null || !Number.isFinite(lo) || hi == null || !Number.isFinite(hi) || !why) {
    throw new Error("need --lo --hi --why")
  }
  return {
    lo,
    hi,
    why,
    metric: input.metric,
    novelty: input.novelty,
    n: input.n,
    note: input.note,
  }
}

export const KILLER_PREAMBLE = [
  "Your job is the cheapest disproof, not agreement.",
  "Find the fastest way this fork is wrong: leak, tiny eval, broken split, recipe that cannot beat the parent, or a prediction interval too wide to be science.",
  "Kill bad forks quickly. Do not cheerlead. Do not propose a bigger model until you have a cheap counterexample on disk (eval, forecast miss, or a memory note).",
].join(" ")
