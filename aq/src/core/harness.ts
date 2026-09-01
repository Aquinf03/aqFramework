/** Continual harness for trains. Trajectory → smallest durable edit. Base prompt stays immutable. */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { randomBytes } from "node:crypto"
import path from "node:path"
import {
  ackLearn,
  calibrationScore,
  labDigest,
  learnDue,
  readBudget,
  readCalibration,
  readSearch,
  type CalEvent,
} from "./forecast.js"
import { listMemory, readMemory, writeMemory } from "../lib/memory.js"
import { listSkills } from "../lib/skill.js"
import { isTrain } from "./schema.js"

export type HarnessKind = "memory" | "skill" | "prompt"

export type RefineRecord = {
  id: string
  at: string
  kind: HarnessKind
  path: string
  action: "create" | "append" | "update"
  trigger: string
  summary: string
  previous?: string
  focus?: string
}

export type TrajectorySnap = {
  calibration: ReturnType<typeof calibrationScore>
  budget: ReturnType<typeof readBudget>
  forks: number
  misses: CalEvent[]
  hits: CalEvent[]
  critique: string | null
  heuristics: string | null
  memory: string[]
  skills: string[]
  learnDue: boolean
  lab: string
}

function art(train: string, name: string): string {
  return path.join(train, "artifacts", name)
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

function safeStem(name: string): string {
  const n = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48)
  return n || "note"
}

function readOptional(train: string, rel: string): string | null {
  const p = path.join(train, rel)
  if (!existsSync(p)) return null
  try {
    return readFileSync(p, "utf8")
  } catch {
    return null
  }
}

export function readTrajectory(train: string): TrajectorySnap {
  const cal = readCalibration(train)
  const misses = cal.filter((e) => !e.hit).slice(-8)
  const hits = cal.filter((e) => e.hit).slice(-8)
  let critique: string | null = null
  const critPath = art(train, "eval-critique.json")
  if (existsSync(critPath)) {
    try {
      const c = JSON.parse(readFileSync(critPath, "utf8")) as { note?: string }
      critique = c.note ?? null
    } catch {
      critique = null
    }
  }
  return {
    calibration: calibrationScore(cal),
    budget: readBudget(train),
    forks: readSearch(train).length,
    misses,
    hits,
    critique,
    heuristics: readOptional(train, "memory/heuristics.md"),
    memory: listMemory(train),
    skills: listSkills(train),
    learnDue: learnDue(train),
    lab: labDigest(train),
  }
}

export function formatTrajectory(train: string): string {
  const t = readTrajectory(train)
  const lines = ["trajectory"]
  lines.push(
    `  calibration ${t.calibration.hits}/${t.calibration.n}  sharpness ${t.calibration.n ? t.calibration.score.toFixed(2) : "—"}  budget ${t.budget.current}/${t.budget.max}`,
  )
  lines.push(`  forks ${t.forks}${t.learnDue ? "  heuristics overdue" : ""}`)
  if (t.critique) lines.push(`  critique  ${t.critique}`)
  if (t.misses.length) {
    lines.push("  recent misses")
    for (const m of t.misses.slice(-5)) {
      lines.push(`    ${m.metric} predicted ${m.lo}–${m.hi} got ${m.actual}  ${m.why}`)
    }
  }
  if (t.hits.length) {
    lines.push("  recent hits")
    for (const h of t.hits.slice(-3)) {
      lines.push(`    ${h.metric} ${h.lo}–${h.hi} ≈ ${h.actual}`)
    }
  }
  lines.push(`  memory  ${t.memory.length ? t.memory.join(" ") : "(none)"}`)
  lines.push(`  skills  ${t.skills.length ? t.skills.join(" ") : "(none)"}`)
  if (t.heuristics) {
    const snip = t.heuristics.trim().replace(/\s+/g, " ").slice(0, 200)
    lines.push(`  heuristics  ${snip}`)
  }
  return lines.join("\n")
}

export type ProposedEdit = {
  kind: HarnessKind
  path: string
  action: "create" | "append" | "update"
  trigger: string
  summary: string
  body: string
}

function alreadyHas(hay: string | null, needle: string): boolean {
  if (!hay) return false
  const a = hay.toLowerCase()
  const b = needle.toLowerCase().slice(0, 80)
  return a.includes(b)
}

/** Smallest evidence-backed harness edits from disk trajectory (+ optional focus). */
export function proposeRefine(train: string, focus?: string): ProposedEdit[] {
  const t = readTrajectory(train)
  const edits: ProposedEdit[] = []
  const focusText = focus?.trim()

  if (focusText) {
    const bullet = `- ${focusText}`
    if (!alreadyHas(t.heuristics, focusText)) {
      edits.push({
        kind: "memory",
        path: "memory/heuristics.md",
        action: t.heuristics ? "append" : "create",
        trigger: "focus",
        summary: "record human/agent focus as a durable heuristic",
        body: t.heuristics
          ? `${t.heuristics.trimEnd()}\n\n${bullet}\n`
          : `# Heuristics\n\nLessons from this train. Smallest useful rules only.\n\n${bullet}\n`,
      })
    }
  }

  if (t.misses.length >= 2) {
    const last = t.misses[t.misses.length - 1]!
    const rule = `Intervals missed often — tighten why and avoid stadium-wide bands (last miss: predicted ${last.lo}–${last.hi}, got ${last.actual}).`
    if (!alreadyHas(t.heuristics, "stadium-wide") && !alreadyHas(t.heuristics, "Intervals missed")) {
      edits.push({
        kind: "memory",
        path: "memory/heuristics.md",
        action: t.heuristics || edits.some((e) => e.path === "memory/heuristics.md") ? "append" : "create",
        trigger: "calibration-miss",
        summary: "compress repeated forecast misses",
        body: rule,
      })
    }
  }

  if (t.critique && /low trust|only \d+ eval rows/i.test(t.critique)) {
    const rule = `Eval is thin: ${t.critique}`
    if (!alreadyHas(t.heuristics, "Eval is thin") && !alreadyHas(t.heuristics, "ignore deltas")) {
      edits.push({
        kind: "memory",
        path: "memory/heuristics.md",
        action: "append",
        trigger: "eval-critique",
        summary: "don't treat noisy tiny-eval deltas as science",
        body: rule,
      })
    }
  }

  const search = readSearch(train)
  const streak = search.slice(-5)
  if (streak.length >= 5 && streak.every((s) => s.novelty === "hyperparam")) {
    const rule =
      "Last five forks were hyperparam-only — change method, data, or architecture before grinding lr again."
    if (!alreadyHas(t.heuristics, "hyperparam-only")) {
      edits.push({
        kind: "memory",
        path: "memory/heuristics.md",
        action: "append",
        trigger: "novelty-streak",
        summary: "force structural diversity after hyperparam grind",
        body: rule,
      })
    }
  }

  if (t.calibration.n >= 3 && t.calibration.coverage < 0.4 && t.budget.current <= 2) {
    const note =
      "Calibration coverage is low and budget is tight — next bets must be narrower or skip forecast-forks until a hit."
    const prev = readOptional(train, "memory/prompt-notes.md")
    if (!alreadyHas(prev, "budget is tight")) {
      edits.push({
        kind: "prompt",
        path: "memory/prompt-notes.md",
        action: prev ? "append" : "create",
        trigger: "budget-pressure",
        summary: "steer future turns toward honest intervals",
        body: prev
          ? `${prev.trimEnd()}\n\n- ${note}\n`
          : `# Prompt notes\n\nAgent-editable layer. Base system prompt is immutable.\n\n- ${note}\n`,
      })
    }
  }

  // Merge multiple appends to the same heuristics file into one edit when creating fresh.
  return coalesceHeuristics(edits, t.heuristics)
}

function coalesceHeuristics(edits: ProposedEdit[], existing: string | null): ProposedEdit[] {
  const heur = edits.filter((e) => e.path === "memory/heuristics.md")
  const other = edits.filter((e) => e.path !== "memory/heuristics.md")
  if (heur.length <= 1) return edits

  const bullets: string[] = []
  const seen = new Set<string>()
  const pushBullet = (line: string) => {
    const t = line.trim()
    if (!t || t.startsWith("#")) return
    const b = t.startsWith("- ") ? t : `- ${t}`
    const key = b.toLowerCase()
    if (seen.has(key)) return
    seen.add(key)
    bullets.push(b)
  }
  for (const e of heur) {
    for (const line of e.body.split("\n")) {
      const t = line.trim()
      if (t.startsWith("- ")) pushBullet(t)
      else if (t && !t.startsWith("#") && !e.body.trimStart().startsWith("#")) pushBullet(t)
    }
    // single-rule body with no leading dash
    if (!e.body.includes("\n") && !e.body.trim().startsWith("#") && !e.body.trim().startsWith("- ")) {
      pushBullet(e.body)
    }
  }
  if (!bullets.length) return other

  const trigger = [...new Set(heur.map((e) => e.trigger))].join("+")
  const summary = [...new Set(heur.map((e) => e.summary))].join("; ")
  if (existing) {
    return [
      ...other,
      {
        kind: "memory",
        path: "memory/heuristics.md",
        action: "append",
        trigger,
        summary,
        body: `${existing.trimEnd()}\n\n${bullets.join("\n")}\n`,
      },
    ]
  }
  return [
    ...other,
    {
      kind: "memory",
      path: "memory/heuristics.md",
      action: "create",
      trigger,
      summary,
      body: `# Heuristics\n\nLessons from this train. Smallest useful rules only.\n\n${bullets.join("\n")}\n`,
    },
  ]
}

function applyBody(train: string, edit: ProposedEdit): { previous?: string } {
  const abs = path.join(train, edit.path)
  const previous = existsSync(abs) ? readFileSync(abs, "utf8") : undefined
  mkdirSync(path.dirname(abs), { recursive: true })
  if (edit.action === "create" || edit.action === "update") {
    writeFileSync(abs, edit.body.endsWith("\n") ? edit.body : edit.body + "\n")
    return { previous }
  }
  // append: body may already be full file content (from coalesce) or a single rule
  if (edit.body.includes("\n# ") || edit.body.startsWith("# ") || (previous && edit.body.startsWith(previous.slice(0, 20)))) {
    writeFileSync(abs, edit.body.endsWith("\n") ? edit.body : edit.body + "\n")
  } else {
    const base = previous?.trimEnd() ?? ""
    const add = edit.body.trim().startsWith("-") ? edit.body.trim() : `- ${edit.body.trim()}`
    const next = base ? `${base}\n\n${add}\n` : `# Heuristics\n\n${add}\n`
    writeFileSync(abs, next)
  }
  return { previous }
}

export type RefineResult = {
  ok: boolean
  dry: boolean
  edits: RefineRecord[]
  skip?: string
  trajectory: string
}

export function runRefine(
  train: string,
  opts?: { focus?: string; dry?: boolean; applyFocusSkill?: boolean },
): RefineResult {
  if (!isTrain(train)) throw new Error("not a train")
  const focus = opts?.focus?.trim()
  const dry = opts?.dry === true
  const proposed = proposeRefine(train, focus)

  if (focus && opts?.applyFocusSkill && focus.length > 24) {
    const stem = safeStem(focus.split(/[.!?\n]/)[0] ?? "tactic")
    const rel = `skills/${stem}.md`
    if (!existsSync(path.join(train, rel))) {
      proposed.push({
        kind: "skill",
        path: rel,
        action: "create",
        trigger: "focus-skill",
        summary: "promote focus into a reusable skill note",
        body: `# ${stem}\n\n${focus.trim()}\n`,
      })
    }
  }

  if (!proposed.length) {
    return {
      ok: false,
      dry,
      edits: [],
      skip: "nothing new to write — trajectory already reflected in memory/skills",
      trajectory: formatTrajectory(train),
    }
  }

  const applied: RefineRecord[] = []
  for (const edit of proposed) {
    const id = randomBytes(3).toString("hex")
    const at = new Date().toISOString()
    let previous: string | undefined
    if (!dry) {
      ;({ previous } = applyBody(train, edit))
    }
    const rec: RefineRecord = {
      id,
      at,
      kind: edit.kind,
      path: edit.path,
      action: edit.action,
      trigger: edit.trigger,
      summary: edit.summary,
      previous,
      focus: focus || undefined,
    }
    if (!dry) appendJsonl(art(train, "refine.jsonl"), rec)
    applied.push(rec)
  }

  if (!dry && existsSync(path.join(train, "memory", "heuristics.md"))) {
    try {
      ackLearn(train)
    } catch {
      /* heuristics may still be thin; ok */
    }
  }

  return { ok: true, dry, edits: applied, trajectory: formatTrajectory(train) }
}

export function listRefine(train: string): RefineRecord[] {
  return readJsonl<RefineRecord>(art(train, "refine.jsonl"))
}

export function rollbackRefine(train: string, id: string): RefineRecord {
  const rows = listRefine(train)
  const rec = rows.find((r) => r.id === id)
  if (!rec) throw new Error(`no refine ${id}`)
  const abs = path.join(train, rec.path)
  if (rec.previous != null) {
    mkdirSync(path.dirname(abs), { recursive: true })
    writeFileSync(abs, rec.previous)
  } else if (existsSync(abs) && (rec.action === "create" || rec.action === "update")) {
    writeFileSync(abs, "")
  }
  const undo: RefineRecord = {
    id: randomBytes(3).toString("hex"),
    at: new Date().toISOString(),
    kind: rec.kind,
    path: rec.path,
    action: "update",
    trigger: `rollback:${rec.id}`,
    summary: `rollback ${rec.id}`,
    previous: existsSync(abs) ? readFileSync(abs, "utf8") : undefined,
  }
  appendJsonl(art(train, "refine.jsonl"), undo)
  return undo
}

/** CRUD helpers for the agent (same surface, disk is source of truth). */
export function harnessCreate(
  train: string,
  kind: HarnessKind,
  name: string,
  body: string,
): { path: string } {
  if (kind === "memory") {
    const stem = writeMemory(train, name, body)
    return { path: `memory/${stem}.md` }
  }
  if (kind === "prompt") {
    const rel = "memory/prompt-notes.md"
    const prev = readOptional(train, rel)
    const text = body.trim()
    const next = prev ? `${prev.trimEnd()}\n\n- ${text}\n` : `# Prompt notes\n\n- ${text}\n`
    mkdirSync(path.join(train, "memory"), { recursive: true })
    writeFileSync(path.join(train, rel), next)
    return { path: rel }
  }
  const stem = safeStem(name)
  const rel = `skills/${stem}.md`
  mkdirSync(path.join(train, "skills"), { recursive: true })
  writeFileSync(path.join(train, rel), body.trim() + (body.endsWith("\n") ? "" : "\n"))
  return { path: rel }
}

export function harnessGet(train: string, kind: HarnessKind, name?: string): string {
  if (kind === "memory") {
    if (!name) return listMemory(train).join("\n") || "(no memory)"
    return readMemory(train, name)
  }
  if (kind === "prompt") {
    return readOptional(train, "memory/prompt-notes.md") || "(no prompt notes)"
  }
  if (!name) return listSkills(train).join("\n") || "(no skills)"
  const p = path.join(train, "skills", name.endsWith(".md") ? name : `${name}.md`)
  if (!existsSync(p)) throw new Error(`no skill ${name}`)
  return readFileSync(p, "utf8")
}

export function harnessDigest(train: string): string {
  if (!isTrain(train)) return ""
  const rows = listRefine(train)
  const notes = readOptional(train, "memory/prompt-notes.md")
  const lines: string[] = []
  if (notes) {
    const snip = notes.trim().replace(/\s+/g, " ").slice(0, 220)
    lines.push(`prompt notes: ${snip}`)
  }
  if (rows.length) {
    const last = rows[rows.length - 1]!
    lines.push(`last refine: ${last.id}  ${last.path}  ${last.summary}`)
  }
  return lines.join("\n")
}

export function formatRefineResult(r: RefineResult): string {
  const lines = [r.dry ? "refine (dry)" : "refine"]
  if (r.skip) {
    lines.push("  " + r.skip)
    return lines.join("\n")
  }
  for (const e of r.edits) {
    lines.push(`  ${e.id}  ${e.action}  ${e.path}`)
    lines.push(`    ${e.summary}`)
  }
  lines.push("  artifacts/refine.jsonl")
  return lines.join("\n")
}
