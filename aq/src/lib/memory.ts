/** Train memory: markdown files in memory/. Searchable. No embeddings. */

import { existsSync, readdirSync, readFileSync } from "node:fs"
import path from "node:path"
import { writeFileAt } from "./files.js"

const EXTS = [".md", ".txt"]

function dir(train: string): string {
  return path.join(train, "memory")
}

function safeName(name: string): string {
  const n = name.trim().replace(/\\/g, "/").split("/").pop() ?? ""
  const stem = n.replace(/\.(md|txt)$/i, "")
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/.test(stem)) {
    throw new Error(`bad memory name: ${name}`)
  }
  return stem
}

function resolveFile(train: string, name: string): string | null {
  const stem = safeName(name)
  const base = path.join(dir(train), stem)
  for (const ext of EXTS) {
    const p = base + ext
    if (existsSync(p)) return p
  }
  return null
}

export function listMemory(train: string): string[] {
  const d = dir(train)
  if (!existsSync(d)) return []
  const names = new Set<string>()
  for (const f of readdirSync(d)) {
    if (f.startsWith(".")) continue
    const ext = path.extname(f)
    if (!EXTS.includes(ext)) continue
    names.add(path.basename(f, ext))
  }
  return [...names].sort()
}

export function readMemory(train: string, name: string): string {
  const p = resolveFile(train, name)
  if (!p) throw new Error(`no memory ${name}`)
  return readFileSync(p, "utf8")
}

export function writeMemory(train: string, name: string, body: string): string {
  const stem = safeName(name)
  const text = body.trim() + (body.endsWith("\n") ? "" : "\n")
  writeFileAt(train, path.join("memory", stem + ".md"), text)
  return stem
}

function score(query: string, name: string, body: string): number {
  const q = query.toLowerCase().split(/\s+/).filter((w) => w.length > 1)
  if (!q.length) return 0
  const hay = (name + "\n" + body).toLowerCase()
  let n = 0
  for (const w of q) {
    if (name.toLowerCase() === w) n += 8
    else if (name.toLowerCase().includes(w)) n += 4
    if (hay.includes(w)) n += 1
  }
  return n
}

export function searchMemory(
  train: string,
  query: string,
  limit = 8,
): { name: string; score: number; snippet: string }[] {
  const hits: { name: string; score: number; snippet: string }[] = []
  for (const name of listMemory(train)) {
    const body = readMemory(train, name)
    const s = score(query, name, body)
    if (s <= 0) continue
    const snippet = body.trim().replace(/\s+/g, " ").slice(0, 180)
    hits.push({ name, score: s, snippet })
  }
  hits.sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
  return hits.slice(0, limit)
}

export function memoryDigest(train: string, maxChars = 1200): string {
  const names = listMemory(train)
  if (!names.length) return ""
  const parts = [`memory (${names.length}): ${names.join(", ")}`]
  let used = parts[0]!.length
  for (const name of names.slice(0, 6)) {
    const body = readMemory(train, name).trim().replace(/\s+/g, " ").slice(0, 160)
    const line = `- ${name}: ${body}`
    if (used + line.length > maxChars) break
    parts.push(line)
    used += line.length
  }
  return parts.join("\n")
}
