import { existsSync, readdirSync, statSync } from "node:fs"
import path from "node:path"

/** A train is a directory. These two files are enough. The rest is optional. */

export const REQUIRED = ["instructions.md", "recipe.yaml"] as const

export const OPTIONAL_DIRS = [
  "data",
  "skills",
  "memory",
  "tools",
  "sandbox",
  "jobs",
  "evals",
  "artifacts",
  "connections",
  "schedules",
  "stages",
  "methods",
] as const

export const OPTIONAL_FILES = ["train.ts"] as const

export function isTrain(dir: string): boolean {
  return REQUIRED.every((name) => existsSync(path.join(dir, name)))
}

export function assertTrain(dir: string): string {
  const root = path.resolve(dir)
  if (!existsSync(root) || !statSync(root).isDirectory()) {
    throw new Error(`not a directory: ${root}`)
  }
  if (!isTrain(root)) {
    throw new Error(`not a train (need ${REQUIRED.join(" and ")}): ${root}`)
  }
  return root
}

/** Train folders directly under dir (relative paths). */
export function childTrains(dir: string): string[] {
  const root = path.resolve(dir)
  if (!existsSync(root) || !statSync(root).isDirectory()) return []
  const out: string[] = []
  for (const name of readdirSync(root)) {
    if (name.startsWith(".")) continue
    const p = path.join(root, name)
    try {
      if (statSync(p).isDirectory() && isTrain(p)) out.push(name)
    } catch {
      /* skip */
    }
  }
  return out.sort()
}

/** First argv token after verb that resolves to a train under cwd. */
export function trainInArgv(cwd: string, parts: string[]): string | null {
  const head = parts[0]
  if (!head) return null
  const tail = parts.slice(1)
  if (head === "job") {
    for (const t of tail) {
      if (t === "--") break
      if (t.startsWith("-")) continue
      const p = path.resolve(cwd, t)
      if (isTrain(p)) return p
    }
    return null
  }
  for (const t of tail) {
    if (t.startsWith("-")) continue
    const p = path.resolve(cwd, t)
    if (isTrain(p)) return p
    if (head === "eval" || head === "serve") break
  }
  return null
}
