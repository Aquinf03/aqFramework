import { existsSync, readdirSync, renameSync, statSync } from "node:fs"
import path from "node:path"

/** A train is a directory. These two files are enough. The rest is optional. */

export const LEGACY_IDENTITY = "instructions.md" as const
export const REQUIRED = ["experiment.md", "recipe.yaml"] as const

export const OPTIONAL_DIRS = [
  "data",
  "skills",
  "tools",
  "jobs",
  "evals",
  "artifacts",
  "stages",
] as const

function hasRecipe(root: string): boolean {
  return existsSync(path.join(root, "recipe.yaml"))
}

function hasIdentity(root: string): boolean {
  return (
    existsSync(path.join(root, "experiment.md")) ||
    existsSync(path.join(root, LEGACY_IDENTITY))
  )
}

/** Rename legacy `instructions.md` to `experiment.md` when present. */
export function migrateLegacyIdentity(dir: string): void {
  const root = path.resolve(dir)
  const exp = path.join(root, "experiment.md")
  const leg = path.join(root, LEGACY_IDENTITY)
  if (!existsSync(exp) && existsSync(leg)) {
    renameSync(leg, exp)
  }
}

export function isTrain(dir: string): boolean {
  const root = path.resolve(dir)
  if (!existsSync(root) || !statSync(root).isDirectory()) return false
  return hasRecipe(root) && hasIdentity(root)
}

export function assertTrain(dir: string): string {
  const root = path.resolve(dir)
  if (!existsSync(root) || !statSync(root).isDirectory()) {
    throw new Error(`not a directory: ${root}`)
  }
  migrateLegacyIdentity(root)
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
