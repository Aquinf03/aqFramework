import { existsSync, statSync } from "node:fs"
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
