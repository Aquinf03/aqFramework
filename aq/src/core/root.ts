/** aq package root and bundled kernel path. Safe from any file under src/. */

import { existsSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

export function aqRoot(): string {
  let d = path.dirname(fileURLToPath(import.meta.url))
  for (let i = 0; i < 8; i++) {
    if (existsSync(path.join(d, "package.json")) && existsSync(path.join(d, "bin"))) return d
    d = path.dirname(d)
  }
  throw new Error("aq package root not found")
}

export function kernelRoot(): string {
  return path.join(aqRoot(), "kernel")
}
