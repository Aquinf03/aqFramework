import { homedir } from "node:os"
import path from "node:path"

export function insideTrain(train: string, rel: string): string {
  const root = path.resolve(train)
  const dest = path.resolve(root, rel)
  const relTo = path.relative(root, dest)
  if (relTo.startsWith("..") || path.isAbsolute(relTo)) {
    throw new Error(`outside train: ${rel}`)
  }
  return dest
}

export function shortPath(p: string): string {
  const abs = path.resolve(p)
  const home = homedir()
  if (abs === home) return "~"
  if (abs.startsWith(home + path.sep)) return "~" + abs.slice(home.length)
  return abs
}
