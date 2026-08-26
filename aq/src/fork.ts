import { cp, mkdir, writeFile } from "node:fs/promises"
import { existsSync } from "node:fs"
import path from "node:path"
import { assertTrain } from "./schema.js"

/** Runtime output. Do not copy into the fork. */
export const FORK_SKIP = new Set(["jobs", "artifacts", "node_modules", ".git"])

export async function fork(srcArg: string, destArg: string): Promise<{ src: string; dest: string }> {
  const src = assertTrain(srcArg)
  const dest = path.resolve(destArg)
  if (src === dest) {
    throw new Error("source and dest are the same")
  }
  if (existsSync(dest)) {
    throw new Error(`already exists: ${dest}`)
  }

  await mkdir(path.dirname(dest), { recursive: true })
  await cp(src, dest, {
    recursive: true,
    filter: (file) => {
      const rel = path.relative(src, file)
      if (!rel || rel === ".") return true
      return !rel.split(path.sep).some((p) => FORK_SKIP.has(p))
    },
  })
  for (const name of ["jobs", "artifacts"] as const) {
    const dir = path.join(dest, name)
    await mkdir(dir, { recursive: true })
    await writeFile(path.join(dir, ".keep"), "", "utf8")
  }

  return { src, dest }
}

/** `aq fork <name>` copies cwd. `aq fork <src> <name>` copies src. */
export function parseForkArgs(argv: string[]): { src: string; dest: string } {
  if (argv.length === 1) return { src: ".", dest: argv[0] }
  if (argv.length === 2) return { src: argv[0], dest: argv[1] }
  throw new Error("usage: aq fork <new-dir>   or  aq fork <src> <new-dir>")
}
