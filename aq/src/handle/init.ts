import { mkdir, readFile, writeFile } from "node:fs/promises"
import { existsSync } from "node:fs"
import path from "node:path"
import { OPTIONAL_DIRS, OPTIONAL_FILES, REQUIRED } from "../core/schema.js"
import { aqRoot } from "../core/root.js"

const templates = path.join(aqRoot(), "templates")

export type InitResult = {
  root: string
  created: string[]
  skipped: string[]
}

async function writeNew(dest: string, body: string, created: string[], skipped: string[]): Promise<void> {
  const rel = path.relative(process.cwd(), dest) || dest
  if (existsSync(dest)) {
    skipped.push(rel)
    return
  }
  await mkdir(path.dirname(dest), { recursive: true })
  await writeFile(dest, body, "utf8")
  created.push(rel)
}

function assertNotCliHome(root: string): void {
  const pkg = path.join(root, "package.json")
  const cli = path.join(root, "src", "cli.ts")
  if (!existsSync(pkg) || !existsSync(cli)) return
  throw new Error("this is the aq CLI package. init a train folder: npm run aq -- init ../my-train")
}

export async function init(dir: string): Promise<InitResult> {
  const root = path.resolve(dir)
  await mkdir(root, { recursive: true })
  assertNotCliHome(root)

  const created: string[] = []
  const skipped: string[] = []

  for (const name of REQUIRED) {
    const dest = path.join(root, name)
    const body = await readFile(path.join(templates, name), "utf8")
    await writeNew(dest, body, created, skipped)
  }

  for (const name of OPTIONAL_FILES) {
    const dest = path.join(root, name)
    const src = path.join(templates, name)
    const body = existsSync(src)
      ? await readFile(src, "utf8")
      : "// optional runtime for this train. aq does not read this yet.\n"
    await writeNew(dest, body, created, skipped)
  }

  for (const name of OPTIONAL_DIRS) {
    const dest = path.join(root, name)
    const keep = path.join(dest, ".keep")
    if (existsSync(dest)) {
      skipped.push(path.relative(process.cwd(), dest) || dest)
      continue
    }
    await mkdir(dest, { recursive: true })
    await writeFile(keep, "", "utf8")
    created.push(path.relative(process.cwd(), dest) || dest)
  }

  return { root, created, skipped }
}
