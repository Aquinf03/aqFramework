import { spawnSync } from "node:child_process"
import { existsSync, readdirSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { pythonBin } from "./python.js"
import { assertTrain, isTrain } from "./schema.js"

const EXTS = [".py", ".ts", ".js", ".sh"]
const aqRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), "..")

function toolsDir(train: string): string {
  return path.join(train, "tools")
}

export function listTools(train: string): string[] {
  const dir = toolsDir(train)
  if (!existsSync(dir)) return []
  const names = new Set<string>()
  for (const f of readdirSync(dir)) {
    if (f.startsWith(".")) continue
    const ext = path.extname(f)
    if (!EXTS.includes(ext)) continue
    names.add(path.basename(f, ext))
  }
  return [...names].sort()
}

export function resolveTool(train: string, name: string): string {
  const base = path.join(toolsDir(train), name)
  for (const ext of EXTS) {
    const p = base + ext
    if (existsSync(p)) return p
  }
  throw new Error(`no tool ${name} in ${path.join(train, "tools")}`)
}

function parseToolArgs(argv: string[]): { train: string; name: string | null; extra: string[] } {
  const dd = argv.indexOf("--")
  const left = dd >= 0 ? argv.slice(0, dd) : argv
  const extra = dd >= 0 ? argv.slice(dd + 1) : []
  if (left.length === 0) return { train: assertTrain("."), name: null, extra }
  if (left.length === 1) {
    if (isTrain(path.resolve(left[0]))) return { train: assertTrain(left[0]), name: null, extra }
    return { train: assertTrain("."), name: left[0], extra }
  }
  if (left.length === 2) return { train: assertTrain(left[0]), name: left[1], extra }
  throw new Error("usage: aq tool [dir] [name] [-- args]")
}

function runFile(train: string, file: string, extra: string[]): void {
  const ext = path.extname(file)
  let cmd = file
  let args = extra
  if (ext === ".py") {
    cmd = pythonBin()
    args = [file, ...extra]
  } else if (ext === ".ts") {
    const tsxCli = path.join(aqRoot, "node_modules", "tsx", "dist", "cli.mjs")
    if (!existsSync(tsxCli)) throw new Error("tsx not found (need it to run .ts tools)")
    cmd = process.execPath
    args = [tsxCli, file, ...extra]
  } else if (ext === ".js") {
    cmd = process.execPath
    args = [file, ...extra]
  } else if (ext === ".sh") {
    cmd = "sh"
    args = [file, ...extra]
  }
  const r = spawnSync(cmd, args, {
    cwd: train,
    stdio: "inherit",
    env: { ...process.env, AQ_TRAIN: train },
  })
  if (r.status !== 0) process.exitCode = r.status ?? 1
}

export async function tool(argv: string[]): Promise<void> {
  const { train, name, extra } = parseToolArgs(argv)
  if (!name) {
    const names = listTools(train)
    if (!names.length) {
      console.log("no tools")
      return
    }
    for (const n of names) console.log(n)
    return
  }
  runFile(train, resolveTool(train, name), extra)
}

export function runFileCaptured(train: string, file: string, extra: string[] = []): string {
  const ext = path.extname(file)
  let cmd = file
  let args = extra
  if (ext === ".py") {
    cmd = pythonBin()
    args = [file, ...extra]
  } else if (ext === ".ts") {
    const tsxCli = path.join(aqRoot, "node_modules", "tsx", "dist", "cli.mjs")
    if (!existsSync(tsxCli)) throw new Error("tsx not found (need it to run .ts tools)")
    cmd = process.execPath
    args = [tsxCli, file, ...extra]
  } else if (ext === ".js") {
    cmd = process.execPath
    args = [file, ...extra]
  } else if (ext === ".sh") {
    cmd = "sh"
    args = [file, ...extra]
  } else {
    throw new Error(`cannot run ${ext}`)
  }
  const r = spawnSync(cmd, args, {
    cwd: train,
    encoding: "utf8",
    timeout: 120_000,
    env: { ...process.env, AQ_TRAIN: train },
  })
  const out = `${r.stdout ?? ""}${r.stderr ?? ""}`.trim()
  if (r.status !== 0) throw new Error(out || `exit ${r.status}`)
  return out || "ok"
}

export function runToolCaptured(train: string, name: string, extra: string[] = []): string {
  return runFileCaptured(train, resolveTool(train, name), extra)
}
