import { spawnSync } from "node:child_process"
import { randomBytes } from "node:crypto"
import { existsSync } from "node:fs"
import { cp, mkdir, readdir, readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { FORK_SKIP } from "../handle/fork.js"
import { clampAsk, detectHost, parseBytes, type HostResources, type ResourceAsk } from "./resources.js"
import { assertTrain, isTrain } from "../core/schema.js"

export type JobStatus = "queued" | "starting" | "running" | "exited" | "canceled" | "error"

export type JobSpec = {
  id: string
  command: string[]
  cwd: string
  pid: number | null
  waiterPid?: number | null
  status: JobStatus
  code: number | null
  signal: string | null
  queuedAt: string
  started: string
  ended: string | null
  host?: HostResources
  resources?: ResourceAsk
}

const waiterPath = fileURLToPath(new URL("./job-wait.mjs", import.meta.url))

const RUN_USAGE =
  "usage: aq job run [dir] [--cpu N] [--ram SIZE] [--disk SIZE] [--gpu N] -- <cmd> [args...]"
const LOG_USAGE = "usage: aq job log [dir] <id>"
const CANCEL_USAGE = "usage: aq job cancel [dir] <id>"
const RESUME_USAGE = "usage: aq job resume [dir] <id>"
const TREE_USAGE = "usage: aq job tree [dir] <id>"

export function jobHelp(): string {
  return [
    "  aq job run [dir] [--cpu N] [--ram SIZE] [--disk SIZE] [--gpu N] -- <cmd>",
    "  aq job list [dir]            jobs in this train",
    "  aq job log [dir] <id>       print jobs/<id>/log",
    "  aq job cancel [dir] <id>    stop a running or queued job",
    "  aq job resume [dir] <id>    queue that job again (same id)",
    "  aq job tree [dir] <id>     process tree for that job",
  ].join("\n")
}

export async function job(argv: string[]): Promise<void> {
  const sub = argv[0]
  if (!sub || sub === "help" || sub === "-h" || sub === "--help") {
    console.log("aq job\n")
    console.log(jobHelp())
    return
  }
  if (sub === "run") return run(argv.slice(1))
  if (sub === "list") return list(argv.slice(1))
  if (sub === "log") return logCmd(argv.slice(1))
  if (sub === "cancel") return cancel(argv.slice(1))
  if (sub === "resume") return resume(argv.slice(1))
  if (sub === "tree") return tree(argv.slice(1))
  throw new Error(`unknown job command: ${sub}\n${jobHelp()}`)
}

function jobsRoot(train: string): string {
  return path.join(train, "jobs")
}

function specPath(train: string, id: string): string {
  return path.join(jobsRoot(train), id, "spec.json")
}

function logPath(train: string, id: string): string {
  return path.join(jobsRoot(train), id, "log")
}

function alive(pid: number | null): boolean {
  if (pid == null) return false
  try {
    process.kill(pid, 0)
    return true
  } catch {
    return false
  }
}

async function readSpec(train: string, id: string): Promise<JobSpec> {
  const p = specPath(train, id)
  if (!existsSync(p)) throw new Error(`no such job: ${id}`)
  return JSON.parse(await readFile(p, "utf8")) as JobSpec
}

async function writeSpec(train: string, spec: JobSpec): Promise<void> {
  await writeFile(specPath(train, spec.id), JSON.stringify(spec, null, 2) + "\n", "utf8")
}

async function refresh(train: string, spec: JobSpec): Promise<JobSpec> {
  if (spec.status === "running" && !alive(spec.pid)) {
    spec.status = "exited"
    spec.ended = spec.ended ?? new Date().toISOString()
    await writeSpec(train, spec)
  }
  if (
    spec.status === "starting" &&
    !alive(spec.pid) &&
    !alive(spec.waiterPid ?? null)
  ) {
    spec.status = "error"
    spec.ended = spec.ended ?? new Date().toISOString()
    await writeSpec(train, spec)
  }
  return spec
}

async function syncJobs(train: string): Promise<void> {
  const ids = await listIds(train)
  for (const id of ids) {
    try {
      await refresh(train, await readSpec(train, id))
    } catch {
      continue
    }
  }
  pump(train)
}

function pump(train: string): void {
  const r = spawnSync(process.execPath, [waiterPath, "--dispatch", jobsRoot(train)], {
    cwd: train,
    stdio: "ignore",
    env: process.env,
  })
  if (r.error) throw r.error
  if (r.status !== 0) throw new Error("queue dispatch failed")
}

function printJob(verb: string, id: string): void {
  console.log(verb)
  console.log("  " + id)
}

async function listIds(train: string): Promise<string[]> {
  const root = jobsRoot(train)
  if (!existsSync(root)) return []
  const entries = await readdir(root, { withFileTypes: true })
  return entries
    .filter((e) => e.isDirectory() && !e.name.startsWith("."))
    .map((e) => e.name)
    .sort()
}

export function jobHome(train: string, id: string): string {
  return path.join(jobsRoot(train), id)
}

export async function resolveJobId(train: string, id: string): Promise<string> {
  const ids = await listIds(train)
  if (ids.includes(id)) return id
  const hits = ids.filter((n) => n.startsWith(id))
  if (hits.length === 1) return hits[0]
  if (hits.length > 1) throw new Error(`ambiguous job id: ${id}`)
  throw new Error(`no such job: ${id}`)
}

function parseDirId(argv: string[], usage: string): { dir: string; id: string } {
  if (argv.length === 1) {
    if (isTrain(path.resolve(argv[0]))) throw new Error(usage)
    return { dir: ".", id: argv[0] }
  }
  if (argv.length === 2) return { dir: argv[0], id: argv[1] }
  throw new Error(usage)
}

function parseIntArg(name: string, v: string, min: number): number {
  if (!/^\d+$/.test(v)) throw new Error(`bad ${name}: ${v}`)
  const n = Number(v)
  if (n < min) throw new Error(`bad ${name}: ${v}`)
  return n
}

function parseRun(argv: string[]): { dir: string; command: string[]; ask: ResourceAsk } {
  const ask: ResourceAsk = {}
  let dir = "."
  let i = 0
  if (argv[0] && !argv[0].startsWith("-") && isTrain(path.resolve(argv[0]))) {
    dir = argv[0]
    i = 1
  }
  while (i < argv.length) {
    const a = argv[i]
    if (a === "--") {
      const command = argv.slice(i + 1)
      if (!command.length) throw new Error(RUN_USAGE)
      return { dir, command, ask }
    }
    if (a === "--cpu" || a === "--ram" || a === "--disk" || a === "--gpu") {
      const v = argv[i + 1]
      if (v == null || v.startsWith("-")) throw new Error(RUN_USAGE)
      if (a === "--cpu") ask.cpu = parseIntArg("cpu", v, 1)
      if (a === "--ram") ask.ram = parseBytes(v)
      if (a === "--disk") ask.disk = parseBytes(v)
      if (a === "--gpu") ask.gpu = parseIntArg("gpu", v, 0)
      i += 2
      continue
    }
    if (a.startsWith("-")) throw new Error(RUN_USAGE)
    const command = argv.slice(i)
    if (!command.length) throw new Error(RUN_USAGE)
    return { dir, command, ask }
  }
  throw new Error(RUN_USAGE)
}

async function captureTree(train: string, dest: string): Promise<void> {
  await mkdir(dest, { recursive: true })
  const entries = await readdir(train, { withFileTypes: true })
  for (const e of entries) {
    if (FORK_SKIP.has(e.name)) continue
    await cp(path.join(train, e.name), path.join(dest, e.name), { recursive: true })
  }
}

async function newId(train: string): Promise<string> {
  for (let i = 0; i < 8; i++) {
    const id = randomBytes(4).toString("hex")
    if (!existsSync(path.join(jobsRoot(train), id))) return id
  }
  throw new Error("could not allocate a job id")
}

export async function enqueueJob(
  train: string,
  command: string[],
  ask?: ResourceAsk,
): Promise<string> {
  const host = detectHost(train)
  const resources = ask && Object.keys(ask).length ? clampAsk(host, ask) : undefined
  const id = await newId(train)
  const home = path.join(jobsRoot(train), id)
  await mkdir(home, { recursive: true })
  await writeFile(path.join(home, "log"), "", "utf8")
  await captureTree(train, path.join(home, "tree"))
  const now = new Date().toISOString()
  const spec: JobSpec = {
    id,
    command,
    cwd: train,
    pid: null,
    status: "queued",
    code: null,
    signal: null,
    queuedAt: now,
    started: now,
    ended: null,
    host,
    resources,
  }
  await writeSpec(train, spec)
  pump(train)
  return id
}

async function run(argv: string[]): Promise<void> {
  const { dir, command, ask } = parseRun(argv)
  const train = assertTrain(dir)
  const id = await enqueueJob(train, command, ask)
  const live = await waitForPid(train, id)
  if (live.status === "error") throw new Error(`job failed to start: ${id}`)
  if (live.status === "queued") printJob("queued", id)
  else printJob("started", id)
}

export async function waitForJob(train: string, id: string): Promise<JobSpec> {
  for (;;) {
    pump(train)
    const spec = await refresh(train, await readSpec(train, id))
    if (spec.status === "exited" || spec.status === "error" || spec.status === "canceled") {
      return spec
    }
    await new Promise((r) => setTimeout(r, 300))
  }
}

export async function waitForPid(train: string, id: string, timeoutMs = 2000): Promise<JobSpec> {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    const spec = await readSpec(train, id)
    if (spec.status === "queued") return spec
    if (spec.pid != null || spec.status === "running" || spec.status === "error") return spec
    if (spec.status !== "starting") return spec
    await new Promise((r) => setTimeout(r, 20))
  }
  return readSpec(train, id)
}

async function list(argv: string[]): Promise<void> {
  if (argv.length > 1) throw new Error("usage: aq job list [dir]")
  const train = assertTrain(argv[0] ?? ".")
  await syncJobs(train)
  const ids = await listIds(train)
  for (const id of ids) {
    let spec: JobSpec
    try {
      spec = await refresh(train, await readSpec(train, id))
    } catch {
      continue
    }
    const cmd = spec.command.join(" ")
    const extra =
      spec.status === "exited" && spec.code != null
        ? String(spec.code)
        : spec.status === "canceled"
          ? spec.signal ?? ""
          : ""
    console.log([id, spec.status, extra, cmd].filter((s) => s !== "").join("  "))
  }
}

async function logCmd(argv: string[]): Promise<void> {
  const { dir, id: raw } = parseDirId(argv, LOG_USAGE)
  const train = assertTrain(dir)
  const id = await resolveJobId(train, raw)
  const p = logPath(train, id)
  if (!existsSync(p)) throw new Error(`no log: ${id}`)
  process.stdout.write(await readFile(p, "utf8"))
}

export function jobLogPath(train: string, id: string): string {
  return logPath(train, id)
}

export async function cancelJob(train: string, id: string): Promise<JobSpec> {
  const spec = await refresh(train, await readSpec(train, id))
  if (spec.status === "queued" || spec.status === "starting") {
    if (spec.pid != null && alive(spec.pid)) {
      try {
        process.kill(-spec.pid, "SIGTERM")
      } catch {
        process.kill(spec.pid, "SIGTERM")
      }
    }
    spec.status = "canceled"
    spec.ended = new Date().toISOString()
    await writeSpec(train, spec)
    pump(train)
    return spec
  }
  if (spec.status !== "running" || !alive(spec.pid)) {
    throw new Error(`not running: ${id} (${spec.status})`)
  }
  const pid = spec.pid as number
  try {
    process.kill(-pid, "SIGTERM")
  } catch {
    process.kill(pid, "SIGTERM")
  }
  spec.status = "canceled"
  spec.signal = "SIGTERM"
  spec.ended = new Date().toISOString()
  await writeSpec(train, spec)
  return spec
}

async function cancel(argv: string[]): Promise<void> {
  const { dir, id: raw } = parseDirId(argv, CANCEL_USAGE)
  const train = assertTrain(dir)
  const id = await resolveJobId(train, raw)
  await cancelJob(train, id)
  printJob("canceled", id)
}

async function resume(argv: string[]): Promise<void> {
  const { dir, id: raw } = parseDirId(argv, RESUME_USAGE)
  const train = assertTrain(dir)
  const id = await resolveJobId(train, raw)
  const spec = await refresh(train, await readSpec(train, id))
  if (spec.status === "running" || spec.status === "starting") {
    throw new Error(`already running: ${id}`)
  }
  if (spec.status === "queued") {
    throw new Error(`already queued: ${id}`)
  }
  spec.status = "queued"
  spec.pid = null
  spec.code = null
  spec.signal = null
  spec.ended = null
  spec.queuedAt = new Date().toISOString()
  await writeSpec(train, spec)
  pump(train)
  const live = await waitForPid(train, id)
  if (live.status === "queued") printJob("queued", id)
  else printJob("started", id)
}

type Proc = { pid: number; ppid: number; cmd: string }

function processSnapshot(): Proc[] {
  const r = spawnSync("ps", ["-axo", "pid=,ppid=,command="], {
    encoding: "utf8",
    timeout: 3000,
  })
  if (r.error) throw r.error
  if (r.status !== 0) throw new Error("ps failed")
  const out: Proc[] = []
  for (const line of (r.stdout ?? "").split("\n")) {
    const m = line.trim().match(/^(\d+)\s+(\d+)\s+(.*)$/)
    if (!m) continue
    out.push({ pid: Number(m[1]), ppid: Number(m[2]), cmd: m[3] })
  }
  return out
}

function printBranch(byPpid: Map<number, Proc[]>, proc: Proc, prefix: string, last: boolean): void {
  const branch = last ? "└─ " : "├─ "
  console.log(prefix + branch + proc.pid + "  " + proc.cmd)
  const kids = byPpid.get(proc.pid) ?? []
  const next = prefix + (last ? "   " : "│  ")
  kids.forEach((k, i) => printBranch(byPpid, k, next, i === kids.length - 1))
}

async function tree(argv: string[]): Promise<void> {
  const { dir, id: raw } = parseDirId(argv, TREE_USAGE)
  const train = assertTrain(dir)
  const id = await resolveJobId(train, raw)
  const spec = await refresh(train, await readSpec(train, id))
  if (spec.pid == null || !alive(spec.pid)) {
    throw new Error(`not running: ${id} (${spec.status})`)
  }
  const procs = processSnapshot()
  const byPid = new Map(procs.map((p) => [p.pid, p]))
  const byPpid = new Map<number, Proc[]>()
  for (const p of procs) {
    const list = byPpid.get(p.ppid)
    if (list) list.push(p)
    else byPpid.set(p.ppid, [p])
  }
  const root = byPid.get(spec.pid)
  if (!root) throw new Error(`not running: ${id} (${spec.status})`)
  console.log(id)
  printBranch(byPpid, root, "", true)
}
