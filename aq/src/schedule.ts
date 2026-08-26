import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { fork } from "./fork.js"
import { enqueueJob, waitForJob } from "./job.js"
import { startAgent } from "./spawn.js"
import { assertTrain, isTrain } from "./schema.js"

const aqRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), "..")

export type Sched = {
  kind: "sweep" | "cron" | "resume" | "pipeline" | "agents"
  run?: string
  ask?: string
  steps?: string[]
  agents?: number
  n?: number
  every?: number
  on?: string
  max?: number
}

type State = { last?: string; count: number; retries: number }

function parseSched(text: string): Sched {
  const rec: Record<string, string> = {}
  for (const line of text.split("\n")) {
    const s = line.split("#", 1)[0].trim()
    if (!s || !s.includes(":")) continue
    const [k, ...rest] = s.split(":")
    rec[k.trim()] = rest.join(":").trim()
  }
  const kind = rec.kind as Sched["kind"]
  if (kind !== "sweep" && kind !== "cron" && kind !== "resume" && kind !== "pipeline" && kind !== "agents") {
    throw new Error("schedule: kind must be sweep, cron, resume, pipeline, or agents")
  }
  const steps = rec.steps
    ? rec.steps.split(",").map((s) => s.trim()).filter(Boolean)
    : undefined
  if (kind === "agents") {
    if (!rec.ask && !rec.run) throw new Error("schedule: agents need ask")
  } else if (kind !== "pipeline" && !rec.run) throw new Error("schedule: need run")
  if (kind === "pipeline" && !steps?.length) throw new Error("schedule: pipeline needs steps")
  const n = rec.n != null ? Number(rec.n) : undefined
  const every = rec.every != null ? Number(rec.every) : undefined
  const max = rec.max != null ? Number(rec.max) : undefined
  const agents = rec.agents != null ? Number(rec.agents) : undefined
  return { kind, run: rec.run, ask: rec.ask, steps, agents, n, every, on: rec.on, max }
}

function listFiles(train: string): { name: string; file: string }[] {
  const dir = path.join(train, "schedules")
  if (!existsSync(dir)) return []
  const out: { name: string; file: string }[] = []
  for (const f of readdirSync(dir)) {
    if (f.startsWith(".")) continue
    if (!f.endsWith(".yaml") && !f.endsWith(".yml") && !f.endsWith(".json")) continue
    const name = f.replace(/\.(yaml|yml|json)$/, "")
    out.push({ name, file: path.join(dir, f) })
  }
  return out.sort((a, b) => a.name.localeCompare(b.name))
}

function readSched(file: string): Sched {
  const text = readFileSync(file, "utf8")
  if (file.endsWith(".json")) {
    const j = JSON.parse(text) as Sched
    if (!j.kind) throw new Error("schedule: need kind")
    if (j.kind === "pipeline") {
      if (!j.steps?.length) throw new Error("schedule: pipeline needs steps")
    } else if (j.kind === "agents") {
      if (!j.ask && !j.run) throw new Error("schedule: agents need ask")
    } else if (!j.run) throw new Error("schedule: need run")
    return j
  }
  return parseSched(text)
}

function statePath(train: string, name: string): string {
  return path.join(train, "artifacts", "schedules", name + ".json")
}

function loadState(train: string, name: string): State {
  const p = statePath(train, name)
  if (!existsSync(p)) return { count: 0, retries: 0 }
  return JSON.parse(readFileSync(p, "utf8")) as State
}

function saveState(train: string, name: string, st: State): void {
  const p = statePath(train, name)
  mkdirSync(path.dirname(p), { recursive: true })
  writeFileSync(p, JSON.stringify(st, null, 2) + "\n")
}

function aqArgv(args: string[]): string[] {
  const tsx = path.join(aqRoot, "node_modules", "tsx", "dist", "cli.mjs")
  const cli = path.join(aqRoot, "src", "cli.ts")
  return [process.execPath, tsx, cli, ...args]
}

function runCmd(train: string, run: string): string[] {
  if (run.startsWith("tool:")) {
    return aqArgv(["tool", train, run.slice(5)])
  }
  if (run.startsWith("stage:")) {
    return aqArgv(["stage", train, run.slice(6)])
  }
  if (run === "train" || run === "eval" || run === "hash") {
    const cmd = run === "hash" ? ["data", "hash", train] : [run, train]
    return aqArgv(cmd)
  }
  throw new Error(`schedule: unknown run ${run} (train, eval, hash, tool:name, stage:name)`)
}

function logPath(train: string, name: string): string {
  return path.join(train, "artifacts", "schedules", name + ".log")
}

function logLine(train: string, name: string, line: string): void {
  const p = logPath(train, name)
  mkdirSync(path.dirname(p), { recursive: true })
  writeFileSync(p, new Date().toISOString() + "  " + line + "\n", { flag: "a" })
}

async function runPipeline(train: string, name: string, spec: Sched): Promise<string[]> {
  const steps = spec.steps ?? []
  const lines: string[] = []
  for (const step of steps) {
    const id = await enqueueJob(train, runCmd(train, step))
    logLine(train, name, "start  " + step + "  " + id)
    const done = await waitForJob(train, id)
    const line = step + "  " + id + "  " + done.status + (done.code != null ? "  " + done.code : "")
    logLine(train, name, line)
    lines.push(name + "  " + line)
    if (done.status !== "exited" || (done.code != null && done.code !== 0)) {
      lines.push(name + "  stop")
      break
    }
  }
  const st = loadState(train, name)
  st.last = new Date().toISOString()
  st.count += 1
  saveState(train, name, st)
  return lines
}

async function runAgents(train: string, name: string, spec: Sched): Promise<string[]> {
  const n = spec.agents ?? 1
  if (n <= 1) return runPipeline(train, name, spec)
  const lines: string[] = []
  const kids: Promise<string[]>[] = []
  for (let i = 1; i <= n; i++) {
    const dest = path.join(train, "artifacts", "agents", String(i))
    if (!existsSync(dest)) await fork(train, dest)
    kids.push(runPipeline(dest, name, spec).then((ls) => ls.map((l) => "agent" + i + "  " + l)))
  }
  for (const ls of await Promise.all(kids)) lines.push(...ls)
  return lines
}

async function fire(train: string, name: string, spec: Sched): Promise<string> {
  if (!spec.run) throw new Error("schedule: need run")
  const id = await enqueueJob(train, runCmd(train, spec.run))
  const st = loadState(train, name)
  st.last = new Date().toISOString()
  st.count += 1
  saveState(train, name, st)
  return id
}

function label(spec: Sched): string {
  if (spec.kind === "agents") return (spec.ask ?? spec.run ?? "") + (spec.n || spec.agents ? `  x${spec.n ?? spec.agents}` : "")
  if (spec.steps?.length) return spec.steps.join(",")
  return spec.run ?? ""
}

function dueCron(spec: Sched, st: State): boolean {
  const every = spec.every ?? 60
  if (!st.last) return true
  return Date.now() - new Date(st.last).getTime() >= every * 60 * 1000
}

function evalFailed(train: string): boolean {
  const p = path.join(train, "artifacts", "eval.json")
  if (!existsSync(p)) return false
  const ev = JSON.parse(readFileSync(p, "utf8")) as { pass?: boolean | null }
  return ev.pass === false
}

async function tickOne(
  train: string,
  name: string,
  spec: Sched,
  force: boolean,
): Promise<string[]> {
  const lines: string[] = []
  const st = loadState(train, name)
  if (spec.kind === "sweep") {
    const n = spec.n ?? 1
    for (let i = 0; i < n; i++) {
      const id = await fire(train, name, spec)
      lines.push(name + "  sweep  " + id)
    }
    return lines
  }
  if (spec.kind === "cron") {
    if (!force && !dueCron(spec, st)) {
      lines.push(name + "  cron  skip")
      return lines
    }
    const id = await fire(train, name, spec)
    lines.push(name + "  cron  " + id)
    return lines
  }
  if (spec.kind === "resume") {
    const max = spec.max ?? 3
    if (!evalFailed(train)) {
      st.retries = 0
      saveState(train, name, st)
      lines.push(name + "  resume  skip")
      return lines
    }
    if (st.retries >= max) {
      lines.push(name + "  resume  max")
      return lines
    }
    const id = await fire(train, name, spec)
    const st2 = loadState(train, name)
    st2.retries = st.retries + 1
    saveState(train, name, st2)
    lines.push(name + "  resume  " + id)
    return lines
  }
  if (spec.kind === "pipeline") {
    return runAgents(train, name, spec)
  }
  if (spec.kind === "agents") {
    const prompt = spec.ask ?? spec.run ?? ""
    const n = spec.n ?? spec.agents ?? 1
    for (let i = 1; i <= n; i++) {
      const a = await startAgent(train, prompt, { name: `${name}-${i}` })
      logLine(train, name, `spawn  ${a.id}  ${a.jobId}`)
      lines.push(`${name}  spawn  ${a.id}  ${a.status}`)
    }
    const st2 = loadState(train, name)
    st2.last = new Date().toISOString()
    st2.count += 1
    saveState(train, name, st2)
    return lines
  }
  return lines
}

export async function schedule(argv: string[]): Promise<void> {
  const sub = argv[0]
  if (sub === "help" || sub === "-h") {
    console.log("aq schedule\n")
    console.log("  aq schedule [dir]           list")
    console.log("  aq schedule tick [dir]     run due cron/resume")
    console.log("  aq schedule run [dir] <name>  fire that file now")
    return
  }
  if (!sub) {
    const train = assertTrain(".")
    const files = listFiles(train)
    if (!files.length) {
      console.log("no schedules")
      return
    }
    for (const { name, file } of files) {
      const spec = readSched(file)
      console.log(name + "  " + spec.kind + "  " + label(spec))
    }
    return
  }
  if (sub === "tick") {
    const train = assertTrain(argv[1] ?? ".")
    const files = listFiles(train)
    if (!files.length) {
      console.log("no schedules")
      return
    }
    for (const { name, file } of files) {
      const spec = readSched(file)
      if (spec.kind === "sweep" || spec.kind === "pipeline") continue
      const lines = await tickOne(train, name, spec, false)
      for (const l of lines) console.log(l)
    }
    return
  }
  if (sub === "run") {
    const rest = argv.slice(1)
    let train: string
    let name: string
    if (rest.length === 1) {
      train = assertTrain(".")
      name = rest[0]
    } else if (rest.length === 2) {
      train = assertTrain(rest[0])
      name = rest[1]
    } else {
      throw new Error("usage: aq schedule run [dir] <name>")
    }
    const hit = listFiles(train).find((x) => x.name === name)
    if (!hit) throw new Error(`no schedule ${name}`)
    const spec = readSched(hit.file)
    const lines = await tickOne(train, name, spec, true)
    for (const l of lines) console.log(l)
    return
  }
  if (isTrain(path.resolve(sub)) && argv.length === 1) {
    const train = assertTrain(sub)
    const files = listFiles(train)
    if (!files.length) {
      console.log("no schedules")
      return
    }
    for (const { name, file } of files) {
      const spec = readSched(file)
      console.log(name + "  " + spec.kind + "  " + label(spec))
    }
    return
  }
  const train = assertTrain(".")
  if (sub !== "list") {
    throw new Error("usage: aq schedule [dir] | tick | run <name>")
  }
  const files = listFiles(train)
  if (!files.length) {
    console.log("no schedules")
    return
  }
  for (const { name, file } of files) {
    const spec = readSched(file)
    console.log(name + "  " + spec.kind + "  " + label(spec))
  }
}
