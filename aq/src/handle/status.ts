import { existsSync, readdirSync, readFileSync } from "node:fs"
import path from "node:path"
import { assertTrain } from "../core/schema.js"

export async function status(argv: string[]): Promise<void> {
  const train = assertTrain(argv[0] ?? ".")
  const jobs = path.join(train, "jobs")
  const rows: string[] = []
  if (existsSync(jobs)) {
    for (const id of readdirSync(jobs).sort()) {
      if (id.startsWith(".")) continue
      const specPath = path.join(jobs, id, "spec.json")
      if (!existsSync(specPath)) continue
      const spec = JSON.parse(readFileSync(specPath, "utf8")) as {
        status?: string
        command?: string[]
      }
      rows.push(id + "  " + (spec.status ?? "?") + "  " + (spec.command ?? []).join(" "))
    }
  }
  console.log("jobs")
  if (!rows.length) console.log("  (none)")
  else for (const r of rows) console.log("  " + r)

  const last = path.join(train, "artifacts", "runs", "last.json")
  console.log("run")
  if (existsSync(last)) {
    const run = JSON.parse(readFileSync(last, "utf8")) as {
      id?: string
      pass?: boolean | null
      metrics?: { metric?: string; score?: number }
    }
    const p = run.pass === true ? "pass" : run.pass === false ? "fail" : "skip"
    console.log("  " + (run.id ?? "last") + "  " + p)
    if (run.metrics?.metric != null) {
      console.log("  " + run.metrics.metric + "  " + String(run.metrics.score ?? ""))
    }
  } else console.log("  (none)")

  const inspect = path.join(train, "artifacts", "inspect.md")
  console.log("inspect")
  if (existsSync(inspect)) console.log("  artifacts/inspect.md")
  else console.log("  (none)")

  const ev = path.join(train, "artifacts", "eval.json")
  console.log("eval")
  if (existsSync(ev)) {
    const e = JSON.parse(readFileSync(ev, "utf8")) as {
      metric?: string
      score?: number
      pass?: boolean | null
    }
    const p = e.pass === true ? "pass" : e.pass === false ? "fail" : "skip"
    console.log("  " + (e.metric ?? "") + "  " + String(e.score ?? "") + "  " + p)
  } else console.log("  (none)")

  const sv = path.join(train, "artifacts", "serve.json")
  console.log("serve")
  if (existsSync(sv)) {
    const s = JSON.parse(readFileSync(sv, "utf8")) as {
      text?: string
      tokens?: number
      checkpoint?: string
    }
    console.log("  " + String(s.text ?? ""))
    console.log("  tokens: " + String(s.tokens ?? ""))
    if (s.checkpoint) console.log("  " + s.checkpoint)
  } else console.log("  (none)")

  const metrics = path.join(train, "artifacts", "metrics.jsonl")
  console.log("metrics")
  if (existsSync(metrics)) {
    const lines = readFileSync(metrics, "utf8").trim().split("\n").filter(Boolean)
    console.log("  artifacts/metrics.jsonl  (" + lines.length + " events)")
    for (const line of lines.slice(-5)) {
      try {
        const row = JSON.parse(line) as {
          event?: string
          step?: number
          loss?: number
          score?: number
          metric?: string
          elapsed_ms?: number
        }
        const bits = [row.event ?? "?"]
        if (row.step != null) bits.push("step " + row.step)
        if (row.loss != null) bits.push("loss " + row.loss)
        if (row.metric != null) bits.push(row.metric + " " + String(row.score ?? ""))
        if (row.elapsed_ms != null) bits.push(row.elapsed_ms + "ms")
        console.log("  " + bits.join("  "))
      } catch {
        console.log("  " + line.slice(0, 80))
      }
    }
  } else console.log("  (none)")

  const slog = path.join(train, "artifacts", "schedules")
  console.log("schedules")
  if (existsSync(slog)) {
    const logs = readdirSync(slog).filter((f) => f.endsWith(".log"))
    if (!logs.length) console.log("  (none)")
    else for (const f of logs) console.log("  artifacts/schedules/" + f)
  } else console.log("  (none)")
}
