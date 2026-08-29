import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { spawnSync } from "node:child_process"
import path from "node:path"
import { assertTrain, isTrain } from "./schema.js"
import { kernelRoot } from "./root.js"

const kernelDir = kernelRoot()
const runPy = path.join(kernelDir, "run.py")

export function pythonBin(): string {
  const venvPy = path.join(kernelRoot(), ".venv", "bin", "python")
  if (existsSync(venvPy)) return venvPy
  for (const bin of ["python3", "python"]) {
    const r = spawnSync(bin, ["-c", "import sys; print(sys.executable)"], {
      encoding: "utf8",
      timeout: 3000,
    })
    if (r.status === 0) return bin
  }
  throw new Error("python3 not found")
}

export type KernelReq = {
  op: string
  snapshot?: boolean
  ckpt?: string
  keep?: string
  probe?: string
  prompt?: string
  max_tokens?: number
  temperature?: number
}

export function runKernel(train: string, req: KernelReq): void {
  const art = path.join(train, "artifacts")
  mkdirSync(art, { recursive: true })
  writeFileSync(path.join(art, "request.json"), JSON.stringify(req, null, 2) + "\n")
  const r = spawnSync(pythonBin(), [runPy, train], {
    encoding: "utf8",
    cwd: kernelDir,
    // live kernel progress (metrics steps) must stream; result still in result.json
    stdio: ["ignore", "inherit", "inherit"],
  })
  const resultPath = path.join(art, "result.json")
  let result: { ok?: boolean; lines?: string[]; error?: string }
  try {
    result = JSON.parse(readFileSync(resultPath, "utf8"))
  } catch {
    throw new Error((r.stderr || r.stdout || "kernel failed").trim())
  }
  if (!result.ok) {
    throw new Error(result.error || "kernel failed")
  }
  const lines = result.lines ?? []
  process.stdout.write(lines.join("\n") + (lines.length ? "\n" : ""))
}

function popFlag(rest: string[], flag: string): { value?: string; rest: string[] } {
  const idx = rest.indexOf(flag)
  if (idx < 0) return { rest }
  const value = rest[idx + 1]
  if (value === undefined) throw new Error(`usage: missing value after ${flag}`)
  return {
    value,
    rest: rest.filter((_, i) => i !== idx && i !== idx + 1),
  }
}

export function kernelStep(step: string, argv: string[]): void {
  let rest = argv
  const ck = popFlag(rest, "--ckpt")
  rest = ck.rest
  const keep = popFlag(rest, "--keep")
  rest = keep.rest
  const mt = popFlag(rest, "--max-tokens")
  rest = mt.rest
  const temp = popFlag(rest, "--temperature")
  rest = temp.rest

  let train: string
  let probe: string | undefined
  let prompt: string | undefined
  if (step === "eval" && rest.length === 2) {
    train = assertTrain(rest[0])
    probe = rest[1]
  } else if (step === "eval" && rest.length === 1 && !isTrain(path.resolve(rest[0]))) {
    train = assertTrain(".")
    probe = rest[0]
  } else if (step === "serve") {
    if (rest.length === 2) {
      train = assertTrain(rest[0])
      prompt = rest[1]
    } else if (rest.length === 1 && !isTrain(path.resolve(rest[0]))) {
      train = assertTrain(".")
      prompt = rest[0]
    } else if (rest.length === 1) {
      train = assertTrain(rest[0])
    } else if (rest.length === 0) {
      train = assertTrain(".")
    } else {
      throw new Error("usage: aq serve [dir] [prompt] [--ckpt name] [--max-tokens n] [--temperature t]")
    }
  } else if (rest.length > 1) {
    throw new Error(`usage: aq ${step} [dir]`)
  } else {
    train = assertTrain(rest[0] ?? ".")
  }
  const req: KernelReq = { op: step }
  if (ck.value) req.ckpt = ck.value
  if (keep.value) req.keep = keep.value
  if (probe) req.probe = probe
  if (prompt) req.prompt = prompt
  if (mt.value !== undefined) req.max_tokens = Number(mt.value)
  if (temp.value !== undefined) req.temperature = Number(temp.value)
  runKernel(train, req)
}
