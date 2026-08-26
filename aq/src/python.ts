import { mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { spawnSync } from "node:child_process"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { assertTrain, isTrain } from "./schema.js"

const kernelRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "kernel")
const runPy = path.join(kernelRoot, "run.py")

export function pythonBin(): string {
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
}

export function runKernel(train: string, req: KernelReq): void {
  const art = path.join(train, "artifacts")
  mkdirSync(art, { recursive: true })
  writeFileSync(path.join(art, "request.json"), JSON.stringify(req, null, 2) + "\n")
  const r = spawnSync(pythonBin(), [runPy, train], {
    encoding: "utf8",
    cwd: kernelRoot,
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

export function kernelStep(step: string, argv: string[]): void {
  const ckptIdx = argv.indexOf("--ckpt")
  let ckpt: string | undefined
  let rest = argv
  if (ckptIdx >= 0) {
    ckpt = argv[ckptIdx + 1]
    rest = argv.filter((_, i) => i !== ckptIdx && i !== ckptIdx + 1)
  }
  const keepIdx = rest.indexOf("--keep")
  let keep: string | undefined
  if (keepIdx >= 0) {
    keep = rest[keepIdx + 1]
    rest = rest.filter((_, i) => i !== keepIdx && i !== keepIdx + 1)
  }
  let train: string
  let probe: string | undefined
  if (step === "eval" && rest.length === 2) {
    train = assertTrain(rest[0])
    probe = rest[1]
  } else if (step === "eval" && rest.length === 1 && !isTrain(path.resolve(rest[0]))) {
    train = assertTrain(".")
    probe = rest[0]
  } else if (rest.length > 1) {
    throw new Error(`usage: aq ${step} [dir]`)
  } else {
    train = assertTrain(rest[0] ?? ".")
  }
  const req: KernelReq = { op: step }
  if (ckpt) req.ckpt = ckpt
  if (keep) req.keep = keep
  if (probe) req.probe = probe
  runKernel(train, req)
}
