import path from "node:path"
import { assertTrain } from "../core/schema.js"
import {
  ackLearn,
  completeInput,
  formatLab,
  inputToForecast,
  parseLabFlags,
  writeForecast,
} from "../core/forecast.js"

export async function forecastCmd(argv: string[]): Promise<void> {
  const { rest, input } = parseLabFlags(argv)
  const dir = rest[0] ?? "."
  const train = assertTrain(dir)
  const hasRange = input.lo != null || input.hi != null || Boolean(input.why)
  if (!hasRange) {
    console.log(formatLab(train))
    return
  }
  const f = inputToForecast(train, completeInput(input))
  writeForecast(train, f)
  const rel = path.relative(process.cwd(), train) || train
  console.log("forecast")
  console.log("  " + rel)
  console.log("  " + f.metric + "  " + f.lo + "–" + f.hi)
  console.log("  " + f.why)
  if (f.eval_note) console.log("  " + f.eval_note)
  console.log("  forecast.yaml")
}

export async function learnCmd(argv: string[]): Promise<void> {
  const dir = argv.find((a) => a && !a.startsWith("-")) ?? "."
  const train = assertTrain(dir)
  const out = ackLearn(train)
  console.log("learn")
  console.log("  " + out.path)
  console.log("  forks " + out.forks)
}
