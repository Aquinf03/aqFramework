import { cp, mkdir, writeFile } from "node:fs/promises"
import { existsSync } from "node:fs"
import path from "node:path"
import { assertTrain } from "../core/schema.js"
import {
  completeInput,
  inputToForecast,
  learnDue,
  parseLabFlags,
  recordSearch,
  requireDiverseNovelty,
  spendForkBudget,
  taxBudget,
  writeForecast,
  type ForecastInput,
} from "../core/forecast.js"

/** Runtime output. Do not copy into the fork. */
export const FORK_SKIP = new Set(["jobs", "artifacts", "node_modules", ".git"])

export type ForkPlan = {
  src: string
  dest: string
  force: boolean
  forecast?: ForecastInput
}

export async function fork(plan: ForkPlan): Promise<{ src: string; dest: string; forecast: boolean }> {
  const src = assertTrain(plan.src)
  const dest = path.resolve(plan.dest)
  if (src === dest) {
    throw new Error("source and dest are the same")
  }
  if (existsSync(dest)) {
    throw new Error(`already exists: ${dest}`)
  }

  if (plan.forecast && !plan.force) {
    if (learnDue(src)) taxBudget(src)
    requireDiverseNovelty(src, plan.forecast.novelty || "hyperparam")
    spendForkBudget(src)
  }

  await mkdir(path.dirname(dest), { recursive: true })
  await cp(src, dest, {
    recursive: true,
    filter: (file) => {
      const rel = path.relative(src, file)
      if (!rel || rel === ".") return true
      return !rel.split(path.sep).some((p) => FORK_SKIP.has(p)) && path.basename(rel) !== "forecast.yaml"
    },
  })
  for (const name of ["jobs", "artifacts"] as const) {
    const dir = path.join(dest, name)
    await mkdir(dir, { recursive: true })
    await writeFile(path.join(dir, ".keep"), "", "utf8")
  }

  if (plan.forecast) {
    const parentRel = path.relative(dest, src) || src
    const f = inputToForecast(dest, plan.forecast, parentRel)
    writeForecast(dest, f)
    recordSearch(src, dest, f)
  }

  return { src, dest, forecast: Boolean(plan.forecast) }
}

/** `aq fork <name>` copies cwd. `aq fork <src> <name>` copies src. Flags: --lo --hi --why [--metric] [--novelty] [--force] */
export function parseForkArgs(argv: string[]): ForkPlan {
  const { rest, force, input } = parseLabFlags(argv)
  let src: string
  let dest: string
  if (rest.length === 1) {
    src = "."
    dest = rest[0]!
  } else if (rest.length === 2) {
    src = rest[0]!
    dest = rest[1]!
  } else {
    throw new Error(
      "usage: aq fork <new-dir> [--lo n --hi n --why \"...\" [--novelty method]]   or  aq fork <src> <new-dir> …",
    )
  }
  const has = input.lo != null || input.hi != null || Boolean(input.why)
  return {
    src,
    dest,
    force,
    forecast: has ? completeInput(input) : undefined,
  }
}
