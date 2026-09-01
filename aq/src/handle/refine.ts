import path from "node:path"
import { assertTrain } from "../core/schema.js"
import {
  formatRefineResult,
  formatTrajectory,
  listRefine,
  rollbackRefine,
  runRefine,
} from "../core/harness.js"

export async function refineCmd(argv: string[]): Promise<void> {
  const rest = [...argv]
  let dry = false
  let focus: string | undefined
  let skill = false
  const positional: string[] = []
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i]!
    if (a === "--dry") {
      dry = true
      continue
    }
    if (a === "--skill") {
      skill = true
      continue
    }
    if (a === "--focus") {
      focus = rest[++i]
      if (!focus) throw new Error("missing value after --focus")
      continue
    }
    positional.push(a)
  }

  const sub = positional[0]
  if (sub === "list") {
    const train = assertTrain(positional[1] ?? ".")
    const rows = listRefine(train)
    console.log("refine")
    if (!rows.length) console.log("  (none)")
    else {
      for (const r of rows.slice(-20)) {
        console.log(`  ${r.id}  ${r.at}  ${r.action}  ${r.path}`)
        console.log(`    ${r.summary}`)
      }
    }
    return
  }
  if (sub === "rollback") {
    const id = positional[1]
    if (!id) throw new Error("usage: aq refine rollback <id> [dir]")
    const train = assertTrain(positional[2] ?? ".")
    const out = rollbackRefine(train, id)
    console.log("rolled back")
    console.log("  " + out.trigger)
    console.log("  " + out.path)
    return
  }
  if (sub === "show" || sub === "trajectory") {
    const train = assertTrain(positional[1] ?? ".")
    console.log(formatTrajectory(train))
    return
  }

  const dir = positional[0] && !positional[0].startsWith("-") ? positional[0] : "."
  const train = assertTrain(dir)
  const r = runRefine(train, { focus, dry, applyFocusSkill: skill })
  console.log(formatRefineResult(r))
  if (r.ok && !dry) {
    const rel = path.relative(process.cwd(), train) || train
    console.log("  " + rel)
  }
}
