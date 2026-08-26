import { readdirSync, existsSync } from "node:fs"
import path from "node:path"
import { init } from "./init.js"
import { assertTrain, isTrain } from "../core/schema.js"
import { kernelStep } from "../core/python.js"

function stagesDir(train: string): string {
  return path.join(train, "stages")
}

export function listStages(train: string): string[] {
  const dir = stagesDir(train)
  if (!existsSync(dir)) return []
  const names: string[] = []
  for (const f of readdirSync(dir, { withFileTypes: true })) {
    if (!f.isDirectory() || f.name.startsWith(".")) continue
    if (isTrain(path.join(dir, f.name))) names.push(f.name)
  }
  return names.sort()
}

function stageTrain(train: string, name: string): string {
  const p = path.join(stagesDir(train), name)
  if (!isTrain(p)) throw new Error(`no stage ${name} (need a train under stages/${name})`)
  return p
}

export async function stage(argv: string[]): Promise<void> {
  const sub = argv[0]
  if (sub === "help" || sub === "-h") {
    console.log("aq stage\n")
    console.log("  aq stage [dir]              list nested trains")
    console.log("  aq stage init [dir] <name>  init stages/<name>")
    console.log("  aq stage [dir] <name>       train that stage")
    console.log("  aq stage eval [dir] <name>  eval that stage")
    return
  }

  if (sub === "init") {
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
      throw new Error("usage: aq stage init [dir] <name>")
    }
    if (name.includes("/") || name.startsWith(".")) throw new Error("bad stage name")
    const dest = path.join(stagesDir(train), name)
    const { created, skipped } = await init(dest)
    console.log("stage")
    console.log("  " + path.relative(process.cwd(), dest) || dest)
    for (const f of created) console.log("  + " + f)
    for (const f of skipped) console.log("  skip " + f)
    return
  }

  if (sub === "eval") {
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
      throw new Error("usage: aq stage eval [dir] <name>")
    }
    kernelStep("eval", [stageTrain(train, name)])
    return
  }

  if (!sub) {
    const train = assertTrain(".")
    const names = listStages(train)
    if (!names.length) {
      console.log("no stages")
      return
    }
    for (const n of names) console.log(n)
    return
  }

  if (isTrain(path.resolve(sub)) && argv.length === 1) {
    const names = listStages(assertTrain(sub))
    if (!names.length) {
      console.log("no stages")
      return
    }
    for (const n of names) console.log(n)
    return
  }

  let train: string
  let name: string
  if (argv.length === 1) {
    train = assertTrain(".")
    name = sub
  } else if (argv.length === 2) {
    train = assertTrain(sub)
    name = argv[1]
  } else {
    throw new Error("usage: aq stage [dir] [name]")
  }
  kernelStep("train", [stageTrain(train, name)])
}
