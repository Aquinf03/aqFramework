#!/usr/bin/env node

import path from "node:path"
import { checkout, parseCheckoutArgs } from "./checkout.js"
import { fork, parseForkArgs } from "./fork.js"
import { init } from "./init.js"
import { diffRuns } from "./diff.js"
import { data } from "./data.js"
import { job } from "./job.js"
import { checkpoint, evalCmd, train } from "./step.js"
import { schedule } from "./schedule.js"
import { status } from "./status.js"
import { stage } from "./stage.js"
import { tool } from "./tool.js"
import { runAgent } from "./agent.js"
import { ask } from "./ask.js"
import { chatCmd } from "./chat.js"
import { providerCmd } from "./provider.js"
import { doctorCmd } from "./doctor.js"
import { spawnCmd } from "./spawn.js"

async function main(): Promise<void> {
  const argv = process.argv.slice(2)
  const cmd = argv[0]
  if (!cmd || cmd === "agent" || cmd === "help" || cmd === "-h" || cmd === "--help") {
    await runAgent(argv, process.cwd())
    return
  }

  if (cmd === "init") {
    const dir = argv[1] ?? "."
    const { created, skipped } = await init(dir)
    if (created.length) {
      console.log("created")
      for (const f of created) console.log("  " + f)
    }
    if (skipped.length) {
      console.log("already there")
      for (const f of skipped) console.log("  " + f)
    }
    if (!created.length && skipped.length) {
      console.log("skeleton already complete. grow it by editing files.")
    }
    return
  }

  if (cmd === "fork") {
    const { src, dest } = parseForkArgs(argv.slice(1))
    const out = await fork(src, dest)
    const relSrc = path.relative(process.cwd(), out.src) || out.src
    const relDest = path.relative(process.cwd(), out.dest) || out.dest
    console.log("forked")
    console.log("  " + relSrc)
    console.log("  -> " + relDest)
    return
  }

  if (cmd === "job") {
    await job(argv.slice(1))
    return
  }

  if (cmd === "data") {
    await data(argv.slice(1))
    return
  }

  if (cmd === "train") {
    await train(argv.slice(1))
    return
  }

  if (cmd === "eval") {
    await evalCmd(argv.slice(1))
    return
  }

  if (cmd === "checkpoint") {
    await checkpoint(argv.slice(1))
    return
  }

  if (cmd === "tool") {
    await tool(argv.slice(1))
    return
  }

  if (cmd === "schedule") {
    await schedule(argv.slice(1))
    return
  }

  if (cmd === "stage") {
    await stage(argv.slice(1))
    return
  }

  if (cmd === "diff") {
    await diffRuns(argv.slice(1))
    return
  }

  if (cmd === "status") {
    await status(argv.slice(1))
    return
  }

  if (cmd === "doctor") {
    await doctorCmd(argv.slice(1))
    return
  }

  if (cmd === "spawn") {
    await spawnCmd(argv.slice(1))
    return
  }

  if (cmd === "provider") {
    await providerCmd(argv.slice(1))
    return
  }

  if (cmd === "ask") {
    await ask(argv.slice(1))
    return
  }

  if (cmd === "chat") {
    await chatCmd(argv.slice(1))
    return
  }

  if (cmd === "checkout") {
    const { dir, id, dest } = parseCheckoutArgs(argv.slice(1))
    const out = await checkout(dir, id, dest)
    const rel = path.relative(process.cwd(), out.dest) || out.dest
    console.log("checked out")
    console.log("  " + out.id)
    console.log("  -> " + rel)
    return
  }

  console.error(`unknown command: ${cmd}`)
  console.error("aq help")
  process.exitCode = 1
}

main().catch((err: unknown) => {
  const msg = err instanceof Error ? err.message : String(err)
  console.error(msg)
  process.exitCode = 1
})
