/** One-shot agent. No chat UI. stdout is the answer. */

import { createInterface } from "node:readline"
import { stdin, stdout, stderr } from "node:process"
import path from "node:path"
import { runTurn } from "./agent-loop.js"
import { renderMarkdown } from "./markdown.js"
import { isTrain } from "./schema.js"

function parseAsk(argv: string[]): { train: string; prompt: string; json: boolean; yes: boolean } {
  const flags = new Set<string>()
  const rest: string[] = []
  for (const a of argv) {
    if (a === "--json") flags.add("json")
    else if (a === "-y" || a === "--yes") flags.add("yes")
    else if (a.startsWith("-")) throw new Error(`unknown flag: ${a}\nusage: aq ask [-y] [--json] [dir] <prompt>`)
    else rest.push(a)
  }
  if (!rest.length) throw new Error("usage: aq ask [-y] [--json] [dir] <prompt>")
  let train = path.resolve(".")
  let words = rest
  if (rest.length >= 2 && isTrain(path.resolve(rest[0]!))) {
    train = path.resolve(rest[0]!)
    words = rest.slice(1)
  }
  const prompt = words.join(" ").trim()
  if (!prompt) throw new Error("usage: aq ask [-y] [--json] [dir] <prompt>")
  return { train, prompt, json: flags.has("json"), yes: flags.has("yes") }
}

function askYesNo(cmd: string): Promise<boolean> {
  if (stdin.isTTY !== true) return Promise.resolve(false)
  stderr.write(`run  ${cmd}\n  yes / no? `)
  return new Promise((resolve) => {
    const rl = createInterface({ input: stdin, output: stderr })
    rl.question("", (line) => {
      rl.close()
      const s = line.trim().toLowerCase()
      resolve(s === "y" || s === "yes")
    })
  })
}

export async function ask(argv: string[]): Promise<void> {
  const { train, prompt, json, yes } = parseAsk(argv)
  const tools: { name: string; args: string }[] = []
  const text = await runTurn(
    train,
    [{ role: "user", content: prompt }],
    () => {},
    (name, args) => {
      tools.push({ name, args: args ?? "" })
      if (!json) stderr.write(`${name}\n`)
    },
    async (command) => {
      if (yes) return true
      return askYesNo(command)
    },
  )
  const answer = text.trim()
  if (json) {
    stdout.write(JSON.stringify({ text: answer, tools }) + "\n")
    return
  }
  stdout.write(renderMarkdown(answer) + "\n")
}
