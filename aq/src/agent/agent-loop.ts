/** One agent turn: model, tools, until the request is done or blocked. */

import {
  AGENT_TOOLS,
  contextBlock,
  parseRunCommand,
  runAgentTool,
  toolsForTrain,
  runDetached,
  runShell,
  type AgentToolDef,
} from "./agent-tools.js"
import { systemPrompt } from "./prompt.js"
import { streamTurn, type ChatMsg } from "./provider.js"

export { type ChatMsg }

const MAX_ROUNDS = 16

function lastUserRequest(history: ChatMsg[]): string {
  for (let i = history.length - 1; i >= 0; i--) {
    const m = history[i]!
    if (m.role !== "user") continue
    const c = m.content.trim()
    if (!c) continue
    if (c.startsWith("Earlier in this chat")) continue
    if (c.startsWith("Answer the user now")) continue
    if (c.startsWith("Recover:")) continue
    if (c.startsWith("Stop calling tools.")) continue
    return c.length > 2000 ? c.slice(0, 2000) + "…" : c
  }
  return ""
}

function clip(s: string, n = 240): string {
  const t = s.replace(/\s+/g, " ").trim()
  return t.length > n ? t.slice(0, n) + "…" : t
}

function isRecoverable(result: string): boolean {
  const m = result.toLowerCase()
  return m.includes("not a train") || m.includes("need instructions.md") || m.includes("outside train")
}

export async function runTurn(
  train: string,
  history: ChatMsg[],
  onDelta: (chunk: string) => void,
  onTool?: (name: string, args?: string) => void,
  onPermit?: (command: string) => Promise<boolean>,
): Promise<string> {
  const objective = lastUserRequest(history)
  const msgs: ChatMsg[] = history.map((m) => ({ ...m }))
  const progress: string[] = []
  let usedTools = false
  let lastFailed = false
  let recovered = false

  const system = () =>
    systemPrompt(train, {
      extra: contextBlock(train),
      objective,
      progress: progress.length ? progress.map((l, i) => `${i + 1}. ${l}`).join("\n") : "",
    })

  for (let i = 0; i < MAX_ROUNDS; i++) {
    const out = await streamTurn(msgs, onDelta, { system: system(), tools: toolsForTrain(train) })
    if (out.toolCalls.length) {
      usedTools = true
      msgs.push({
        role: "assistant",
        content: out.text,
        tool_calls: out.toolCalls,
      })
      lastFailed = false
      for (const call of out.toolCalls) {
        let result: string
        let failed = false
        try {
          if (call.name === "run") {
            const spec = parseRunCommand(call.args)
            const label = spec.detach ? `${spec.command}  [detach]` : spec.command
            const ok = onPermit ? await onPermit(label) : false
            if (!ok) {
              result = "denied by user"
              failed = true
            } else {
              onTool?.("run", spec.detach ? `${spec.command} detach=true` : spec.command)
              result = spec.detach
                ? await runDetached(train, spec.command)
                : runShell(train, spec.command)
            }
          } else {
            onTool?.(call.name, call.args)
            result = await runAgentTool(train, call.name, call.args)
          }
        } catch (err) {
          failed = true
          result = err instanceof Error ? err.message : String(err)
        }
        lastFailed = lastFailed || (failed && isRecoverable(result))
        progress.push(
          failed
            ? `${call.name} failed: ${clip(result)}`
            : `${call.name} ${clip(call.args, 80)} → ${clip(result)}`,
        )
        msgs.push({ role: "tool", content: result, tool_call_id: call.id })
      }
      continue
    }
    if (!out.text.trim() && usedTools) {
      msgs.push({
        role: "user",
        content:
          "Answer the user now in a few sentences. Cite files as markdown links like [recipe.yaml](recipe.yaml). Do not invent scores or pass/fail. Do not only list filenames. Do not claim a train is missing if this turn created one.",
      })
      continue
    }
    if (out.text.trim() && lastFailed && !recovered) {
      recovered = true
      msgs.push({
        role: "assistant",
        content: out.text,
      })
      msgs.push({
        role: "user",
        content:
          "Recover: that last tool failed. Retry once with the train folder in args if cwd is not a train, or tell the user the exact command using the folder already in this turn's progress. Do not say nothing was built if progress shows init or writes.",
      })
      continue
    }
    return out.text
  }
  const wrap = await streamTurn(
    [
      ...msgs,
      {
        role: "user",
        content:
          "Stop calling tools. Tell the user what you already did this turn, the train folder if any, and the exact next command. Do not claim nothing was created if progress shows otherwise.",
      },
    ],
    onDelta,
    { system: system(), tools: [] },
  )
  return wrap.text.trim() || "too many tool rounds — say the train folder and `aq train <folder>` if you created one."
}

export function defs(): AgentToolDef[] {
  return AGENT_TOOLS
}
