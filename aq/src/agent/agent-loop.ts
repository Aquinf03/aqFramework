/** One agent turn: model, tools, until text. */

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

export async function runTurn(
  train: string,
  history: ChatMsg[],
  onDelta: (chunk: string) => void,
  onTool?: (name: string, args?: string) => void,
  onPermit?: (command: string) => Promise<boolean>,
): Promise<string> {
  const system = systemPrompt(train, contextBlock(train))
  const msgs: ChatMsg[] = history.map((m) => ({ ...m }))
  let usedTools = false
  for (let i = 0; i < 8; i++) {
    const out = await streamTurn(msgs, onDelta, { system, tools: toolsForTrain(train) })
    if (out.toolCalls.length) {
      usedTools = true
      msgs.push({
        role: "assistant",
        content: out.text,
        tool_calls: out.toolCalls,
      })
      for (const call of out.toolCalls) {
        let result: string
        try {
          if (call.name === "run") {
            const spec = parseRunCommand(call.args)
            const label = spec.detach ? `${spec.command}  [detach]` : spec.command
            const ok = onPermit ? await onPermit(label) : false
            if (!ok) result = "denied by user"
            else {
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
          result = err instanceof Error ? err.message : String(err)
        }
        msgs.push({ role: "tool", content: result, tool_call_id: call.id })
      }
      continue
    }
    if (!out.text.trim() && usedTools) {
      msgs.push({
        role: "user",
        content: "Answer the user now in a few sentences. Do not only list filenames.",
      })
      continue
    }
    return out.text
  }
  throw new Error("too many tool rounds")
}

export function defs(): AgentToolDef[] {
  return AGENT_TOOLS
}
