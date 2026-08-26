/** Chats live in artifacts/chats/<id>/. Directory is the session. */

import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs"
import { randomBytes } from "node:crypto"
import path from "node:path"
import type { ChatMsg } from "./provider.js"
import { streamTurn } from "./provider.js"

export type ChatSpec = {
  id: string
  name: string
  created: string
  updated: string
}

function root(cwd: string): string {
  return path.join(cwd, "artifacts", "chats")
}

function specPath(cwd: string, id: string): string {
  return path.join(root(cwd), id, "spec.json")
}

function msgsPath(cwd: string, id: string): string {
  return path.join(root(cwd), id, "messages.json")
}

export function setTabTitle(name: string): void {
  const t = `aq · ${name}`
  process.stdout.write(`\x1b]0;${t}\x07`)
  process.stdout.write(`\x1b]2;${t}\x07`)
}

export function createChat(cwd: string): ChatSpec {
  const id = randomBytes(4).toString("hex")
  const now = new Date().toISOString()
  const spec: ChatSpec = { id, name: "new chat", created: now, updated: now }
  mkdirSync(path.join(root(cwd), id), { recursive: true })
  writeFileSync(specPath(cwd, id), JSON.stringify(spec, null, 2) + "\n")
  writeFileSync(msgsPath(cwd, id), "[]\n")
  return spec
}

export function loadSpec(cwd: string, id: string): ChatSpec {
  const p = specPath(cwd, id)
  if (!existsSync(p)) throw new Error(`no chat ${id}`)
  return JSON.parse(readFileSync(p, "utf8")) as ChatSpec
}

export function loadMessages(cwd: string, id: string): ChatMsg[] {
  const p = msgsPath(cwd, id)
  if (!existsSync(p)) return []
  try {
    const j = JSON.parse(readFileSync(p, "utf8")) as ChatMsg[]
    return Array.isArray(j) ? j : []
  } catch {
    return []
  }
}

export function saveChat(cwd: string, spec: ChatSpec, msgs: ChatMsg[]): void {
  spec.updated = new Date().toISOString()
  mkdirSync(path.join(root(cwd), spec.id), { recursive: true })
  writeFileSync(specPath(cwd, spec.id), JSON.stringify(spec, null, 2) + "\n")
  writeFileSync(msgsPath(cwd, spec.id), JSON.stringify(msgs, null, 2) + "\n")
}

export function listChats(cwd: string): ChatSpec[] {
  const dir = root(cwd)
  if (!existsSync(dir)) return []
  const out: ChatSpec[] = []
  for (const name of readdirSync(dir)) {
    const p = specPath(cwd, name)
    if (!existsSync(p)) continue
    try {
      out.push(JSON.parse(readFileSync(p, "utf8")) as ChatSpec)
    } catch {
      /* skip */
    }
  }
  out.sort((a, b) => (a.updated < b.updated ? 1 : -1))
  return out
}

export function latestChat(cwd: string): ChatSpec | null {
  return listChats(cwd)[0] ?? null
}

export function renameChat(cwd: string, spec: ChatSpec, name: string, msgs: ChatMsg[]): ChatSpec {
  const n = name.replace(/\s+/g, " ").trim().slice(0, 48) || "new chat"
  spec.name = n
  saveChat(cwd, spec, msgs)
  return spec
}

export function deleteChat(cwd: string, id: string): void {
  const dir = path.join(root(cwd), id)
  if (existsSync(dir)) rmSync(dir, { recursive: true, force: true })
}

function stubName(text: string): string {
  const s = text.replace(/\s+/g, " ").trim().slice(0, 48)
  return s || "new chat"
}

export async function nameChat(
  cwd: string,
  spec: ChatSpec,
  user: string,
  assistant: string,
  msgs: ChatMsg[],
): Promise<ChatSpec> {
  spec.name = stubName(user)
  saveChat(cwd, spec, msgs)
  setTabTitle(spec.name)
  try {
    const out = await streamTurn(
      [
        {
          role: "user",
          content: `Name this chat in 2-5 words. No quotes. No trailing punctuation.\n\nUser: ${user.slice(0, 400)}\nAssistant: ${assistant.slice(0, 400)}`,
        },
      ],
      () => {},
      { system: "Reply with only the name.", tools: [] },
    )
    const n = out.text
      .trim()
      .split("\n")[0]
      ?.replace(/^["'\s]+|["'\s.]+$/g, "")
      .slice(0, 48)
    if (n) spec.name = n
  } catch {
    /* stub name is enough */
  }
  saveChat(cwd, spec, msgs)
  setTabTitle(spec.name)
  return spec
}
