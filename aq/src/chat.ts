import { startChatUi } from "./chat-ui.js"
import { latestChat, listChats, loadSpec } from "./chat-store.js"

export async function chatCmd(argv: string[]): Promise<void> {
  const cwd = process.cwd()
  const sub = argv[0]
  if (!sub || sub === "new") {
    await startChatUi(cwd)
    return
  }
  if (sub === "list" || sub === "ls") {
    const chats = listChats(cwd)
    if (!chats.length) {
      console.log("no chats")
      return
    }
    for (const c of chats) {
      console.log(`${c.id}  ${c.name}`)
    }
    return
  }
  if (sub === "last") {
    const last = latestChat(cwd)
    if (!last) throw new Error("no chats yet")
    await startChatUi(cwd, last.id)
    return
  }
  loadSpec(cwd, sub)
  await startChatUi(cwd, sub)
}
