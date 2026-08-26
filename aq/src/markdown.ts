/** Small markdown → ANSI. Good enough for chat, not a spec. */

const RESET = "\x1b[0m"
const BOLD = "\x1b[1m"
const DIM = "\x1b[38;5;245m"
const ITAL = "\x1b[3m"
const CODE = "\x1b[38;5;252m"

function inline(s: string): string {
  let t = s
  t = t.replace(/`([^`]+)`/g, (_, x: string) => `${CODE}${x}${RESET}`)
  t = t.replace(/\*\*([^*]+)\*\*/g, (_, x: string) => `${BOLD}${x}${RESET}`)
  t = t.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, (_, x: string) => `${ITAL}${x}${RESET}`)
  t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label: string, url: string) => `${label}${DIM} (${url})${RESET}`)
  return t
}

export function renderMarkdown(src: string): string {
  const lines = src.replace(/\r\n/g, "\n").split("\n")
  const out: string[] = []
  let fence = false
  for (const line of lines) {
    if (/^```/.test(line)) {
      fence = !fence
      continue
    }
    if (fence) {
      out.push(`${DIM}${line}${RESET}`)
      continue
    }
    const h = line.match(/^(#{1,6})\s+(.*)$/)
    if (h) {
      out.push(`${BOLD}${inline(h[2] ?? "")}${RESET}`)
      continue
    }
    const li = line.match(/^\s*[-*]\s+(.*)$/)
    if (li) {
      out.push(`  • ${inline(li[1] ?? "")}`)
      continue
    }
    const num = line.match(/^\s*\d+\.\s+(.*)$/)
    if (num) {
      out.push(`  ${inline(num[1] ?? "")}`)
      continue
    }
    if (line.startsWith("> ")) {
      out.push(`${DIM}${inline(line.slice(2))}${RESET}`)
      continue
    }
    out.push(inline(line))
  }
  return out.join("\n")
}
