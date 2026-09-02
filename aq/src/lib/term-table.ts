/** Plain aligned ASCII tables for CLI output. */

export function fmtCell(v: unknown): string {
  if (v == null || v === "") return "—"
  if (typeof v === "boolean") return v ? "yes" : "no"
  if (typeof v === "number" && Number.isFinite(v)) {
    const x = v
    if (Math.abs(x) >= 1000 || (Math.abs(x) > 0 && Math.abs(x) < 1e-3)) return x.toExponential(4)
    const s = String(Number(x.toFixed(6)))
    return s
  }
  return String(v)
}

export function renderTable(headers: string[], rows: unknown[][], indent = "  "): string {
  const grid = [headers.map(String), ...rows.map((r) => r.map(fmtCell))]
  const width = headers.length
  const widths = Array.from({ length: width }, (_, i) =>
    Math.max(...grid.map((row) => (row[i] ?? "").length)),
  )
  const line = (cells: string[]) =>
    indent +
    cells
      .map((c, i) => {
        const w = widths[i] ?? 0
        const looksNum = i > 0 && /^-?\d/.test(c)
        return looksNum ? c.padStart(w) : c.padEnd(w)
      })
      .join("  ")
  const out = [line(headers.map(String))]
  out.push(indent + widths.map((w) => "─".repeat(w)).join("  "))
  for (const row of rows) out.push(line(row.map(fmtCell)))
  return out.join("\n")
}

export function printTable(headers: string[], rows: unknown[][]): void {
  const text = renderTable(headers, rows)
  if (text) console.log(text)
}

export function printSection(title: string, headers: string[], rows: unknown[][]): void {
  console.log(title)
  if (!rows.length) {
    console.log("  (none)")
    return
  }
  printTable(headers, rows)
}
