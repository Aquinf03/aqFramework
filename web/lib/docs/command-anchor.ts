export function commandAnchor(command: string): string {
  return command
    .toLowerCase()
    .replace(/^curl .*\| bash/, "curl-install")
    .replace(/^pip install /, "pip-install-")
    .replace(/^aq /, "aq-")
    .replace(/^aquin /, "aquin-")
    .replace(/[<>]/g, "")
    .replace(/\|/g, "")
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "");
}
