/** Framework version — single source: aq/package.json */

import { readFileSync } from "node:fs"
import path from "node:path"
import { aqRoot, kernelRoot } from "./root.js"

export type FrameworkVersion = {
  name: string
  version: string
  aqRoot: string
  kernel: string
  node: string
}

export function frameworkVersion(): FrameworkVersion {
  const root = aqRoot()
  let version = "0.0.0"
  let name = "aq"
  try {
    const pkg = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8")) as {
      name?: string
      version?: string
    }
    if (pkg.version) version = String(pkg.version)
    if (pkg.name) name = String(pkg.name)
  } catch {
    /* keep defaults */
  }
  return {
    name,
    version,
    aqRoot: root,
    kernel: kernelRoot(),
    node: process.version,
  }
}

/** One line for `aq version` / `-v`. */
export function versionLine(): string {
  const v = frameworkVersion()
  return `aq ${v.version} (Aquin framework)`
}

/** Multi-line detail for `aq version --verbose`. */
export function versionReport(verbose = false): string {
  const v = frameworkVersion()
  const lines = [`aq ${v.version}`, `Aquin framework`]
  if (verbose) {
    lines.push(`node ${v.node}`)
    lines.push(`root ${v.aqRoot}`)
    lines.push(`kernel ${v.kernel}`)
  }
  return lines.join("\n")
}
