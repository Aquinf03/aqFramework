#!/usr/bin/env node
/** Sync install.sh + changelog into Next public/content — Windows-safe (no mkdir -p / cp). */
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const webRoot = path.join(path.dirname(fileURLToPath(import.meta.url)), "..")
const repo = path.join(webRoot, "..")

function cpGlob(dir, dest, re) {
  if (!existsSync(dir)) {
    console.warn(`sync-framework-content: skip missing ${dir}`)
    return
  }
  mkdirSync(dest, { recursive: true })
  for (const name of readdirSync(dir)) {
    if (!re.test(name)) continue
    copyFileSync(path.join(dir, name), path.join(dest, name))
  }
}

mkdirSync(path.join(webRoot, "public", "framework"), { recursive: true })
mkdirSync(path.join(webRoot, "content", "changelog", "versions"), { recursive: true })

const installSrc = path.join(repo, "install.sh")
if (existsSync(installSrc)) {
  copyFileSync(installSrc, path.join(webRoot, "public", "framework", "install.sh"))
} else {
  console.warn(`sync-framework-content: skip missing ${installSrc}`)
}

cpGlob(path.join(repo, "changelog"), path.join(webRoot, "content", "changelog"), /\.md$/)
cpGlob(
  path.join(repo, "changelog", "versions"),
  path.join(webRoot, "content", "changelog", "versions"),
  /\.md$/,
)
