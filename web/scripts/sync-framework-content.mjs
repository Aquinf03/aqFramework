#!/usr/bin/env node
/** Sync install.sh + changelog into Next public/content. */
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs"
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

const site = (process.env.NEXT_PUBLIC_APP_URL || "https://aq.aquin.app").replace(/\/$/, "")
const llms = [
  "# Aquin / aq",
  "",
  "> Developer environment and framework for building and checking models.",
  "> Docs are HTML pages on this site (no separate markdown tree).",
  "",
  `Home: ${site}/`,
  `Getting started: ${site}/docs`,
  `Sitemap: ${site}/sitemap.xml`,
  "",
  "## Pages",
  "",
  `- ${site}/`,
  `- ${site}/docs`,
  `- ${site}/docs/install`,
  `- ${site}/docs/train`,
  `- ${site}/docs/recipe`,
  `- ${site}/docs/cli`,
  `- ${site}/docs/agent`,
  `- ${site}/docs/eval`,
  `- ${site}/docs/jobs`,
  `- ${site}/changelog`,
  "",
].join("\n")

writeFileSync(path.join(webRoot, "public", "llms.txt"), llms)

console.log("sync-framework-content: install.sh + changelog + llms.txt")
