#!/usr/bin/env node
/** Sync install.sh, changelog, and docs markdown into Next public/content. */
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs"
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

function copyTree(src, dest, { skipDirs = [] } = {}) {
  if (!existsSync(src)) {
    console.warn(`sync-framework-content: skip missing ${src}`)
    return
  }
  mkdirSync(dest, { recursive: true })
  for (const name of readdirSync(src)) {
    if (skipDirs.includes(name)) continue
    const from = path.join(src, name)
    const to = path.join(dest, name)
    if (statSync(from).isDirectory()) {
      copyTree(from, to, { skipDirs })
    } else {
      copyFileSync(from, to)
    }
  }
}

function listMarkdown(dir, prefix = "") {
  if (!existsSync(dir)) return []
  const out = []
  for (const name of readdirSync(dir)) {
    const full = path.join(dir, name)
    const rel = prefix ? `${prefix}/${name}` : name
    if (statSync(full).isDirectory()) {
      if (name === "author") continue
      out.push(...listMarkdown(full, rel))
    } else if (name.endsWith(".md")) {
      out.push(rel)
    }
  }
  return out.sort()
}

mkdirSync(path.join(webRoot, "public", "framework"), { recursive: true })
mkdirSync(path.join(webRoot, "content", "changelog", "versions"), { recursive: true })
mkdirSync(path.join(webRoot, "content", "docs"), { recursive: true })
mkdirSync(path.join(webRoot, "public", "docs"), { recursive: true })

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

const docsSrc = path.join(repo, "docs")
copyTree(docsSrc, path.join(webRoot, "content", "docs"), { skipDirs: ["author"] })
copyTree(docsSrc, path.join(webRoot, "public", "docs"), { skipDirs: ["author"] })

const site = (process.env.NEXT_PUBLIC_APP_URL || "https://aq.aquin.app").replace(/\/$/, "")
const mdFiles = listMarkdown(path.join(webRoot, "public", "docs"))
const llms = [
  "# Aquin / aq",
  "",
  "> Developer environment and framework for building and checking models.",
  "> Prefer these markdown sources when scraping. HTML docs mirror the same content.",
  "",
  `Home: ${site}/`,
  `Sitemap: ${site}/sitemap.xml`,
  `Full dump: ${site}/llms-full.txt`,
  "",
  "## Docs (markdown)",
  "",
  ...mdFiles.map((f) => `- ${site}/docs/${f}`),
  "",
  "## HTML",
  "",
  `- ${site}/`,
  `- ${site}/docs`,
  `- ${site}/docs/install`,
  `- ${site}/docs/train`,
  `- ${site}/docs/recipe`,
  `- ${site}/docs/cli`,
  `- ${site}/docs/agent`,
  `- ${site}/changelog`,
  "",
].join("\n")

writeFileSync(path.join(webRoot, "public", "llms.txt"), llms)

const fullParts = [
  `# Aquin / aq — full docs dump`,
  `# Generated for agent scrape. Source: repo docs/`,
  "",
]
for (const f of mdFiles) {
  const body = readFileSync(path.join(webRoot, "public", "docs", f), "utf8")
  fullParts.push(`\n\n---\n# ${f}\n# ${site}/docs/${f}\n---\n\n`)
  fullParts.push(body.trimEnd())
  fullParts.push("\n")
}
writeFileSync(path.join(webRoot, "public", "llms-full.txt"), fullParts.join(""))

console.log(`sync-framework-content: ${mdFiles.length} docs markdown files → public/docs + llms.txt`)
