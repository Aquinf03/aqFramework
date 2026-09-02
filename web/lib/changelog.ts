import fs from "fs";
import path from "path";

export type ChangelogEntry = {
  slug: string;
  title: string;
  body: string;
  unreleased: boolean;
};

const CHANGELOG_DIRS = [
  path.join(process.cwd(), "content/changelog"),
  path.join(process.cwd(), "../changelog"),
];

function changelogDir(): string | null {
  for (const dir of CHANGELOG_DIRS) {
    if (fs.existsSync(dir)) return dir;
  }
  return null;
}

function collectMarkdown(dir: string): { file: string; abs: string }[] {
  const out: { file: string; abs: string }[] = [];
  for (const name of fs.readdirSync(dir)) {
    const abs = path.join(dir, name);
    const st = fs.statSync(abs);
    if (st.isDirectory() && name === "versions") {
      for (const nested of fs.readdirSync(abs)) {
        if (nested.endsWith(".md")) out.push({ file: nested, abs: path.join(abs, nested) });
      }
      continue;
    }
    if (st.isFile() && name.endsWith(".md")) out.push({ file: name, abs });
  }
  return out;
}

export function listChangelog(): ChangelogEntry[] {
  const dir = changelogDir();
  if (!dir) return [];

  return collectMarkdown(dir)
    .sort((a, b) => b.file.localeCompare(a.file))
    .map(({ file, abs }) => {
      const body = fs.readFileSync(abs, "utf8");
      const slug = file.replace(/\.md$/, "");
      const first = body.split("\n")[0]?.replace(/^#\s*/, "") ?? slug;
      const head = body.slice(0, 200);
      const unreleased =
        /unreleased/i.test(first) ||
        /to be shipped/i.test(first) ||
        /unreleased/i.test(head) ||
        /to be shipped/i.test(head);
      return { slug, title: first, body: body.replace(/^#[^\n]+\n+/, ""), unreleased };
    });
}
