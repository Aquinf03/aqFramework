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

export function listChangelog(): ChangelogEntry[] {
  const dir = changelogDir();
  if (!dir) return [];

  return fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .sort((a, b) => b.localeCompare(a))
    .map((file) => {
      const body = fs.readFileSync(path.join(dir, file), "utf8");
      const slug = file.replace(/\.md$/, "");
      const first = body.split("\n")[0]?.replace(/^#\s*/, "") ?? slug;
      const unreleased = /unreleased/i.test(first) || /unreleased/i.test(body.slice(0, 200));
      return { slug, title: first, body: body.replace(/^#[^\n]+\n+/, ""), unreleased };
    });
}
