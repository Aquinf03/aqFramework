import type { Metadata } from "next";
import { OG_IMAGE_HEIGHT, OG_IMAGE_WIDTH } from "@/lib/utils";

const SITE = (process.env.NEXT_PUBLIC_APP_URL ?? "https://aq.aquin.app").replace(/\/$/, "");
const TWITTER_HANDLE = "@AquinF03";
const OG_IMAGE = "/og/docs.jpg";

export type DocsPageMeta = {
  path: string;
  title: string;
  description: string;
  keywords: string[];
  /** Markdown file under content/docs/ (or public/docs/) for agent scrape. */
  markdown?: string;
  priority?: number;
};

export const DOCS_PAGES: DocsPageMeta[] = [
  {
    path: "/",
    title: "Aquin — install, sign in, start a train",
    description:
      "Install the aq CLI, sign in, run aq doctor, and start a train. Account and docs for Aquin.",
    keywords: ["aq", "aquin", "install", "aq login", "aq doctor", "aq init", "CLI"],
    markdown: "getting-started.md",
    priority: 1,
  },
  {
    path: "/docs",
    title: "Getting started with aq",
    description:
      "What Aquin is: developer environment and framework. Train folders, aq CLI, Python kernel, and the in-train agent.",
    keywords: ["aq", "aquin", "developer environment", "framework", "train folder", "kernel", "agent"],
    markdown: "getting-started.md",
    priority: 0.95,
  },
  {
    path: "/docs/install",
    title: "Install aq",
    description:
      "Install the aq CLI via curl release script or npm link from checkout. Node ≥ 18, python3 kernel, aq doctor, aq login, aq provider.",
    keywords: ["aq install", "aq.aquin.app", "npm link", "aq doctor", "aq login"],
    markdown: "install.md",
    priority: 0.94,
  },
  {
    path: "/docs/train",
    title: "Train folder",
    description:
      "Canonical layout after aq init: instructions.md, recipe.yaml, data/, evals/, methods/, tools/, skills/, artifacts/, jobs/, stages/. Fork and checkout.",
    keywords: ["aq init", "train folder", "artifacts", "aq fork", "aq checkout"],
    markdown: "train-folder.md",
    priority: 0.94,
  },
  {
    path: "/docs/recipe",
    title: "Recipe",
    description:
      "recipe.yaml is the full train API: tabular, transformers, LLM/LoRA/QLoRA, custom methods, and opt-in guard.",
    keywords: ["recipe.yaml", "lora", "qlora", "tabular", "transformer", "guard"],
    markdown: "recipe.md",
    priority: 0.93,
  },
  {
    path: "/docs/cli",
    title: "CLI reference",
    description: "aq train, eval, checkpoint, serve, status, diff, tool, doctor. Account login and provider keys.",
    keywords: ["aq train", "aq eval", "aq serve", "aq diff", "aq doctor", "cli"],
    markdown: "cli.md",
    priority: 0.93,
  },
  {
    path: "/docs/methods/tabular",
    title: "Methods · Tabular",
    description: "scikit-learn tabular methods: linear, logistic, ridge, lasso, elasticnet, tree, forest, boosting, gp.",
    keywords: ["tabular", "sklearn", "ridge", "boosting", "aq train"],
    markdown: "methods/tabular.md",
    priority: 0.9,
  },
  {
    path: "/docs/methods/transformers",
    title: "Methods · Transformers",
    description: "Hugging Face transformer train with arch encoder, decoder, or encoder-decoder. model required.",
    keywords: ["transformer", "encoder", "decoder", "huggingface"],
    markdown: "methods/transformers.md",
    priority: 0.9,
  },
  {
    path: "/docs/methods/llm",
    title: "Methods · LLM / LoRA",
    description:
      "Foundation training with LoRA, QLoRA, SFT, full-ft, FIM, MLM, span, continued-pretrain. CUDA, MPS, ROCm, CPU.",
    keywords: ["lora", "qlora", "sft", "llm", "peft", "aq serve"],
    markdown: "methods/llm.md",
    priority: 0.91,
  },
  {
    path: "/docs/methods/custom",
    title: "Methods · Custom",
    description: "Override the kernel with methods/<name>.py implementing fit(src, rec).",
    keywords: ["custom method", "methods/", "fit"],
    markdown: "methods/custom.md",
    priority: 0.88,
  },
  {
    path: "/docs/eval",
    title: "Eval & inspect",
    description:
      "User probes in evals/. aq eval is the human gate. artifacts/inspect.md for post-fit dumps. aq diff for run compare.",
    keywords: ["aq eval", "evals/", "inspect.md", "aq diff"],
    markdown: "eval-and-inspect.md",
    priority: 0.9,
  },
  {
    path: "/docs/metrics",
    title: "Metrics & guard",
    description: "artifacts/metrics.jsonl live stream. Opt-in guard.safety and guard.leak fail closed.",
    keywords: ["metrics.jsonl", "guard", "safety", "leak"],
    markdown: "metrics-and-guard.md",
    priority: 0.88,
  },
  {
    path: "/docs/jobs",
    title: "Jobs",
    description: "Local job queue: aq job run/list/log/cancel/resume/tree with cpu ram disk gpu asks.",
    keywords: ["aq job", "jobs/", "checkout", "resources"],
    markdown: "jobs.md",
    priority: 0.88,
  },
  {
    path: "/docs/agent",
    title: "Agent",
    description: "aq agent chat in the train folder. aq ask, aq chat, aq spawn. Same cwd and verbs.",
    keywords: ["aq agent", "aq ask", "aq chat", "aq spawn"],
    markdown: "agent.md",
    priority: 0.9,
  },
  {
    path: "/docs/skills",
    title: "Skills, tools & MCP",
    description: "tools/, skills/ with SKILL.md and mcp.json, memory/. Convention over registration.",
    keywords: ["aq tool", "skills", "mcp", "memory"],
    markdown: "skills-tools-mcp.md",
    priority: 0.88,
  },
  {
    path: "/docs/schedules",
    title: "Schedules & stages",
    description: "aq schedule tick/run and nested stages/ trains via aq stage init/train/eval.",
    keywords: ["aq schedule", "aq stage", "stages/", "cron"],
    markdown: "schedules-stages.md",
    priority: 0.88,
  },
];

const PAGE_MAP = new Map(DOCS_PAGES.map((p) => [p.path, p]));

export function getDocsPageMeta(pathname: string): DocsPageMeta {
  return PAGE_MAP.get(pathname) ?? DOCS_PAGES[0];
}

export function docsMarkdownHref(pathname: string): string {
  const page = getDocsPageMeta(pathname);
  return `/docs/${page.markdown ?? "README.md"}`;
}

export function buildDocsMetadata(path: string): Metadata {
  const page = getDocsPageMeta(path);
  const canonical = `${SITE}${page.path}`;
  const fullTitle = `${page.title} | aq Docs`;
  const mdHref = docsMarkdownHref(page.path);

  return {
    metadataBase: new URL(SITE),
    title: { absolute: fullTitle },
    description: page.description,
    keywords: page.keywords,
    applicationName: "Aquin",
    category: "technology",
    authors: [{ name: "Aquin Labs", url: SITE }],
    creator: "Aquin Labs",
    publisher: "Aquin Labs",
    alternates: {
      canonical,
      types: {
        "text/markdown": `${SITE}${mdHref}`,
        "text/plain": `${SITE}/llms.txt`,
      },
    },
    openGraph: {
      title: fullTitle,
      description: page.description,
      url: canonical,
      siteName: "Aquin",
      type: "article",
      locale: "en_US",
      images: [
        {
          url: OG_IMAGE,
          width: OG_IMAGE_WIDTH,
          height: OG_IMAGE_HEIGHT,
          alt: page.title,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: fullTitle,
      description: page.description,
      site: TWITTER_HANDLE,
      creator: TWITTER_HANDLE,
      images: [OG_IMAGE],
    },
    robots: {
      index: true,
      follow: true,
      googleBot: {
        index: true,
        follow: true,
        "max-image-preview": "large",
        "max-snippet": -1,
        "max-video-preview": -1,
      },
    },
  };
}

export function docsSitemapEntries(baseUrl: string) {
  const lastModified = new Date();
  return DOCS_PAGES.map((page) => ({
    url: `${baseUrl}${page.path}`,
    lastModified,
    changeFrequency: (page.path === "/" ? "weekly" : "monthly") as "weekly" | "monthly",
    priority: page.priority ?? 0.8,
  }));
}

export function buildAskPrompt(pathname: string): string {
  const page = getDocsPageMeta(pathname);
  const url = `${SITE}${page.path}`;
  const md = `${SITE}${docsMarkdownHref(page.path)}`;
  return [
    `I'm reading the aq (Aquin) documentation: "${page.title}" (${url}).`,
    page.description,
    `Prefer the markdown source at ${md} when scraping.`,
    "Explain what this section covers, how it fits into Aquin as a developer environment and framework (train folder, aq CLI, Python kernel, agent), and answer questions about these parts.",
    `Docs live at ${SITE} (Getting started: ${SITE}/docs).`,
    `Docs index for agents: ${SITE}/llms.txt`,
  ].join(" ");
}
