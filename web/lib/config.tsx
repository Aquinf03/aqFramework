export const siteConfig = {
  name: "Aquin Labs",
  description:
    "Developer environment and framework for building and checking models. Train folders, aq CLI, Python kernel, and the in-train agent.",
  url: process.env.NEXT_PUBLIC_APP_URL ?? "https://aq.aquin.app",
  keywords: [
    "Aquin",
    "Aquin Labs",
    "aq",
    "aquin.app",
    "train folder",
    "recipe.yaml",
    "CLI",
    "interpretability",
    "AI research",
  ],
  links: {
    email: "aquin@aquin.app",
    mainSite: "https://www.aquin.app",
    docs: "/docs",
    changelog: "/changelog",
    login: "/",
  },
};

export type SiteConfig = typeof siteConfig;
