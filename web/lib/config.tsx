export const siteConfig = {
  name: "Aquin Labs",
  description:
    "Sign in to Aquin — CLI, desktop, and account management for reverse engineering intelligence with interpretability.",
  url: process.env.NEXT_PUBLIC_APP_URL ?? "https://auth.aquin.app",
  keywords: [
    "Aquin",
    "Aquin Labs",
    "aquin.app",
    "CLI login",
    "API keys",
    "account",
    "sign in",
    "interpretability",
    "AI research",
    "aq CLI",
  ],
  links: {
    email: "aquin@aquin.app",
    mainSite: "https://www.aquin.app",
    docs: "https://www.aquin.app/docs",
    changelog: "https://www.aquin.app/changelog",
  },
};

export type SiteConfig = typeof siteConfig;
