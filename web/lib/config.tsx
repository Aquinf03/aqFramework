export const siteConfig = {
  name: "Aquin Labs",
  description: "Sign in to Aquin — CLI, desktop, and account management.",
  url: process.env.NEXT_PUBLIC_APP_URL ?? "https://auth.aquin.app",
  keywords: ["Aquin", "aquin.app", "CLI login", "API keys"],
  links: {
    email: "aquin@aquin.app",
    mainSite: "https://www.aquin.app",
    docs: "https://www.aquin.app/docs",
  },
};

export type SiteConfig = typeof siteConfig;
