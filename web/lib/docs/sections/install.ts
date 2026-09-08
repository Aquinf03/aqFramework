import type { DocSection } from "../types";

export const INSTALL: DocSection = {
  title: "Install",
  intro:
    "Aquin ships as the aq CLI (TypeScript) plus a Python kernel. Needs Node ≥ 18, npm, and python3. After install run aq help, then aq doctor.",
  prerequisite: "Node ≥ 18 · npm · python3 · curl/tar for the release install",
  tools: [
    {
      command: "curl -fsSL https://aq.aquin.app/framework/install.sh | bash",
      description:
        "Install the team release. Downloads aq-latestv.tar.gz into ~/.local/share/aquin-framework, runs npm install && npm link, and sets up the kernel venv.",
      flags: [
        { name: "AQUIN_INSTALL_DIR", description: "Override install location (default ~/.local/share/aquin-framework)." },
        { name: "AQUIN_RELEASE_URL", description: "Override the release tarball URL." },
        { name: "AQUIN_NO_RELEASE", description: "Skip the release download and use a local/source tree." },
      ],
      example: "curl -fsSL https://aq.aquin.app/framework/install.sh | bash",
    },
    {
      command: "cd aq && npm install && npm link",
      description: "Install from an aqfw checkout.",
      flags: [],
      example: "cd aq && npm install && npm link",
    },
    {
      command: "python3 -m venv aq/kernel/.venv",
      description: "Create the kernel virtualenv in a checkout install.",
      flags: [],
      example: [
        "python3 -m venv aq/kernel/.venv",
        "aq/kernel/.venv/bin/pip install -r aq/kernel/requirements.txt",
      ],
    },
    {
      command: "aq help",
      description: "Short Unix-style help for the aq CLI.",
      flags: [],
      example: "aq help",
    },
    {
      command: "aq doctor",
      description: "Health check: node, python, kernel, provider, train folder, skills, MCP. Exits 1 if any check fails.",
      flags: [{ name: "[dir]", description: "Optional train directory to validate." }],
      example: "aq doctor",
    },
    {
      command: "aq login",
      description:
        "Sign in via the auth portal (or paste a token). Account tokens live in ~/.aquin/config.json. Provider API keys are separate under ~/.aq/config.json.",
      flags: [
        { name: "--token aq-…", description: "Install an existing CLI token." },
        { name: "--check", description: "Print who is signed in." },
      ],
      example: [
        "aq login",
        "aq login --token aq-…",
        "aq login --check",
        "aq logout",
        "aq switch",
      ],
    },
    {
      command: "aq provider",
      description: "Configure LLM providers for the agent (openai, anthropic, grok, ollama). Keys in ~/.aq/config.json.",
      flags: [
        { name: "aq provider <name>", description: "Save key and select provider." },
        { name: "aq provider use <name>", description: "Switch active provider." },
      ],
      example: ["aq provider", "aq provider openai", "aq provider use anthropic"],
    },
  ],
};
