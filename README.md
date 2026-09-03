# Aquin Framework

A train is a directory. **aq** is the CLI. The Python kernel ships inside the `aq` package.

**Docs (how to use aq):** [`docs/`](./docs/README.md) — start with [Getting started](./docs/getting-started.md).

## Install

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

Needs **Node ≥ 18**, **npm**, and **python3**. Then `aq help` · `aq doctor`.

From a checkout:

```bash
cd aq && npm install && npm link
# or
./install.sh
```

More options: [docs/install.md](./docs/install.md).

## Use

```bash
aq init my-train && cd my-train
# edit recipe.yaml + data
aq train && aq eval && aq status
```

Or open the agent on a TTY: `aq`

**Recipe is the train API.** For LLM/LoRA set `model:` to a hub id. QLoRA needs CUDA + bitsandbytes. Unsupported knobs fail closed.

## Maintainers

- Publish: [`scripts/release.sh`](./scripts/release.sh) · [docs/author/release.md](./docs/author/release.md)  
- Internals / coverage: [`docs/author/`](./docs/author/README.md) · [`internals/`](./internals/TODO.md)
