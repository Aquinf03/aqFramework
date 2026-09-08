# Aquin Framework

A train is a directory. **aq** is the CLI. The Python kernel ships inside the `aq` package.

**Docs:** https://aq.aquin.app — Getting started at [/docs](https://aq.aquin.app/docs).  
Markdown source (canonical): [`web/content/docs/`](./web/content/docs/README.md).

## Published build install

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

Needs **Node ≥ 18**, **npm**, and **Python >= 3.10**. Then `aq help` · `aq doctor`.

## From a checkout (dev)

```bash
cd aq && npm install && npm link
# or
./install.sh
```

More options: [web/content/docs/install.md](./web/content/docs/install.md).

## Use

```bash
aq init my-train && cd my-train
# edit recipe.yaml + data
aq train && aq eval && aq status
```

## Layout

| Path | Role |
|------|------|
| `aq/` | CLI + Python kernel |
| `web/` | Docs + auth app (`aq.aquin.app`) |
| `web/content/docs/` | User + author markdown |
| `tests/` | Train fixtures |
| `internals/` | Builder checklists |

- Publish: [`scripts/release.sh`](./scripts/release.sh) · [web/content/docs/author/release.md](./web/content/docs/author/release.md)
- Internals / coverage: [`web/content/docs/author/`](./web/content/docs/author/README.md) · [`internals/`](./internals/TODO.md)
