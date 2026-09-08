# Aquin Framework

A train is a directory. **aq** is the CLI. The Python kernel ships inside the `aq` package.

**Docs:** https://aq.aquin.app — Getting started at [/docs](https://aq.aquin.app/docs).

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

More options: [Install](https://aq.aquin.app/docs/install).

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
| `tests/` | Train fixtures |
| `internals/` | Builder checklists + author notes |

- Publish: [`scripts/release.sh`](./scripts/release.sh) · [internals/author/release.md](./internals/author/release.md)
- Internals / coverage: [`internals/author/`](./internals/author/README.md) · [`internals/`](./internals/TODO.md)
