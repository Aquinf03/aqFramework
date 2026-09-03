# Install

## Team install (usual)

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

This downloads the latest release tarball into `$HOME/.local/share/aquin-framework` (override with `AQUIN_INSTALL_DIR`), then runs `npm install` and `npm link` so `aq` is on your PATH under `~/.local/bin`.

### Requirements

- Node ≥ 18  
- npm  
- python3  
- curl + tar  

Kernel ML packages install into the framework’s `aq/kernel/.venv` from `aq/kernel/requirements.txt` (torch, transformers, peft, scikit-learn, Pillow, …).

### Smoke

```bash
aq help
aq doctor
aq version          # should match the release you installed
```

If `aq` is not found, add `~/.local/bin` to your PATH (the installer prints a hint).

## From a checkout

```bash
cd aq && npm install && npm link
# or from the repo root:
./install.sh
```

After TypeScript edits:

```bash
cd aq && npm run build
```

## Point install at a local tree

```bash
AQUIN_SOURCE=/path/to/aqfw ./install.sh
```

## Private git remote

```bash
AQUIN_REPO=git@github.com:YOUR_ORG/aqfw.git ./install.sh
# optional: AQUIN_BRANCH=main
```

## Environment overrides

| Env | Effect |
|-----|--------|
| `AQUIN_NO_RELEASE=1` | Skip the R2 tarball path |
| `AQUIN_RELEASE_URL` | Custom tarball URL |
| `AQUIN_INSTALL_DIR` | Install root (default `~/.local/share/aquin-framework`) |
| `AQUIN_NPM_PREFIX` | npm global prefix (default `~/.local`) |

## Upgrading

Re-run the install script. It replaces the framework copy cleanly so you do not keep stale CLI files from an older release.

Publishing releases (R2, wrangler) is maintainer work — see [author/release](./author/release.md).
