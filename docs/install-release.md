# Install and release

## Requirements

- Node ≥ 18  
- npm  
- python3  
- curl + tar (for release install)

## User install paths

### 1. Team release (default)

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

Downloads `https://aq.aquin.app/releases/aq-latestv.tar.gz` into `$HOME/.local/share/aquin-framework` (override with `AQUIN_INSTALL_DIR`), then `npm install && npm link`.

### 2. From this checkout

```bash
cd aq && npm install && npm link
# or
./install.sh
```

### 3. Point install at a local tree

```bash
AQUIN_SOURCE=/path/to/aqfw ./install.sh
```

### 4. Private git remote

```bash
AQUIN_REPO=git@github.com:YOUR_ORG/aqfw.git ./install.sh
# optional: AQUIN_BRANCH=main
```

### Opt-outs / overrides

| Env | Effect |
|-----|--------|
| `AQUIN_NO_RELEASE=1` | Skip R2 tarball path |
| `AQUIN_RELEASE_URL` | Custom tarball URL |
| `AQUIN_INSTALL_DIR` | Install root |

After CLI TypeScript edits: `cd aq && npm run build`.

## What gets linked

The `aq` npm package includes `bin/`, `dist/`, `kernel/`, `assets/`. `kernelRoot()` resolves next to the package root, so the Python worker always matches the CLI you linked.

## Publish pipeline

| Piece | Path |
|-------|------|
| Pack + upload | `scripts/release.sh` - build `aq/`, tar `aq` + `install.sh` + `README.md`, put objects in R2 bucket `aqfw-releases` as `aq-<version>v.tar.gz` and `aq-latestv.tar.gz` |
| Download proxy | `scripts/cloudflare/releases-worker/` - routes `aq.aquin.app/releases/*` |
| Install script host | Auth app `prebuild` copies `install.sh` → `public/framework/` |

Optional Cloudflare Access can lock `/releases/*` while keeping the install script public.

## Smoke after install

```bash
aq help
aq doctor
aq init /tmp/smoke-train && cd /tmp/smoke-train && aq status
```
