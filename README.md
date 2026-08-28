# Aquin Framework

A train is a directory. **aq** is the CLI. The Python kernel ships inside the `aq` package (`aq/kernel/`).

## Install

**Team (R2 release on aq.aquin.app):**

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

Tarballs live in Cloudflare R2; a Worker proxies `https://aq.aquin.app/releases/aq-latestv.tar.gz`. See `scripts/cloudflare/releases-worker/README.md` to set up the bucket and publish.

From a checkout:

```bash
cd aq && npm install && npm link
# or
./install.sh
```

**Private git remote** (SSH access to the repo):

```bash
AQUIN_REPO=git@github.com:YOUR_ORG/aqfw.git ./install.sh
```

Requires **Node ≥ 18**, **npm**, **python3**, and **curl/tar** for release install. Default install dir: `$HOME/.local/share/aquin-framework` (`AQUIN_INSTALL_DIR` to override).

## Use

```bash
aq help
aq
aq agent
aq init my-train
cd my-train
aq train
aq job run -- echo hello
```

After CLI changes: `cd aq && npm run build`.
