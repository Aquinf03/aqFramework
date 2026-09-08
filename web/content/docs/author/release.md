# Release (maintainers)

How we publish `aq` tarballs to Cloudflare R2. **Users should follow [Install](../install.md)** instead.

## Pack + upload

```bash
chmod +x scripts/release.sh
./scripts/release.sh 0.0.3          # → aq-0.0.3v.tar.gz + updates aq-latestv.tar.gz
./scripts/release.sh latest         # → aq-latestv.tar.gz only
```

Requires `wrangler` logged in and bucket `aqfw-releases`. Details: `scripts/cloudflare/releases-worker/README.md`.

Objects:

- `aq-<version>v.tar.gz`  
- `aq-latestv.tar.gz`  
- `framework/install.sh`  

Public URLs (via Worker):

- https://aq.aquin.app/releases/aq-latestv.tar.gz  
- https://aq.aquin.app/releases/aq-<version>v.tar.gz  

## What the tarball contains

`aq/` (built CLI + kernel sources, **no** `node_modules` / `.venv`), plus root `install.sh` and `README.md`.

## Download metrics

```bash
./scripts/download-metrics.sh
./scripts/download-metrics.sh --days 7
```

Loads R2 creds from `scripts/.env.r2` or `web/.env` when present.
