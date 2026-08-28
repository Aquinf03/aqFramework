# Framework releases (R2 + Worker)

Packaged `aq` tarballs live in R2. A Worker on `aq.aquin.app` serves them at:

```
https://aq.aquin.app/releases/aq-latestv.tar.gz
https://aq.aquin.app/releases/aq-<version>v.tar.gz
```

The install script is static on the auth app:

```
https://aq.aquin.app/framework/install.sh
```

## One-time setup

1. Create R2 bucket **`aqfw-releases`** (Cloudflare dashboard → R2).

2. Deploy the worker:

   ```bash
   cd scripts/cloudflare/releases-worker
   npm install
   npx wrangler deploy
   ```

3. Add a **Worker route** (Workers & Pages → aqfw-releases → Settings → Domains & Routes):

   ```
   aq.aquin.app/releases/*
   ```

   Requests on this path hit R2. Everything else on `aq.aquin.app` stays on your Next auth app.

4. Optional — **Cloudflare Access** on `/releases/*` so only your team can download. Install script stays public; tarball is gated.

## Publish a release

From repo root:

```bash
chmod +x scripts/release.sh
./scripts/release.sh latest          # → aq-latestv.tar.gz
./scripts/release.sh 0.0.1           # → aq-0.0.1v.tar.gz + updates aq-latestv.tar.gz
```

Requires `wrangler login`.

## Team install

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

Downloads `aq-latestv.tar.gz` from R2 via the Worker, runs `npm install && npm link` in `aq/`.
