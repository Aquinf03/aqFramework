> **Author docs** — for framework builders. End-user guide: [../README.md](../README.md).

# Web and auth

Package: `web/` — Next.js **docs + auth** app at **https://aq.aquin.app**.  
Marketing site docs at `aquin.app/docs` are gone; they redirect here.

> Product description (package.json): “Aquin docs + auth — aq CLI docs, Supabase login, CLI tokens, desktop handoff, and SDK API”

## Surfaces

| Surface | Role |
|---------|------|
| `/` Home / `AuthPortal` | Install commands, sign-in, signup, ready, desktop handoff |
| `/docs` … | Getting started + full aq docs (ported from former aquin.app/docs) |
| Desktop/CLI handoff | `/?view=desktop&client=cli` — mint code + `aquin://` deep link |
| User profile | `/user/[username]` |
| Auth routes | Reset password, OAuth callback |
| Policies | Terms, Privacy, License, AUP, Security → marketing site (`www.aquin.app`) |

Branding defaults (`web/lib/config.tsx`): **Aquin Labs**, URL **`https://aq.aquin.app`**.

Framework install script is copied into `public/framework/install.sh` on `prebuild` so:

```bash
curl -fsSL https://aq.aquin.app/framework/install.sh | bash
```

## Identity

- Supabase auth + `profiles` (name, username, avatar)
- Username allocate/update under `/api/account/username/`
- CLI tokens / API keys UI (`CliTokenSection`) + `/api/keys` (list, generate, reveal, revoke)

## CLI bridge

```bash
aq login                 # opens aq.aquin.app/?view=desktop&client=cli; paste code
aq login --token aq-…    # existing token
aq login --check
aq logout / aq switch
```

Tokens land in `~/.aquin/config.json`. Override portal base with `AQUIN_AUTH_URL` when developing.

## SDK API (`web/app/api/sdk/`)

Keyed endpoints for VM/run-style telemetry: `ping`, `whoami`, `heartbeat`, `metrics`, `logs`, `notify`, `finish`, `analyses` (+ runs). Env: Supabase keys, `VM_SERVICE_KEY`, optional billing (`DODO_*`) - see `web/.env.example`.

## Design alignment

Same stone / Host Grotesk language as the marketing site. Docs chrome includes search, policies dropdown, and CLI token when signed in.
