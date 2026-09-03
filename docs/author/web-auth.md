> **Author docs** — for framework builders. End-user guide: [../README.md](../README.md).

# Web and auth

Package: `web/` - Next.js **auth portal** and thin keyed **SDK API**. It is **not** the train UI. Disk remains source of truth for trains.

> Product description (package.json): “Aquin auth portal - Supabase login, CLI tokens, desktop handoff, and SDK API”

The stock create-next-app `web/README.md` is outdated; this document is the reference.

## Surfaces

| Surface | Role |
|---------|------|
| Home / `AuthPortal` | Email → password → signup → ready → desktop handoff |
| Desktop/CLI handoff | `?view=desktop&client=cli` - mint code + `aquin://` deep link |
| User profile | `/user/[username]` |
| Auth routes | Reset password, OAuth callback |
| Policies | Terms, Privacy, License, AUP, Security → marketing site |

Branding defaults (`web/lib/config.tsx`): **Aquin Labs**, auth URL family around `auth.aquin.app` / `aq.aquin.app`.

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
aq login                 # opens portal desktop/CLI view; paste code
aq login --token aq-…    # existing token
aq login --check
aq logout / aq switch
```

Tokens land in `~/.aquin/config.json`. Override portal base with `AQUIN_AUTH_URL` when developing.

## SDK API (`web/app/api/sdk/`)

Keyed endpoints for VM/run-style telemetry: `ping`, `whoami`, `heartbeat`, `metrics`, `logs`, `notify`, `finish`, `analyses` (+ runs). Env: Supabase keys, `VM_SERVICE_KEY`, optional billing (`DODO_*`) - see `web/.env.example`.

## Design alignment

PLAN: “The UI is a viewer over the same directories. It does not own state.” Auth gives you identity and tokens so CLI/cloud pieces can trust who you are. It does not replace `artifacts/` or `recipe.yaml`.
