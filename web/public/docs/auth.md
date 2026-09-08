# Login & accounts

`aq login` connects the CLI to your **Aquin account** (portal token). This is separate from `aq provider` model keys.

```bash
aq login                     # open portal, paste code
aq login --token aq-…        # paste an existing token
aq login --check             # who am I
aq switch                    # list accounts
aq switch you@example.com    # switch active
aq logout                    # drop active
aq logout you@example.com    # drop that account
```

Tokens live under **`~/.aquin/`**. Provider API keys live under **`~/.aq/`**.

If login fails, check the browser handoff URL, that the code has not expired, and that `aq login --check` sees the account you expect.

Maintainer / portal implementation notes: [author/web-auth](./author/web-auth.md).
