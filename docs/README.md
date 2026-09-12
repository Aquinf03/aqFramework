# Static aq docs (GitHub Pages)

HTML mirror of the product docs. Same stone / `#f5f5f3` chrome as `web/app/docs`.

**Regenerate** (after editing `web/lib/docs/sections`):

```bash
node --experimental-strip-types scripts/build-static-docs.mjs
```

Point GitHub Pages at the `/docs` folder on your default branch.
