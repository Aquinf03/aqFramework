# Plot charts from train artifacts

When the user asks for a graph, chart, plot, or diagram of training metrics or jobs:

1. Use the **plot** tool (or `aq plot`).
2. `kind: metrics` — loss/lr curve from `artifacts/metrics.jsonl`
3. `kind: jobs` — job status bar chart from `jobs/`
4. `kind: runs` — compare experiment scores from `artifacts/runs/`
5. `kind: all` — all of the above (default)

Output lands in `artifacts/plots/` (PNG by default). Tell the user the file path.

Optional `recipe.yaml`:

```yaml
plot:
  auto: true
  format: png
  dpi: 150
```

Global defaults also live in `~/.aq/config.json` under `plot`.
