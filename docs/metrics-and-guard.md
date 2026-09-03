# Metrics & guard

## Live output

While `aq train` runs on a real terminal you get a **live dashboard**: progress bars, latest stats, and a step table (latest row highlighted). Eval uses a separate **scoreboard**.

Pipes and CI stay plain tables. Turn the TUI off anytime:

```bash
AQ_TUI=0 aq train
```

Every event is also appended to **`artifacts/metrics.jsonl`** (one JSON object per line): steps, epochs, eval probes, errors, end summaries.

```bash
aq plot                 # charts from metrics / jobs / runs → artifacts/plots/
aq plot metrics
```

## Guard (opt-in)

Add to `recipe.yaml` only when you want fail-closed watches:

```yaml
guard:
  safety: true    # exploding loss / NaNs → abort
  leak: true      # train data overlapping eval probes → abort
```

Off by default so a first train stays simple. Turn them on once the loop is real.
