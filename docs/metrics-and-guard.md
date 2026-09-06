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
  safety: true    # sustained exploding loss / NaNs → abort
  leak: true      # train data overlapping eval probes → abort
```

Off by default so a first train stays simple. Turn them on once the loop is real.

### Safety behavior

A **single** spike does **not** kill the run. `guard.safety` warns (`guard.warn`) and only aborts after consecutive bad steps:

| knob | default | meaning |
|------|---------|---------|
| `blowup_factor` | `8` | flag when `loss > best * factor` |
| `blowup_warmup` | `20` | steps before blow-up checks |
| `blowup_patience` | `3` | consecutive spike steps before abort |
| `nan_patience` | `2` | consecutive non-finite losses before abort |
| `max_loss` | unset | optional absolute ceiling (same patience as blow-up) |

Set `blowup_patience: 1` if you want the old hair-trigger stop on the first spike.
