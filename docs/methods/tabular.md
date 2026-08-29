# Tabular methods

All of these expect CSV-like data with `data.path` and usually `data.target`. Eval uses shared feature alignment against the checkpoint’s feature list where `predict` exists.

| Method | Task | Notes |
|--------|------|-------|
| `linear` | regression | Closed-form / simple fit; 3-arg `fit(src, target, metric)` |
| `logistic` | binary classification | GD (~4000 steps, lr=0.2 hardcoded); classes `[neg, pos]` |
| `ridge` | regression + L2 | `lambda` / `penalty.lambda` (default 1) |
| `lasso` | regression + L1 | sparsity; default λ ≈ 0.5 |
| `elasticnet` | L1+L2 | `lambda`, `l1_ratio` (defaults ~0.4 / 0.5) |
| `tree` | CART regression | `depth` / `tree.depth` (default 3); inspect as flowchart |
| `forest` | bagged trees | `trees`, `depth`, `mtry`, `seed` |
| `boosting` | residual trees | `trees`, `depth`, `lr` (not an XGBoost wrap) |
| `gp` | RBF GP regression | `lengthscale`, `signal`, `noise`; auto-searches lengthscale if omitted |

## Checkpoint shape (typical)

Common fields: `kind`, `task`, `features`, weights / tree structures, sometimes `train_loss`. Inspect writers dump coefficients or tree text into `artifacts/inspect.md`.

## Manual E2E trains

See `tests/linear-regression`, `logistic-regression`, `ridge`, `lasso`, `elastic-net`, `decision-trees`, `random-forests`, `gradient-boosting`, `gaussian-processes`. Each folder is a real train with a README describing the gate.
