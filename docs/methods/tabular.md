# Tabular methods

Classic fits on a table. You need `data.path` and `data.target` in the recipe.

| Method | Notes |
|--------|------|
| `linear` | Linear regression |
| `logistic` | Classification |
| `ridge` / `lasso` / `elasticnet` | Use `lambda` / `l1_ratio` |
| `tree` | Decision tree (`depth`) |
| `forest` | Random forest |
| `boosting` | XGBoost → LightGBM → CatBoost → sklearn; set `library:` to force |
| `gp` | Gaussian process |

```yaml
family: tabular
method: ridge
lambda: 1.0
data:
  path: data.csv
  target: y
eval:
  metric: mse
```

After train, check `artifacts/inspect.md` and run `aq eval` on probes under `evals/`.

Full recipe keys: [Recipe](../recipe.md).
