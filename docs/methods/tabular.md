# Tabular methods

All use **scikit-learn** (real estimators). `data.path` + `data.target` required.

| Method | Notes |
|--------|------|
| `linear` | LinearRegression |
| `logistic` | LogisticRegression |
| `ridge` / `lasso` / `elasticnet` | with `lambda` / `l1_ratio` |
| `tree` | DecisionTreeRegressor (`depth`) |
| `forest` | RandomForestRegressor |
| `boosting` | XGBoost → LightGBM → CatBoost → sklearn GBR (`library:` to force) |
| `gp` | GaussianProcessRegressor |

Checkpoints store coefficients and/or `estimator.joblib`. Eval uses `evaluate()` with the saved estimator when present.

Manual trains: `tests/linear-regression`, `logistic-regression`, `ridge`, `lasso`, `elastic-net`, `decision-trees`, `random-forests`, `gradient-boosting`, `gaussian-processes`.
