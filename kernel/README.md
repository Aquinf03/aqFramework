# kernel

Python. Train, eval, checkpoint, data hash. `aq` calls `run.py` with a train dir. Spec is `recipe.yaml`.

```
run.py          worker. reads artifacts/request.json, writes artifacts/result.json
protocol/       recipe, revision (hash), run record, method loader
engine/         train / eval / checkpoint steps
methods/        fit adapters. filename is the method (linear, logistic, ridge, lasso, elasticnet, tree, forest, boosting, gp, lora)
```

A train may add `methods/<name>.py`. Loader looks there first, then `kernel/methods/`.
