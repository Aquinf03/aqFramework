# Custom methods (via `tools/`)

Custom **fit** adapters live in **`tools/<name>.py`**, same folder as other train scripts. Set `recipe.method: <name>`. On `aq train`, the kernel loads `tools/<name>.py` if it defines `fit()`; otherwise it uses `aq/kernel/methods/<name>.py`.

You can also run the same file with `aq tool <name>` — one file, two entry points.

## Minimal example

```python
# tools/watchdemo.py
from __future__ import annotations
from pathlib import Path

def fit(src: Path, rec: dict) -> dict:
    # read csv/jsonl from src using recipe fields
    # optionally:
    #   from protocol import metrics as aq_metrics
    #   aq_metrics.step(step=i, loss=loss, lr=lr)
    return {
        "kind": "watchdemo",
        "train_loss": 0.0,
        # … weights / state …
    }

def predict(model: dict, X: list) -> list:
    return [0.0] * len(X)

# optional:
# def evaluate(model, src, rec): ...
# def generate(model, prompt, rec, max_tokens=None, temperature=None): ...
# def write_inspect(train, model) -> str:  # relative path under train
```

## Fit signatures

- **2-arg** `fit(src, rec)` - default for custom methods and most built-ins.  
- **3-arg** `fit(src, target, metric)` - only if your function’s parameter count is ≥ 3 (used by `linear` / `logistic`).

`src` is the resolved `Path` to `data.path`. `rec` is the full recipe dict.

## Metrics & guard

If you call `aq_metrics.step(..., loss=…)`, users see live steps and can enable `guard.safety`. Always return a dict; set `train_loss` if you want post-fit safety to see a final number.

## Reference

`tests/guard-safety/*/tools/watchdemo.py` is a full custom fit adapter that emits steps and can settle or blow up based on `recipe.demo`.
