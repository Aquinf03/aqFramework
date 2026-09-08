# Custom methods

When a built-in is not enough, add a fit adapter in **this train**:

```
tools/<method>.py
```

Set `recipe.method` to the same name. On `aq train`, if that file defines `fit()`, it wins over the built-in.

You can also run it with `aq tool <name>`.

## Minimal example

```python
# tools/watchdemo.py
from pathlib import Path

def fit(src: Path, rec: dict) -> dict:
    # read data from src using recipe fields
    # optional live metrics:
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
# def evaluate(model, src, rec) -> tuple[float, int]: ...
# def generate(model, prompt, rec, ...): ...
# def write_inspect(train, model) -> str: ...
```

Keep the returned dict JSON-serializable (paths as strings). Put large tensors on disk under the checkpoint slot and store relative paths in the dict.

See also [Skills, tools & MCP](../skills-tools-mcp.md).
