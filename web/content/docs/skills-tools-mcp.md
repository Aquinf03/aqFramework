# Skills, tools & MCP

Extend **one train** without changing the global CLI.

## Tools (`tools/`)

Scripts you run with:

```bash
aq tool                 # list
aq tool my_script -- --flag
```

Supported: `.py`, `.ts`, `.js`, `.sh`. cwd is the train; `AQ_TRAIN` is set to its absolute path.

### Custom training method

If a built-in is not enough, add **`tools/<method>.py`** with:

```python
def fit(src, rec):
    ...
    return model_dict  # JSON-serializable checkpoint payload
```

Optional: `predict`, `evaluate`, `generate`, `write_inspect`. When `recipe.method` matches the filename, this `fit` **wins** over the kernel built-in for that train only.

See [Custom methods](./methods/custom.md).

## Skills (`skills/`)

Markdown (and optional runners) the agent can load as playbooks. Keep them short and operational — “how we plot metrics here,” not essays.

## MCP

Drop an MCP config the agent understands (stdio servers) under the train’s skills/MCP conventions. The agent can call those tools in chat the same way it calls built-ins.

## Memory

Agent memory is **global to your machine**: `~/.aq/memory/`. It is not copied by `aq fork`. Old in-train `memory/` folders migrate automatically when present.
