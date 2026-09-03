# Concepts

Three ideas carry almost everything.

## 1. A train is a directory

Identity is two files inside the folder:

- **`experiment.md`** — why this train exists (for you and the agent)  
- **`recipe.yaml`** — what the kernel should do  

The folder name does not matter. Rename freely. `aq init` creates a **new** folder; it never scatters files into `.`.

## 2. The recipe is the API

For built-in methods you should not need custom Python. Change family, method, data columns, steps, and eval in YAML. Custom fit code, when you need it, lives in **`tools/<name>.py`** in that train — not a global plugin registry.

## 3. Humans own eval

`aq eval` scores probes under `evals/` (or your data path). When `eval.min_score` is set you get pass/fail. When it is `null`, you get a score and no invented verdict. The agent must not invent gates.

---

### Naming

| Word | Meaning |
|------|---------|
| **Aquin** | Product / company |
| **aq** | This CLI |
| **train** | One experiment directory |
| **recipe** | `recipe.yaml` |

### Account vs model keys

| Store | Command | Purpose |
|-------|---------|---------|
| `~/.aquin/` | `aq login` | Aquin account / portal token |
| `~/.aq/` | `aq provider` | OpenAI / Anthropic / Grok / Ollama keys for the agent |

Do not mix them up.

### What aq will not do for you

- Silently swap an unsupported knob for “something close”  
- Pretend QLoRA works without CUDA bitsandbytes  
- Treat `size: llm` as a download of Llama  

Unsupported features **fail closed**. That is a feature.

Deeper product principles: [author/philosophy](./author/philosophy.md).
