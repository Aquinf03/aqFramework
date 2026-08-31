# Changelog

All notable changes to **Aquin** (the `aq` CLI and Python kernel) are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Version numbers come from `aq/package.json` (`aq version`).

---

## [0.0.1] — 2026-08-31

First release. A **train** is a directory; **`recipe.yaml`** is the spec; **`aq`** runs fit, eval, jobs, and an optional agent on top.

### Added — CLI & train protocol

- **`aq init`** — creates a skeleton train (`instructions.md`, `recipe.yaml`, `data/`, `evals/`, `tools/`, `skills/`, …). Never dumps into cwd: uses `aq-experiment` or `aq-experiment-newN` (or a named folder with the same collision rule).
- **`aq fork`** — copy a train; skips `jobs/` and `artifacts/`.
- **`aq checkout`** — restore a prior run tree (optionally into a new directory).
- **`aq data hash`** — hash `recipe.data.path`; optional `--snapshot`.
- **`aq train` / `aq eval` / `aq checkpoint` / `aq serve`** — kernel-backed fit, scoring, checkpoint management, and generation.
- **`aq tool`** — run scripts under `tools/`.
- **`aq schedule`** — sweeps, cron, resume-on-fail, and ordered pipelines (`train → eval → …`).
- **`aq stage`** — nested trains under `stages/<name>/`.
- **`aq diff` / `aq status`** — compare run records; surface jobs, last run, eval, schedule logs.
- **`aq version`** (`-v` / `--version`; `--verbose` for paths).
- **`aq doctor`** — health check (Node, Python, kernel, provider, train layout, skills/MCP).

### Added — Jobs

- **`aq job run | list | log | cancel | resume | tree`** — one running job per train, queued jobs under `jobs/`, resource asks, process trees, survives terminal disconnect when used as a background worker.

### Added — Agent

- **`aq` / `aq agent`** — TTY chat with tools, memory, and provider switching.
- **`aq ask` / `aq ask --json`** — one-shot prompts without the chat UI.
- **`aq chat list | last | <id>`** — resume sessions; chats live under `~/.aq/chats/` (global list with `aq chat list --all`).
- **`aq provider`** — OpenAI, Anthropic, Grok, Ollama; keys in `~/.aq`.
- **`aq login` / `aq logout` / `aq switch`** — Aquin account tokens via aq.aquin.app.
- **`aq spawn run | list | log | cancel`** — background worker agents as jobs under `artifacts/agents/`.
- **Skills, tools, MCP** — `skills/` (markdown, code, stdio MCP), `tools/` (runnable files), agent builtins (`write`, `grep`, `aq_*`, …).

### Added — Kernel (Python)

- Recipe-driven training — no custom Python for built-in methods.
- **IPC:** `artifacts/request.json` → `artifacts/result.json`; live step metrics on stderr and in `artifacts/metrics.jsonl`.
- **Opt-in guards:** `guard.safety` (non-finite / exploding loss), `guard.leak` (data leakage heuristics).
- **Devices:** CUDA, Apple MPS, AMD ROCm (Torch HIP), CPU.

#### Tabular (scikit-learn + optional boosters)

Linear and logistic regression, ridge, lasso, elastic net, decision trees, random forests, gradient boosting (prefers XGBoost → LightGBM → CatBoost → sklearn when libraries are installed), Gaussian processes.

#### Transformers (Hugging Face; `model:` required)

Encoder, decoder, and encoder–decoder architectures.

#### LLM / foundation (Hugging Face + PEFT; `model:` required)

- Objectives: `next-token`, `sft`, `full-ft`, `lora`, `qlora`, `mlm`, `span`, `fim`, `continued-pretrain`.
- Train-local tokenizers: BPE, Unigram, WordPiece, byte-level (`tokenizer:` in recipe).
- Pack / mixture weights / context windows.
- **`init.checkpoint`** — continue from a prior aq checkpoint.
- **Post-train deploy knobs that are real:** magnitude **`prune:`**, aq weight dump **`quant: int8|int4|binary`** (inspect/export; serve still loads HF weights).
- **QLoRA** (`bits: 4` or `objective: qlora`) on NVIDIA CUDA with bitsandbytes.

### Added — Install & release

- **`curl -fsSL https://aq.aquin.app/framework/install.sh | bash`** — user-local install under `~/.local` (no sudo).
- Stable launcher at **`~/.local/bin/aq`** (absolute paths to Node + `dist/cli.js`; survives npm link quirks).
- Kernel venv at **`aq/kernel/.venv`** via `python3 -m venv` + `python -m pip install -r aq/kernel/requirements.txt`.
- Release tarballs on R2 (`aq-<version>v.tar.gz`, `aq-latestv.tar.gz`); see `docs/install-release.md`.

### Changed

- **Honesty pass:** removed toy/stub paths. Unsupported recipe knobs **fail closed** with a clear error instead of fake success (see *Known limitations*).
- **`size:` is label-only** (`llm` / `slm` / `edge`). You must set **`model:`** to a Hugging Face id or local path.
- **SFT** uses completion-only loss (prompt tokens masked); **full-ft** trains all non-pad tokens.
- **Dynamic padding** + seq2seq collator for LM training (large speedup vs fixed `max_length` padding on every row).
- **AMP / dtype:** load bf16/fp16 weights where supported; never force bf16 on GPUs that lack it (e.g. T4 → fp16). Gradient checkpointing is **opt-in** (`gradient_checkpointing: true`).
- **VRAM:** no `device_map=auto` for normal full-GPU trains; fused Adam on CUDA when available; clearer OOM hints.
- **`aq init` cwd behavior** — always creates a new folder; safe to rename the train directory anytime.

### Fixed

- **Install:** `aq: No such file or directory` after curl install — launcher now always written and smoke-tested; `hash -r` hint when bash cached a dead path.
- **Install:** venv/pip shebang failures on Debian/Ubuntu — recreate venv, use `python -m pip`, hint `python3-venv`.
- **Train interrupt:** single **Ctrl+C** stops a run (kernel in its own process group; Node SIGKILLs the tree). No more HF Trainer “first Ctrl+C = soft stop, second = exit”.
- **FP16 GradScaler** crashes / OOM on mixed setups — safer dtype coercion and loading.
- **Chat storage** — sessions under `~/.aq/chats/`; legacy train-local chat paths migrated on access; agent no longer creates empty train `artifacts/` for chat alone.
- **Module resolution** — ESM/CJS extension handling in the CLI build.

### Removed

- **PyAquin** and other placeholder backends that pretended to train or deploy without real libraries.
- Fake success for **`formats`**, **`speculative`**, **`paged_kv`**, **`objective: mtp`**, and **`size:`-as-model-picker**.

### Known limitations (0.0.1)

These recipe knobs **error on purpose** until implemented:

| Knob | Status |
|------|--------|
| `objective: mtp` | No multi-token prediction heads |
| `formats: true` | No GPTQ / AWQ / GGUF / EXL2 exporter |
| `speculative: true` | No draft-model speculative decoding |
| `paged_kv: true` | No paged attention / continuous batching |
| `size:` without `model:` | Label only; does not select weights |

Also:

- **QLoRA** requires CUDA + bitsandbytes (use LoRA elsewhere).
- **CatBoost / LightGBM** for boosting are optional pip installs (`pip install catboost lightgbm`).
- **Serve** loads Hugging Face weights; aq quant dumps are for inspect/export, not production inference formats.
- Coverage walk continues — see `internals/TODO.md` for methods not yet shipped (SVM, DPO, vision nets, …).

### Requirements

- **Node.js ≥ 18**, **npm**, **python3** (with `python3-venv` on Debian/Ubuntu).
- Kernel deps: `pip install -r aq/kernel/requirements.txt` (torch, transformers, peft, scikit-learn, xgboost, …).

[0.0.1]: https://github.com/Aquinf03/Framework/releases/tag/v0.0.1
