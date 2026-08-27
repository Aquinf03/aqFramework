# LLMs — end to end

This folder is a train. **Size class: LLM** (decoder transformer), not a new architecture. Difference from `tests/transformers/decoder`: a **BPE tokenizer** hashed with the checkpoint (`artifacts/tokenizer.json`), **packed context windows**, next-token on English. Gate is holdout **loss** (`min_score` 2.5). Inspect must say `class: llm`, `arch: decoder`, `context`, `vocab`, `tokenizer_sha256`.

Not Llama weights. Not Hugging Face. Kernel-owned. SFT / LoRA stay later boxes.

Do not tick TODO until CLI and agent both work.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/llms
aq train
aq eval
cat artifacts/inspect.md
cat artifacts/tokenizer.json | head
aq status
```

Expect:

- train writes checkpoint, `artifacts/inspect.md`, `artifacts/tokenizer.json`
- eval: `loss` below 2.5, **pass**
- inspect: `class: llm`, `arch: decoder`, `layers: 2`, a **context**, a **vocab**, `tokenizer: bpe`, a sha256

If eval fails, do not mark the size class done.

---

## 2. Agent

```
cd tests/llms
aq
```

Train this LLM, then eval. Report loss and pass/fail from aq eval. Read inspect.md. Vocab, context, tokenizer hash?

```
aq ask tests/llms -y "Train the LLM, then aq eval. Report loss and pass/fail. Read inspect.md. class, context, vocab, tokenizer_sha256?"
```

---

## Pass bar

- CLI: eval **pass**, inspect is an LM slot (tokenizer + context), not the `ab`/`cd` transformer toy
- Agent: same numbers from `aq eval`, no fake pass
