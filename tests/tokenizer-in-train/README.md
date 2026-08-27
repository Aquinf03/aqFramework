# Tokenizer in the train — hashed like weights

This folder is a train. The tokenizer is **`artifacts/tokenizer.json`**. Its sha256 is on the checkpoint and in the run record. Eval encodes with **that file**, not a new fit. Change the file, eval must refuse.

Do not tick TODO until CLI and agent both work.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/tokenizer-in-train
aq train
aq eval
aq data hash
cat artifacts/inspect.md
```

Expect:

- train prints `artifacts/tokenizer.json` and `tokenizer sha256:...`
- that hash is in `artifacts/checkpoints/last.json` as `tokenizer_sha256`
- eval **pass** (loads the pinned file)
- `aq data hash` after train also prints the tokenizer sha256

Then:

```
printf '%s\n' '{"kind":"bpe","itos":["x"],"merges":[]}' > artifacts/tokenizer.json
aq eval
```

Expect: **fail**: tokenizer.json hash does not match checkpoint. Re-train to restore the file before you tick.

---

## 2. Agent

```
cd tests/tokenizer-in-train
aq
```

Train, then eval. Report pass/fail from aq eval. Read inspect.md. What is tokenizer_sha256? Then break tokenizer.json and eval again. Did it refuse?

```
aq ask tests/tokenizer-in-train -y "Train, then aq eval. Report pass/fail. Read inspect.md tokenizer_sha256. Then write junk to artifacts/tokenizer.json and aq eval. Did it fail on hash?"
```

---

## Pass bar

- CLI: train pins hash, eval passes, junk tokenizer.json makes eval refuse
- Agent: same, no fake pass
