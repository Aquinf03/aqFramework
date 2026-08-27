# Continued pretrain

Two trains. **base** is next-token from scratch. **continue** loads that checkpoint, keeps the tokenizer, and next-token trains on extra lines. Inspect on continue: `objective: continued-pretrain`, `parent`, `parent_sha256`.

Train **base first**. Then continue.

Do not tick TODO until both CLI paths work (agent on continue after base exists).

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/continued-pretrain/base
aq train
aq eval

cd ../continue
aq train
aq eval
cat artifacts/inspect.md
```

Expect: both eval **pass**. Continue inspect: `objective: continued-pretrain`, a `parent` path, `parent_sha256`.

---

## 2. Agent

From continue, after base is trained:

```
cd tests/continued-pretrain/continue
aq
```

Train this continued pretrain (parent is ../base). Then aq eval. Report pass/fail. Read inspect.md. objective, parent, parent_sha256?

```
aq ask tests/continued-pretrain/continue -y "Train, then aq eval. Report pass/fail. Read inspect.md. objective, parent, parent_sha256?"
```

If base has no checkpoint, train `tests/continued-pretrain/base` first.
