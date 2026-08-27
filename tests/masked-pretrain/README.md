# Masked LM / span corruption

Two trains. **Not** next-token.

- `mlm/` — BERT-style: hide tokens, encoder fills them (`objective: mlm`, `causal: false`)
- `span/` — T5-style: drop a span, encoder-decoder emits it (`objective: span`)

Do not tick TODO until both CLI paths work (agent on at least one).

```
cd aq && npm run build
cd ..
```

---

## 1. MLM

```
cd tests/masked-pretrain/mlm
aq train
aq eval
cat artifacts/inspect.md
```

Expect: eval **pass**. Inspect `objective: mlm`, `causal: false`, `mask_rate`.

## 2. Span

```
cd tests/masked-pretrain/span
aq train
aq eval
cat artifacts/inspect.md
```

Expect: eval **pass**. Inspect `objective: span`.

---

## Agent

```
cd tests/masked-pretrain/mlm
aq
```

Train, then aq eval. Report pass/fail. Read inspect.md. Is it mlm, not next-token?

```
aq ask tests/masked-pretrain/mlm -y "Train, then aq eval. Report pass/fail. Read inspect.md. objective, causal, mask_rate?"
```
