# Multi-token prediction

Same causal decoder as AR pretrain, plus **n_predict: 2** heads: each position also predicts the token after next. Inspect: `objective: mtp`, `causal: true`, `n_predict: 2`.

Do not tick TODO until CLI and agent both work.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/mtp-pretrain
aq train
aq eval
cat artifacts/inspect.md
```

Expect: eval **pass**. Inspect `objective: mtp`, `causal: true`, `n_predict: 2`.

---

## 2. Agent

```
cd tests/mtp-pretrain
aq
```

Train, then aq eval. Report pass/fail. Read inspect.md. objective, n_predict, causal?

```
aq ask tests/mtp-pretrain -y "Train, then aq eval. Report pass/fail. Read inspect.md. objective, n_predict, causal?"
```
