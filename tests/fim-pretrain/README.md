# Fill-in-the-middle

Causal decoder. Each doc is rearranged **prefix / suffix / middle** so infill is next-token. Inspect: `objective: fim`, `causal: true`, `fim_order: psm`.

Do not tick TODO until CLI and agent both work.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/fim-pretrain
aq train
aq eval
cat artifacts/inspect.md
```

Expect: eval **pass**. Inspect `objective: fim`, `causal: true`, `fim_order: psm`.

---

## 2. Agent

```
cd tests/fim-pretrain
aq
```

Train, then aq eval. Report pass/fail. Read inspect.md. objective, fim_order, causal?

```
aq ask tests/fim-pretrain -y "Train, then aq eval. Report pass/fail. Read inspect.md. objective, fim_order, causal?"
```
