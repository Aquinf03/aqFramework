# QLoRA

**base** is next-token from scratch. **tune** loads that checkpoint, **quantizes it to 4-bit**, freezes it, and trains LoRA on the LM head. Inspect: `objective: qlora`, `quant: int4`, `rank`, `frozen: true`, `parent`.

Train **base first**.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/qlora/base
aq train
aq eval

cd ../tune
aq train
aq eval
cat artifacts/inspect.md
```

Expect: both eval **pass**. Tune inspect: `objective: qlora`, `quant: int4`, `rank: 4`, `frozen: true`.

---

## 2. Agent

After base is trained:

```
cd tests/qlora/tune
aq
```

Train this QLoRA, then aq eval. Report pass/fail. Read inspect.md. objective, quant, rank, frozen?

```
aq ask tests/qlora/tune -y "Train, then aq eval. Report pass/fail. Read inspect.md. objective, quant, rank, frozen, parent?"
```
