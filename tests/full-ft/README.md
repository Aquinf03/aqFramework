# Full fine-tune

**base** is next-token from scratch. **tune** loads that checkpoint and trains on prompt → completion with **loss on every token** (not completion-only). Inspect: `objective: full-ft`, `loss_on: all`, `parent`.

Train **base first**.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/full-ft/base
aq train
aq eval

cd ../tune
aq train
aq eval
cat artifacts/inspect.md
```

Expect: both eval **pass**. Tune inspect: `objective: full-ft`, `loss_on: all`, `parent`.

---

## 2. Agent

After base is trained:

```
cd tests/full-ft/tune
aq
```

Train this full fine-tune, then aq eval. Report pass/fail. Read inspect.md. objective, loss_on, parent?

```
aq ask tests/full-ft/tune -y "Train, then aq eval. Report pass/fail. Read inspect.md. objective, loss_on, parent?"
```
