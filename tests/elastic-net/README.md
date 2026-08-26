# Elastic Net — end to end

This folder is a train. `y ≈ 2x + 3`. `z` is a noisy copy of `x`. `junk` is noise. Elastic Net (L1+L2, `lambda: 0.3`, `l1_ratio: 0.5`) should keep `x` and `z` finite and shrink `junk` toward 0. Gate is MSE on `evals/holdout.csv` (`min_score` 1).

```
cd aq && npm run build
cd ..
```

Do not tick TODO until this works.

---

## 1. Manual CLI

```
cd tests/elastic-net
aq train
aq eval
cat artifacts/inspect.md
aq status
```

Expect:

- train writes checkpoint + `artifacts/inspect.md`
- eval: `mse`, small score, **pass**
- inspect: `lambda` and `l1_ratio`, `x` and `z` nonzero, `junk` near 0

---

## 2. Agent

```
cd tests/elastic-net
aq
```

- Train this elastic net train, then eval the holdout. Report mse and pass/fail from aq eval. Do not invent a pass.
- Read inspect.md. Are x and z both in? Is junk near zero?

```
aq ask tests/elastic-net -y "Train elasticnet, then aq eval. Report mse and pass/fail. Read inspect.md."
```

---

## Pass bar

- CLI: eval pass, x and z finite, junk ~ 0
- Agent: same, no fake pass
