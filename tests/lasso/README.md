# Lasso — end to end

This folder is a train. `y = 2x + 3`. `junk` is noise, not a copy of `x`. Lasso (L1, `lambda: 0.4`) should keep `x` and shrink `junk` toward **0**. Gate is MSE on `evals/holdout.csv` (`min_score` 1, lower is better).

```
cd aq && npm run build
cd ..
```

`aq` is on your PATH or `./aq/bin/aq`.

Do not tick TODO until this works.

---

## 1. Manual CLI

```
cd tests/lasso
aq train
aq eval
cat artifacts/inspect.md
aq status
```

Expect:

- train writes checkpoint + `artifacts/inspect.md`
- eval: `mse`, small score, **pass**
- inspect: `lambda: 0.4`, `x` clearly nonzero, `junk` near 0 (may be exact 0)

---

## 2. Agent

```
cd tests/lasso
aq
```

- Train this lasso train, then eval the holdout. Report mse and pass/fail from aq eval. Do not invent a pass.
- Read inspect.md. Is junk near zero? Is x still in the model?

```
aq ask tests/lasso -y "Train lasso, then aq eval. Report mse and pass/fail. Read inspect.md."
```

---

## Pass bar

- CLI: eval pass, `x` alive, `junk` ~ 0
- Agent: same, no fake pass
