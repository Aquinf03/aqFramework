# Random forests — end to end

This folder is a train. `y` is **1** for small `x` and **9** for large `x`. `junk` is noise. A forest (`trees: 20`, bootstrap) should still track `x`. Gate is MSE on `evals/holdout.csv` (`min_score` 1). Inspect lists tree count and how often each feature was used, not one flowchart.

```
cd aq && npm run build
cd ..
```

Do not tick TODO until this works.

---

## 1. Manual CLI

```
cd tests/random-forests
aq train
aq eval
cat artifacts/inspect.md
aq status
```

Expect:

- train writes checkpoint + `artifacts/inspect.md`
- eval: `mse` small, **pass**
- inspect: `trees: 20`, `x` used more often than `junk`

---

## 2. Agent

```
cd tests/random-forests
aq
```

- Train this forest, then eval the holdout. Report mse and pass/fail from aq eval. Do not invent a pass.
- Read inspect.md. How many trees? Did x get used more than junk?

```
aq ask tests/random-forests -y "Train the forest, then aq eval. Report mse and pass/fail. Read inspect.md."
```

---

## Pass bar

- CLI: eval pass, inspect shows a forest not a single tree
- Agent: same, no fake pass
