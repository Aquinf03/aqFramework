# Multi-token prediction

`objective: mtp` trains auxiliary heads on a causal LM: at each position, predict **t+1**, **t+2**, … (**`n_predict:`**, default 2).

```
cd aq && npm run build
cd tests/mtp-pretrain
aq train
aq eval
cat artifacts/inspect.md
```

Expect: eval **pass**. Inspect `objective: mtp`, `n_predict: 2`.

Requires **`n_predict >= 2`**. LoRA/QLoRA not supported yet.
