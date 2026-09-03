# Stages

A **stage** is a nested train under `stages/<name>/`. It has its own `experiment.md` and `recipe.yaml` — a full train, not a half recipe.

```bash
aq stage                    # list
aq stage init prep          # scaffold stages/prep/
aq stage prep               # train that stage
aq stage eval prep          # eval that stage
```

Use stages when a pipeline needs separate fits (tokenize → pretrain → SFT) that should stay inspectable as their own trains, not one mega-recipe.

Parent trains and stages share the same CLI verbs; only the path changes.
