# Methods

Built-in methods are thin wrappers. **The recipe is the API.** Training uses real libraries under `aq/kernel/backends/`.

Install: `pip install -r aq/kernel/requirements.txt`

| method | Backend | Needs `recipe.model`? |
|--------|---------|------------------------|
| linear, logistic, ridge, lasso, elasticnet, tree, forest, gp | scikit-learn | no |
| boosting | xgboost → lightgbm → sklearn | no |
| llm, lora, qlora | Hugging Face + PEFT | **yes** |
| transformer | Hugging Face | **yes** |

Custom override: `{train}/methods/<name>.py` still wins if present.

- [Tabular](./tabular.md)
- [Transformers](./transformers.md)
- [LLM & LoRA](./llm.md)
- [Custom methods](./custom.md)
