# Transformers (text)

Hugging Face transformers with a required **`model:`**.

```yaml
method: transformer
model: bert-base-uncased
arch: encoder                 # decoder | encoder | encoder-decoder
steps: 50
data:
  path: data.csv
  text: text
  target: label               # encoder
  # src / tgt for encoder-decoder
eval:
  metric: accuracy            # or loss, depending on task
```

| arch | Typical use |
|------|-------------|
| `encoder` | Classification / encoding with a target column |
| `decoder` | Causal / generative text |
| `encoder-decoder` | Seq2seq (`data.src` / `data.tgt`) |

For LoRA / SFT-style LLM recipes prefer `family: llm` — see [LLM & LoRA](./llm.md).
