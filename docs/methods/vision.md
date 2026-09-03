# Vision & VLM methods

## Vision (`family: vision`)

| method | Architectures | Data | Eval |
|--------|---------------|------|------|
| `cnn` | LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet, ConvNeXt | ImageFolder or path+label table | accuracy / loss |
| `vit` | ViT, Swin, DeiT, BEiT | same | accuracy / loss |

DeiT warms a CNN **teacher** on your data, then distills. BEiT trains a discrete VQ tokenizer, runs blockwise masked modeling, then classifies.

Full knobs: [Recipe — Vision](../recipe.md).

## Vision–language (`family: vlm`)

| method | Role | Data | Eval |
|--------|------|------|------|
| `clip` (`arch: clip` \| `siglip`) | Contrastive image↔text | pairs jsonl `{image, text}` | `recall@1` |
| `llava` (`arch: llava` \| `gpt4v-style`) | Chat with images | chat jsonl | `loss` |
| `flamingo` | Same job, gated cross-attn connector | chat jsonl | `loss` |

Generative VLMs need a causal LM in `model:` (Hugging Face id or a local folder). Vision towers and connectors are aq-owned.

Examples live under `tests/cnn-vision`, `tests/vit-vision`, `tests/vlm-clip`, `tests/vlm-llava` in the repo.
