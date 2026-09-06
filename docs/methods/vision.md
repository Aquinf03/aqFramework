# Vision & VLM methods

## Vision (`family: vision`)

| method | Architectures | Data | Eval |
|--------|---------------|------|------|
| `cnn` | LeNet, AlexNet, VGG, ResNet, Inception, EfficientNet, ConvNeXt | ImageFolder or path+label table | accuracy / loss |
| `vit` | ViT (t/s/b/l/h/g · patch 14/16/32), Swin, DeiT, BEiT | same | accuracy / loss |

DeiT warms a CNN **teacher** on your data, then distills. BEiT trains a discrete VQ tokenizer, runs blockwise masked modeling, then classifies.

Full knobs: [Recipe — Vision](../recipe.md).

## Vision–language (`family: vlm`)

| method | Role | Data | Eval |
|--------|------|------|------|
| `clip` (`arch: clip` \| `siglip`) | Contrastive image↔text | pairs jsonl `{image, text}` | `recall@1` |
| `llava` (`arch: llava` \| `gpt4v-style`) | Chat with images | chat jsonl | `loss` |
| `flamingo` | Same job, gated cross-attn connector | chat jsonl | `loss` |

Set `vision:` to any size you want (`vit-h/14`, `ViT-H-14`, …). Optional `vision_pretrained:` loads an OpenCLIP weight tag (`pip install open-clip-torch`). Generative VLMs also need a causal LM in `model:` (HF id or local folder).

After train:

```bash
aq serve "what is in this image?" --image data/foo.png --max-tokens 64   # llava / flamingo
aq serve "a photo of a cat" --image data/foo.png                         # clip similarity
aq serve --image data/foo.png                                            # cnn / vit classify
```

Examples live under `tests/cnn-vision`, `tests/vit-vision`, `tests/vlm-clip`, `tests/vlm-llava` in the repo.
