> **Author docs** — for framework builders. End-user guide: [../README.md](../README.md).

# Coverage and roadmap

Internal checklist: [COMPLETED.md](../../../../internals/COMPLETED.md) (honest ticks) and [TODO.md](../../../../internals/TODO.md).

## Works today (proven)

- Tabular via scikit-learn (+ boosting libs when installed)
- HF transformers (encoder / decoder / enc-dec) with explicit `model:`
- HF LM: next-token, SFT, full-ft, LoRA, FIM, MLM, span, pack/mixture, continued-pretrain, serve, live steps
- QLoRA on **CUDA** with bitsandbytes
- Tokenizers: HF pin + optional local BPE/Unigram/WordPiece/byte
- Devices: CUDA / MPS / ROCm / CPU for supported paths
- Fail-closed rejects for fake deploy theater (`paged_kv` inactive; missing format tools)

## Not done

See [TODO.md](../../../../internals/TODO.md) — especially **Deploy / inference / size** (MTP, GPTQ/AWQ/GGUF, speculative, paged_kv, size zoo, AdaBoost method, vision/TinyML).

## Rule

Tick COMPLETED only when a stranger can `aq train` a recipe and get real weights — not metadata stubs. Unfinished items live in TODO only.
