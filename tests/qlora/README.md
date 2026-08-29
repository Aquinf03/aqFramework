# QLoRA

Needs **NVIDIA CUDA + bitsandbytes**. On MPS/CPU/ROCm, `bits: 4` / `method: qlora` fails closed — use LoRA without bits.

**base** is next-token. **tune** continues with QLoRA from that checkpoint (`init.checkpoint`).

```
cd tests/qlora/base && aq train   # on a CUDA machine
cd ../tune && aq train && aq eval
```

Expect checkpoint `objective: qlora`, `qlora_engine: bitsandbytes`, `bits: 4`.
