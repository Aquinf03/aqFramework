# tests

Manual end-to-end trains. Not `aq eval` probes. Each folder is a train plus a README.

- [linear-regression](linear-regression/) — first coverage family
- [logistic-regression](logistic-regression/) — binary classification
- [ridge](ridge/) — linear + L2 penalty
- [lasso](lasso/) — linear + L1, can zero weights
- [elastic-net](elastic-net/) — L1 + L2 mix
- [decision-trees](decision-trees/) — CART splits
- [random-forests](random-forests/) — bagged trees
- [gradient-boosting](gradient-boosting/) — sequential residual trees
- [gaussian-processes](gaussian-processes/) — RBF posterior mean
- [transformers](transformers/) — encoder, decoder, encoder-decoder (kernel-owned)
- [llms](llms/) — decoder LM slot: BPE + packed context + next-token
- [slms](slms/) — smaller decoder (`size: slm`)
- [edge](edge/) — TinyML-scale decoder (`size: edge`)
- [distilled](distilled/) — teacher → student
- [quantized](quantized/) — int8
- [weight-formats](weight-formats/) — q8 / GPTQ-lite / AWQ-lite / EXL2-lite / GGUF-lite
- [pruned](pruned/) — magnitude prune
- [speculative](speculative/) — draft + verify
- [paged-kv](paged-kv/) — paged KV
- [size-deploy](size-deploy/) — index of those knobs

