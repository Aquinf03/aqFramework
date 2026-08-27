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
- [tokenizer-in-train](tokenizer-in-train/) — tokenizer file hashed like weights; eval loads that file
- [tokenizer-algorithms](tokenizer-algorithms/) — byte, BPE, WordPiece, Unigram
- [pack-mixture](pack-mixture/) — pack, mixture weights, context windows
- [ar-pretrain](ar-pretrain/) — next-token / causal LM
- [masked-pretrain](masked-pretrain/) — MLM and span corruption
- [mtp-pretrain](mtp-pretrain/) — multi-token prediction (t+1, t+2)
- [fim-pretrain](fim-pretrain/) — fill-in-the-middle (prefix / suffix / middle)
- [continued-pretrain](continued-pretrain/) — warm-start from a parent checkpoint
- [sft](sft/) — supervised fine-tune (loss on completion)
- [full-ft](full-ft/) — full fine-tune (loss on all tokens)
- [lora](lora/) — LoRA (frozen parent, rank adapters)

