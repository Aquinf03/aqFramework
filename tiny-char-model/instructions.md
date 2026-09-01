# Tiny garden language model

Train a very small causal language model on the original garden-and-robot corpus in `data.jsonl`.

Workflow:

1. Run `aq train`.
2. Run `aq eval` and report the measured holdout loss without inventing a pass/fail gate.
3. Run `aq serve` with a short garden-themed prompt to inspect generated text.

This is a learning experiment, not a production model. The corpus is intentionally tiny and the expected output is imitation rather than factual language ability.
