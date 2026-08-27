# Pack / mixture / context windows

This train mixes **web** and **code** rows 50/50, packs them into **context 16** with EOS between docs. Inspect must name `context`, `pack: eos`, `mixture`, and `packed_tokens` for both sources.

Do not tick TODO until CLI and agent both work.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/pack-mixture
aq train
aq eval
cat artifacts/inspect.md
```

Expect:

- eval **pass**
- inspect: `context: 16`, `pack: eos`, mixture `web` and `code`, packed_tokens for both

---

## 2. Agent

```
cd tests/pack-mixture
aq
```

Train, then aq eval. Report pass/fail. Read inspect.md. What is context? What mixture? Did both sources get packed tokens?

```
aq ask tests/pack-mixture -y "Train, then aq eval. Report pass/fail. Read inspect.md. context, pack, mixture, packed_tokens?"
```

---

## Pass bar

- CLI: eval pass, inspect shows context + pack + mixture + both sources
- Agent: numbers from aq eval, no fake pass
