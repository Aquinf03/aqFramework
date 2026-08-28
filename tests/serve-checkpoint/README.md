# Serve a checkpoint

After **aq train**, **aq serve** loads the checkpoint and generates text. Kernel-owned forward pass. Tokenizer hash must match. Output: `artifacts/serve.json` plus the text on stdout.

Do not tick TODO until CLI and agent both work.

```
cd aq && npm run build
cd ..
```

---

## 1. Manual CLI

```
cd tests/serve-checkpoint
aq train
aq eval
aq serve "the cat"
cat artifacts/serve.json
aq status
```

Expect:

- eval **pass**
- serve prints generated **text** (prompt + completion)
- `artifacts/serve.json` has `prompt`, `text`, `completion`, `tokens`, `checkpoint`
- `aq status` shows the last serve line

Also works with recipe default:

```
aq serve
```

(`serve.prompt: the cat` in recipe.yaml)

---

## 2. Agent

```
cd tests/serve-checkpoint
aq
```

Train if needed, then aq serve with a prompt. Read artifacts/serve.json. Report text and token count. Do not invent output.

```
aq ask tests/serve-checkpoint -y "Train if needed, then aq serve the cat. Read artifacts/serve.json. text, tokens, checkpoint?"
```

---

## Pass bar

- CLI: train, eval pass, serve writes serve.json with non-empty text
- Agent: output from aq serve / serve.json, no fake generation
