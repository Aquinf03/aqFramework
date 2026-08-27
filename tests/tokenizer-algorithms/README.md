# Tokenizer algorithms

Four trains. Same tiny decoder, four **kinds**: `byte`, `bpe`, `wordpiece`, `unigram`. Inspect `tokenizer:` must match the recipe. File still hashed in the train (`artifacts/tokenizer.json`).

Do not tick TODO until all four CLI paths work (agent on at least one).

```
cd aq && npm run build
cd ..
```

---

## 1. Byte-level

UTF-8 bytes. Recipe `tokenizer: byte`.

```
cd tests/tokenizer-algorithms/byte
aq train
aq eval
cat artifacts/inspect.md
```

Expect: `tokenizer: byte`, eval **pass**.

## 2. BPE

```
cd tests/tokenizer-algorithms/bpe
aq train
aq eval
cat artifacts/inspect.md
```

Expect: `tokenizer: bpe`.

## 3. WordPiece

```
cd tests/tokenizer-algorithms/wordpiece
aq train
aq eval
cat artifacts/inspect.md
```

Expect: `tokenizer: wordpiece`.

## 4. Unigram

```
cd tests/tokenizer-algorithms/unigram
aq train
aq eval
cat artifacts/inspect.md
```

Expect: `tokenizer: unigram`.

---

## Agent

```
cd tests/tokenizer-algorithms/bpe
aq
```

Train, then aq eval. Report pass/fail. Read inspect.md. What tokenizer kind?

```
aq ask tests/tokenizer-algorithms/bpe -y "Train, then aq eval. Report pass/fail. Read inspect.md. tokenizer kind?"
```

---

## Pass bar

- All four: eval **pass**, inspect kind matches the recipe
- Agent: numbers from `aq eval`, no fake pass
