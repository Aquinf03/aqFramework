# Aquin Framework

A train is a directory. **aq** is the CLI.

Install once (from this repo):

```bash
cd aq && npm install && npm link
```

Then use it like any other command. Any directory, any terminal:

```bash
aq help
aq
aq agent
aq init my-train
cd my-train
aq train
aq job run -- echo hello
```

`npm link` puts `aq` on your PATH. After pulling CLI changes: `cd aq && npm run build`.