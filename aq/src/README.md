# aq/src

TypeScript CLI. The current directory is the train.

```
cli.ts          entry. routes verbs.
help.ts
core/           schema, paths, kernel bridge, package roots
handle/         train verbs: init fork checkout status diff data step stage schedule tool
job/            queue, waiter, resources
agent/          chat, ask, spawn, doctor, provider
lib/            files, explore, memory, skills, mcp, web, registry
```

Kernel lives in `kernel/` inside this package. Templates in `templates/`. Assets in `assets/`.
