# guard-safety

Opt-in `guard.safety`: one good train, one bad train that blows up **for several steps** and gets killed.

```
aq train tests/guard-safety/good
aq train tests/guard-safety/bad
```

**good** — should finish (~2s).

**bad** — should fail mid-run with `guard.safety: loss blew up for N consecutive steps…`.
