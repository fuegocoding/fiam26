# Earlier standalone implementation

This is the earlier root-level strategy, moved intact alongside its signal,
graph, reporting and regression-test modules. It is retained for comparison
with [final_v3](../../final_v3/README.md).

`main.py` is the single archived entrypoint. The old Git tree tracked both
`MAIN.py` and `main.py` with identical contents; they are consolidated here
to avoid a case-only collision on Windows.

Shared competition and external inputs still resolve to the repository's
`data/` and `external_data/` directories. Outputs are local and ignored.

```bash
python archive/initial_strategy/main.py --help
python -m pytest archive/initial_strategy/tests
```

The regression suite needs its research dependencies plus pytest. It checks
this archived implementation, not the final hybrid portfolio. Superseded
submission CSVs and portfolio exports are excluded from the public snapshot.
