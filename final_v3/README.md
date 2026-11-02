# Final strategy implementation

This is the implementation behind Stochastic Pirates' final McGill-FIAM 2026
presentation. The team finished **4th**. The published local hybrid replay
returned **15.65% CAGR**, **1.22 Sharpe**, **0.80 IR** and **-10.96% monthly
maximum drawdown** over January 2021-August 2026.

- [Strategy and construction](../docs/strategy.md)
- [Final results and limitations](../docs/results.md)
- [Research process and agent experiments](../docs/research-process.md)
- [Inputs](../docs/data.md) and [reproduction commands](../docs/reproducibility.md)
- [Aggregate evidence](../results/final_strategy/README.md)

## Entrypoint

Run from the repository root:

```bash
python final_v3/main.py --help
python final_v3/main.py --no-discovery --device cpu --reuse
```

The second command requires the separately supplied inputs and trains any
missing components. `--reuse` accepts only compatible caches. The default
application mode enables discovery, so select `--no-discovery` explicitly for
the final fixed baseline. No live Arcana API call is needed for the baseline;
the saved beta input is read from `data/arcana/`.

## Modules

| Module | Responsibility |
| :--- | :--- |
| `main.py`, `config.py` | CLI, orchestration and input locations |
| `sleeves/`, `combiner.py` | Signal preparation, annual training and fixed blend |
| `graph/` | Company-text/peer graph and earnings-forecast rank overlay |
| `optimizer.py`, `deployment.py`, `turnover.py` | Selection, liquidity-aware sizing, turnover and capital settlement |
| `neutrality.py`, `risk.py` | Formation beta estimates and position/book controls |
| `official.py`, `disclosed_ibes.py`, `identity.py` | Competition input adapters, licensed exception and security identity |
| `evaluate.py`, `report.py`, `part2.py` | Accounting, benchmarks, diagnostics and reporting |
| `checkpoints.py`, `timing.py` | Payload/input hashes, replay and causal timing checks |
| `discovery/` | Optional research agents and evidence-gated sleeves |

`final_v2/` remains at the repository root because discovery imports its shared
Carbon Arc transport and reads its pod charters. Its workflow has the separate
entrypoint `final_v2/run.py`.

## Saved work and historical notes

Generated models, forecasts, checkpoint archives and private exports were
relocated to ignored `local/artifacts/final_v3/`. New runs still use `output/`
and normal caching. Aggregate final results are published under
`results/final_strategy/`.

The documents in `docs/` retain detailed implementation and experiment history.
[Current strategy](docs/CURRENT_STRATEGY.md), [checkpoints](docs/CHECKPOINTS.md),
[Part 2 accounting](docs/PART2.md) and the [input map](docs/OFFICIAL_DATA_MAP.md)
are the technical references. Older rerun reports describe their named
experiments, not the final presentation. Internal research notes and draft
reports are excluded from the public snapshot.
