# Reproducing the strategy

The public tree lets readers inspect the method and verify aggregate monthly
results. It does not contain all inputs for a complete model rebuild. The
published figures are a saved historical replay, not a new run performed
as part of repository cleanup.

## Environment and inputs

Use Python 3.13 and the recorded dependencies:

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python final_v3/main.py --help
```

The dependency file records the research environment; installation is not a
bit-for-bit replay guarantee. Supply the files in [the data guide](data.md),
including the licensed revision container and saved betas for the final hybrid.

## Train a new fixed-baseline run

From the repository root:

```bash
python final_v3/main.py --no-discovery --device cpu --data-dir data --external-file external_data/external.parquet --arcana-beta-file data/arcana/arcana_spy_beta_previous_holdings.csv --max-abs-beta 0.025 --output-dir final_v3/output
```

Use `--device cuda` with an appropriate CUDA/Torch installation. This fits
boosted-tree and graph models, constructs the book and generates reports.
`--reuse` resumes only compatible source/input caches. A change in inputs,
implementation, model environment or accepted agent sleeves can change results.

The application's default mode enables discovery. The explicit
`--no-discovery` above selects the fixed baseline, without LLM/scouting calls.
`--with-discovery` requires configured research providers and may make external
calls. No live Arcana request is required when the saved beta CSV is supplied.

## Replay frozen predictions

The final presentation matches the local `arcana_beta_hybrid_rebuild` report.
It reused the separately held `annual_turnover_400pct_20261028_v2` forecast
checkpoint. Its directory/archive is preserved locally under
`local/artifacts/final_v3/checkpoints/`; it is excluded from the public export.
Extract the archive if only a `.tar.gz` is present.

With that directory and the authorized inputs available:

```bash
python final_v3/main.py --from-predictions local/artifacts/final_v3/checkpoints/annual_turnover_400pct_20261028_v2 --replay-with-local-reporting --no-discovery --device cpu --data-dir data --reporting-file external_data/external.parquet --arcana-beta-file data/arcana/arcana_spy_beta_previous_holdings.csv --max-abs-beta 0.025 --output-dir final_v3/replay_output
```

The command rebuilds construction/reporting without refitting forecasts.
The supplied reporting container must include the required FRED data.
`--replay-with-local-reporting` verifies archived forecast payloads and
competition input hashes while recording the local reporting file separately.
It does not reopen or reverify the original licensed training container.
This is the limitation recorded in the saved final replay.

For strict original-input replay, supply the original external/reporting
containers and omit `--replay-with-local-reporting`. For reporting from an
already saved allocation, use `--from-weights CHECKPOINT` instead of
`--from-predictions`. Checkpoints reject payload corruption, incompatible
periods/modes and checkpoint/output overlap. See
[checkpoint behavior](../final_v3/docs/CHECKPOINTS.md).

## What can be checked without source data

- Compound the 68 monthly strategy returns and compare cumulative return,
  CAGR, annual returns and month-end drawdown with the saved summary.
- Compare presentation headline figures to `performance.json` and
  `comparison_metrics.csv`.
- Inspect annual turnover and the original compliance/reconciliation receipts.
- Inspect source code, file provenance and the separation of final results
  from earlier experiments.

These checks do not reconstruct model forecasts, verify point-in-time input
availability, resolve security mappings or certify daily volatility. Exact
evidence and file hashes are in [the final results folder](../results/final_strategy/README.md).

Existing regression tests for the earlier implementation are retained with
that implementation under `archive/initial_strategy/tests/`. They are not a
full final-v3 integration or compliance test suite. One-off cleanup validation
scripts stay in the system temporary directory.
