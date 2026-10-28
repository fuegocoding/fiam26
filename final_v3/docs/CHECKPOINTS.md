# Saved model weights and fast historical replay

The latest valuation-enabled snapshot with a 5% Treasury cap and a strictly
below-400% calendar-year one-way turnover cap (initial funding exempt; all
forced exits included; no monthly ceiling) is
`final_v3/checkpoints/annual_turnover_400pct_20261028_v2/`. Substitute that path in
the replay commands below for the latest book. It also preserves the fresh
valuation pricing forecasts and selection cache, so the historical scores can
be rebuilt without refitting valuation. Existing neural and other XGBoost
weights are preserved. See [updated results](ANNUAL_TURNOVER_400PCT_RERUN.md).
The previous monthly-cap snapshot remains at
`final_v3/checkpoints/turnover_30pct_reserved_20261028/`.
The preceding snapshot without the hard turnover cap remains at
`final_v3/checkpoints/treasury_5pct_20261028/`.
The prior Treasury-sweep snapshot remains at
`final_v3/checkpoints/valuation_treasury_20261028/`.

Replays use the current cash-accounting implementation. The original micro-off
snapshot retains its original report files; regenerating that report under
the new Treasury policy changes its returns, even with frozen stock weights.
The replay timings and exact-match checks below were measured before this
cash-policy change.

The completed 2026-10-28 no-agent A100 run is archived locally at
`final_v3/checkpoints/runpod_20261028_no_agents/`. This directory is outside
`output/`, so a fresh rebuild does not clear it. It is excluded from Git.

The snapshot contains:

- 36 PyTorch state dictionaries: two architectures, three seeds, six years.
- 24 characteristics XGBoost models and 24 filing-event XGBoost models.
- Historical stock predictions, component scores and graph forecasts.
- All 68 months of portfolio allocations: `portfolio.parquet`,
  `portfolio_weights.csv` and the percentage-weight submission CSV.
- `last_month_weights.csv`, explicitly dated to the last historical month.
- Content hashes, input hashes, mode, model code and original run provenance.

The filing-event boosters were not persisted by the earlier runner. They were
recreated using the same data, parameters and seeds on the A100; the resulting
270,149 forecasts matched the completed run exactly. The recovery receipt is
included. Future fitted filing-event models are saved directly by the runtime.

`weight_fraction` is a fraction of NAV; `weight_percent` is percent of NAV.
These are historical allocations, not instructions for a new live month.

## Replay the saved allocations and regenerate reports

From the repository root, using the input locations on this machine:

```bash
python final_v3/main.py --from-weights final_v3/checkpoints/runpod_20261028_no_agents --device cpu --no-discovery --data-dir data/raw/final --external-file final/external_data/external.parquet
```

This runs reporting with the original allocations. It performs no model fitting,
graph construction or portfolio optimization. The default destination is
`final_v3/replay_output/`; the canonical completed run in `output/` stays intact.

## Rebuild construction from saved stock predictions

```bash
python final_v3/main.py --from-predictions final_v3/checkpoints/runpod_20261028_no_agents --device cpu --no-discovery --data-dir data/raw/final --external-file final/external_data/external.parquet
```

This reruns construction, chronological NAV settlement and reporting. It skips all graph
preparation, XGBoost fitting and neural-network fitting. It is the appropriate
path for controlled construction experiments on the fixed historical forecasts.

Use `--output-dir PATH` for a separate result directory. Checkpoint/output
overlap is rejected. Explicit nonempty output directories without a recognized
run marker are not cleared. The checkpoint validates every archived payload and
input by content hash, so moving identical data does not invalidate it.

Changed data, incompatible modes and a different evaluation period are rejected.
A checkpoint does not extend forecasts to unseen months; those require valid
new features and the appropriate walk-forward models.

## Archive a future completed run

```bash
python final_v3/main.py --save-checkpoint final_v3/checkpoints/NEW_RUN_NAME --no-discovery --data-dir data/raw/final --external-file final/external_data/external.parquet
```

Use a new checkpoint name. Export refuses to overwrite an existing checkpoint.
Normal full runs now record input-content provenance when saving predictions.
Explicit checkpoint replay does not modify or relabel the original broad
`--reuse` training manifest.

## Chronological NAV settlement and historical timings

Current construction settles each holding month before sizing the next one.
Dollar liquidity limits use capital compounded through prior net returns.
An independent ledger verifies the complete formation-capital schedule at
relative tolerance 1e-12. This replaces repeated whole-path NAV iterations.
The measurements below describe the preceding iterative implementation.

Portfolio construction now caches the monthly peer-squeeze probabilities and
edge-throttle features that are independent of portfolio weights. The cache is
scoped to panel path/size/mtime, the actual score frame, and the relevant risk
parameters. Changed scores or feature parameters invalidate it. Book-dependent
expected shortfall, selected-name risk adjustments and holdings remain freshly
computed. Validation on all 68 months reproduced the original weights, and cold
versus warm cache results were identical.

Measured local CPU validation: saved-allocation/report replay took **33.5 s**;
construction/report replay from saved predictions took **173.8 s**. Both matched
the completed run's holdings, monthly returns and headline metrics. Warm cached
portfolio passes took **15.0 s** versus **36.4 s** uncached (2.43x faster).
Corrupt payloads, changed input contents, escaping paths and checkpoint/output
overlap were rejected. All 36 neural parameter files and 48 XGBoost model files
were loaded successfully on CPU.
