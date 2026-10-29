# Designated main strategy: no-agent Arcana-beta hybrid

This is the strategy label for the current Arcana-beta-weighted portfolio
construction in `final_v3`. The baseline does not run LLM research/scouting
agents. The agent-enabled variant starts from the same established signal and
construction recipe, then runs the discovery implementation; accepted
discovery proposals can change the signal stack. The saved six-year discovery
record admitted no proposals, so discovery did not alter that historical
baseline.

This label describes the saved **frozen-prediction replay** in
`arcana_beta_hybrid_rebuild/`. `main.py` now defaults to this no-agent mode,
Arcana input, and 2.5% beta limit; the command below repeats those values
explicitly for auditability. The Arcana upload CSV is a draft handoff; this
document describes the construction recipe, not Arcana upload readiness or a
full compliance certificate.

## Reproduce the saved no-agent construction

From the repository root, use the frozen prediction checkpoint and pass the
beta inputs and sizing limit explicitly:

```powershell
python final_v3/main.py `
  --from-predictions final_v3/checkpoints/annual_turnover_400pct_20261028_v2 `
  --replay-with-local-reporting `
  --no-discovery `
  --device cpu `
  --data-dir data `
  --reporting-file final/external_data/external.parquet `
  --arcana-beta-file arcana_upload/sp500_beta_005/arcana_spy_beta_previous_holdings.csv `
  --max-abs-beta 0.025 `
  --output-dir final_v3/current_main_output
```

This reuses archived forecasts and rebuilds portfolio sizing/reporting; it does
not retrain the stock or graph models. `--no-discovery` disables agent
research/scouting. `--arcana-beta-file` and `--max-abs-beta` are deliberately
explicit so the run record shows the intended inputs and bound. The replay
requires the checkpoint and competition inputs to pass the runtime's hash
checks. The local reporting option is explicit because the original
reporting/training container is not reopened in this replay mode. Use
`--with-discovery` for a fresh agent-enabled forecast run; discovery proposals
cannot be retrofitted onto a frozen no-agent prediction checkpoint. Use
`--no-arcana-beta` for a local-beta-only experiment.

## Construction map

| Stage | Current implementation |
|---|---|
| Signal forecasts | Frozen forecasts from `annual_turnover_400pct_20261028_v2`; no training or graph preparation in this replay. The signal forecasts and signal ranking are preserved. |
| Security selection | Existing `final_v3/optimizer.py`: 100 longs and 100 shorts; sector-balanced name selection (`SECTOR_NEUTRAL`); retain incumbents within a 0.5 rank buffer; $5 minimum price; ADV participation limits of 5% long and 2% short over five execution days; short market cap floor $500m; short-interest guard retained. |
| Beta inputs | At each formation, load the supplied Arcana-US model 3 stock beta versus SPY asset 21 when a valid prior-month observation is present. The input is joined by holding month and PERMNO after the Arcana asset mapping has been checked. |
| Beta fallback | For missing Arcana rows, use the local trailing 252-session S&P 500 excess-return regression beta. The local daily and monthly regressions (36 months) remain separate constraints. Their minimum observations are 126 daily pairs and 12 monthly pairs; short histories use formation-sector, then cross-sectional median priors. |
| Beta sizing constraint | The deployment and turnover sizing solvers constrain the local daily estimate, local monthly estimate, and the hybrid Arcana/local SPY beta to absolute value at most 0.025 at formation. The post-sizing guard checks the same columns. This is a portfolio-weight constraint; Arcana does not run the optimizer. |
| Arcana beta coverage | The saved run reports 8,752 holdings with Arcana beta and 4,848 with local daily-beta fallback, out of 13,600 month/security observations. Therefore the third beta constraint is hybrid, not Arcana-only. |
| Other sizing rules | 5% maximum Treasury/cash weight at rebalance; gross/net, sector exposure, position size, short eligibility, liquidity, and capacity checks remain active. Calendar-year one-way turnover is capped at 400%, initial deployment is exempt, and forced exits count. The 5 percentage-point forced-exit reserve is a turnover-budget reserve. |

The Arcana beta value is one input to final sizing, alongside the two local
regression constraints. It is not used to retrain or rerank the alpha forecasts.
All beta estimates are formation-time constraints; positions and estimated
portfolio beta can drift between monthly rebalances.

## Saved evidence and limitations

- `arcana_beta_hybrid_rebuild/optimizer_config.json` records the parameter
  values, Arcana input path/hash, beta-source counts, and portfolio-level
  turnover/net diagnostics.
- `arcana_beta_hybrid_rebuild/beta_estimation.json` records local regression
  windows, minimum observations, benchmark, risk-free-rate timing, and fallback.
- `arcana_beta_hybrid_rebuild/replay_provenance.json` records the frozen
  checkpoint and the reporting-input caveat. The forecast payloads were
  verified; the file explicitly does not claim complete input verification.
- `arcana_beta_hybrid_rebuild/report/compliance.json` reports the 68 monthly
  formation checks. Its overall `all_pass` is false because daily held returns
  are incomplete and daily/monthly NAV reconciliation fails, leaving realized
  volatility unavailable. Do not label this run as fully submission-ready.
- The generated Arcana handoff is `../arcana_upload/sp500_beta_005/`; its
  manifest marks the draft as unresolved for six historical Coherent mappings.

The agent-enabled counterpart uses the same underlying baseline rules and adds
the `discovery` proposal/scouting path in `final_v3/discovery/`. It is activated
through `config.DISCOVERY`; proposals enter a book only if they pass the
documented pre-year validation gate. The no-agent current-main command above
turns that path off. Do not compare a newly admitted agent sleeve as if it were
the same signal portfolio; it is a deliberate signal change on top of this
construction baseline.
