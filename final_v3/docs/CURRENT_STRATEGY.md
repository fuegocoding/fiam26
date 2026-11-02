# Final strategy: fixed blend with hybrid beta sizing

The final presentation is [StochasticPirates.pdf](../../docs/presentation/StochasticPirates.pdf).
The matching local replay is identified in
[the final-result provenance](../../results/final_strategy/source.json), with
15.65% CAGR and 1.22 Sharpe over January 2021-August 2026.

Use [the strategy description](../../docs/strategy.md) for construction,
[the process notes](../../docs/research-process.md) for the agent experiments,
and [the reproduction guide](../../docs/reproducibility.md) for executable
commands. The final fixed baseline uses `--no-discovery`; the application's
agent-enabled default is retained.

The historical replay reused frozen forecasts from
`annual_turnover_400pct_20261028_v2`. It applied local daily, local monthly, and
hybrid Arcana/local SPY beta constraints of ±0.025 at formation. Arcana supplied
8,752 beta observations; 4,848 holdings used the local fallback, out of 13,600
month/security observations. This was portfolio resizing, not forecast retraining.

The saved beta input is now local-only at
`data/arcana/arcana_spy_beta_previous_holdings.csv`. Full checkpoints and
original workspaces are local-only under `local/artifacts/final_v3/`.

The saved compliance receipt fails daily held-return completeness and
daily/monthly NAV reconciliation. It cannot certify the 12% daily volatility
ceiling. Historical security mappings and the point-in-time availability of
exported Arcana betas also require verification. See
[the published saved checks](../../results/final_strategy/README.md).

[The original replay notes](../../archive/research/arcana_hybrid_replay_notes.md)
are preserved as historical context, including their original paths.
