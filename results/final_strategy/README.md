# Final strategy evidence

This is a selected copy of the **aggregate report** from the saved
`arcana_beta_hybrid_rebuild` replay. Its 15.65% CAGR, 1.22 Sharpe, 0.80 IR and
-10.96% monthly drawdown match the final
[StochasticPirates presentation](../../docs/presentation/StochasticPirates.pdf).
The window is January 2021-August 2026.

| File | Meaning |
| :--- | :--- |
| [source.json](source.json) | Source context, limitations and SHA-256 hashes |
| [performance.json](performance.json) | Original saved summary, including risk/cost metadata |
| [comparison_metrics.csv](comparison_metrics.csv) | Strategy, T-bill + 4% and S&P 500 comparisons |
| [returns.csv](returns.csv) | 68 monthly net strategy returns; `Date` is the holding month's first day |
| [calendar_years.csv](calendar_years.csv) | Compounded annual and partial-year returns |
| [monthly_detail.csv](monthly_detail.csv) | Aggregate monthly P&L, costs, exposures and turnover |
| [calendar_year_turnover.csv](calendar_year_turnover.csv) | One-way annual budget accounting, with initial funding separated |
| [standardised_performance.csv](standardised_performance.csv) | Presentation-oriented measures and units |
| [contributors.csv](contributors.csv), [top_average_holdings.csv](top_average_holdings.csv) | Team portfolio attribution and average holdings |
| [compliance.json](compliance.json), [part2_status.json](part2_status.json) | Original saved runtime checks |
| [daily_monthly_reconciliation.csv](daily_monthly_reconciliation.csv) | Aggregate evidence of the daily/monthly NAV disagreement |
| [charts/](charts/) | Saved cumulative, drawdown, distribution, rolling IR and contributor charts |

Numbers expressed as fractions use `0.01 = 1%`. Read field definitions before
comparing CAGR, arithmetic returns, annualized turnover and leg P&L. The
`calendar_years.csv` legacy `month` column contains the calendar year; its
`active` column compounds monthly relative returns rather than subtracting
the two annual returns.

Receipts are preserved as saved. They report `all_pass: false` and
`submission_ready: false` because held daily returns are incomplete and daily
NAV does not reconcile with the monthly ledger. Some pending-task strings
describe the run's original reporting state, before the final presentation.
They are not a statement that the hackathon is still ongoing.

Raw stock-return panels, per-security prediction caches, model weights,
source beta exports and original machine-specific run receipts are excluded.
This directory supports aggregate verification; it is not a complete replay
checkpoint or a fresh independent performance audit.
