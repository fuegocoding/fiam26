# Annual turnover below 400% - 2026-10-28

This run replaces the 30% monthly limit with a strictly-below-400% calendar-year one-way turnover ceiling. Initial funding is exempt; its transaction costs remain included in net returns. Every forced liquidation counts toward turnover. Monthly turnover may exceed 30% or 33.33%.

The turnover formula follows Part 2 Section 5.3: half the absolute change from carried, drifted positions, expressed per NAV. The calendar-year ceiling and initialization exception are the user-specified restrictions for this run.

## Portfolio and budget policy

Own-capital Treasury is capped at 5% at monthly rebalances, with 95% deployed to longs. Both preferred leg budgets are preserved. There are 100 longs and 100 shorts, gross at most 200%, absolute net at most 50%, per-name positions at most 5%, per-sector net at most 20%, and the $500m short floor. ADV capacity, eligibility, rank-buffer selection and individual peer/run-up protections remain unchanged. As in the prior 5% Treasury run, the cash ceiling overrides global ES/edge reductions to cash.

Only already settled turnover is charged before sizing the next month. Remaining annual budget is paced over the remaining calendar months through December, retaining preferred headroom for forced exits. If that preferred budget is infeasible, a formation-only minimum-trade projection can borrow from later monthly allocations. The realized annual total is checked again after settlement and a breach rejects the run. No future holding-month outcomes determine current trades.

January-August 2026 remains a partial calendar year. The controller retains budget for September-December; the end of the historical sample does not release that budget. Annualized observed-month statistics are provided separately from actual year-to-date sums.

## Combined results versus the prior monthly cap

| Metric | Prior monthly-cap run | Annual-cap run |
|---|---:|---:|
| Net CAGR | 13.64% | 16.17% |
| Sharpe | 0.86 | 1.05 |
| Information ratio | 0.52 | 0.71 |
| Maximum month-end drawdown | -12.77% | -9.00% |

Counted one-way turnover annualized over the 68 observed months: 391.43%. Maximum counted monthly turnover: 45.90%. Initial funding excluded from the budget: 88.58% one-way. Raw turnover including funding is also retained separately in performance.json.

Maximum Treasury 5.0000%; average long/short 95.00%/86.69%; maximum gross 195.82%; maximum absolute net 29.57%.

## Full-period standalone pod metrics

Each standalone pod starts with its own $1bn and compounds its own capital path under the same restrictions. These are not additive contributions to the combined book. Graph forecasts are retained in Combined only. No model fitting or agents were used.

| Book | CAGR | Sharpe | Mean rank IC | Common-eligible IC | Max DD | CAPM alpha/year | Four-factor alpha/year |
|---|---:|---:|---:|---:|---:|---:|---:|
| Combined | 16.17% | 1.049 | 0.108 | 0.050 | -9.00% | 14.52% | 1.89% |
| Fundamentals | 10.96% | 0.648 | 0.051 | 0.023 | -11.15% | 7.45% | 11.47% |
| Valuation | 5.82% | 0.231 | -0.008 | 0.026 | -21.95% | 5.47% | -11.37% |
| Events | 6.70% | 0.382 | 0.082 | 0.017 | -9.78% | 4.32% | -5.76% |
| Dynamics | 7.33% | 0.308 | 0.166 | 0.044 | -22.03% | 7.04% | -10.22% |

## Actual calendar-year one-way turnover

Initial funding excluded; all forced exits included. 2026 is January-August, not a full year.

| Year | Combined | Fundamentals | Valuation | Events | Dynamics |
|---|---:|---:|---:|---:|---:|
| 2021 | 395.00% | 395.00% | 387.04% | 395.00% | 385.39% |
| 2022 | 395.00% | 395.00% | 395.00% | 395.00% | 395.00% |
| 2023 | 395.00% | 395.26% | 392.38% | 395.00% | 395.00% |
| 2024 | 395.00% | 395.00% | 383.94% | 395.00% | 395.00% |
| 2025 | 395.00% | 395.29% | 341.64% | 395.00% | 395.00% |
| 2026 Jan-Aug | 243.13% | 243.65% | 224.12% | 243.55% | 243.30% |

## Monthly detail and definitions

See [all 68 months by book](ANNUAL400_MONTHLY_POD_METRICS.md) and final_v3/annual_turnover_400pct_output/research_audit/pod_metrics/monthly_metrics.csv. Each monthly row has actual net return and rank IC, plus trailing-12-month and expanding CAGR, Sharpe, drawdown and alpha. The first 11 months have no trailing-12-month performance statistics.

Rank IC is monthly cross-sectional Spearman correlation against observed next-month excess stock returns. Summary IC is the mean of monthly ICs. The common-eligible version uses the identical intersection of covered, liquid, long-and-short eligible stocks across all books.

CAPM alpha is the annualized intercept against S&P 500 price-index excess returns. Four-factor alpha additionally controls for the runtime panel-built size, value and momentum factors; these are not official Fama-French series. Drawdown is measured at month ends.

## Validation and saved assets

Independent return replay, annual turnover reconstruction, realized compounded capital, leg targets, ADV bounds, short safety caps, Treasury, name counts, sector and gross/net constraints were checked. The existing daily held-return gaps and daily/monthly reconciliation failures remain; daily volatility and submission readiness are not certified.

Latest verified checkpoint: final_v3/checkpoints/annual_turnover_400pct_20261028_v2/. It includes 36 neural models, 24 characteristic XGBoost models, 24 filing-event XGBoost models, valuation caches, historical forecasts and updated allocations. Source/configuration metadata and the annual budget diagnostics are preserved. The earlier v1 snapshot retains the original raw-turnover headline convention; v2 aligns the headline with the initial-funding exception and preserves the raw figure separately. Portfolio returns and weights are identical between those snapshots.

Archive SHA-256: 96b594e3ab1112d494ff3f85d07b4d40bfb32fd8f2a69eb4cf1ed3c0dcc83a28

The standalone pod allocations, forecasts and all metric tables are also preserved in final_v3/checkpoints/annual_turnover_400pct_pod_metrics_20261028.tar.gz. Archive SHA-256: 9d4eb3e74a76c74aca063fa27b05220b56f346bef36715eaf0899cb92be45277
