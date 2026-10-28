# Treasury capped at 5% - 2026-10-28

The Treasury cap is enforced at monthly rebalances. Longs receive 95% of NAV; the short target follows the beta hedge within the gross/net constraints. Short proceeds remain restricted collateral, separate from the own-capital Treasury sweep. Daily price drift can move exposures between rebalances.

No agents or model fitting were used. The valuation-enabled stock and graph forecasts were reused. The old run and checkpoint are preserved.

## Results

| Metric | Previous Treasury-sweep book | 5% Treasury cap |
|---|---:|---:|
| Net CAGR | 9.10% | 18.25% |
| Sharpe | 0.97 | 1.13 |
| IR vs T-bill + 4% | 0.28 | 0.82 |
| Maximum month-end drawdown | -4.12% | -8.79% |

One-way turnover averages 63.93% of NAV per month (767.21% annualised), versus 31.82% previously. The monthly schedule, 0.5 retention buffer, five-day execution window and 5% long / 2% short ADV participation limits are unchanged. Actual traded dollars increase when stock positions are enlarged; no artificial turnover cap was applied, per user instruction.

## Exposure checks

Maximum Treasury allocation at a rebalance: 4.999999%. Average long exposure: 95.00%; average short exposure: 86.49%. Maximum gross: 198.332312%; maximum absolute net: 30.21%.

All 68 months retain 100 longs and 100 shorts. Final positions are checked against 5% single-name, actual dollar ADV capacity and the $500m short floor. Sector breadth and +/-20% sector net limits remain enforced. Portfolio returns were independently recomputed and matched stored monthly returns to 1e-12.

## Annual results

2026 covers January-August. Returns are compounded net of trading/borrow costs. Sharpe is annualised from monthly returns in excess of the reporting T-bill rate.

| Year | Net return | Sharpe | Mean Treasury | Mean long | Mean short | Max gross | Mean monthly one-way turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 38.10% | 2.00 | 5.00% | 95.00% | 88.35% | 195.55% | 62.26% |
| 2022 | 31.85% | 2.07 | 5.00% | 95.00% | 87.23% | 189.34% | 62.82% |
| 2023 | 2.76% | -0.25 | 5.00% | 95.00% | 90.17% | 193.03% | 59.14% |
| 2024 | 16.40% | 1.48 | 5.00% | 95.00% | 93.08% | 198.33% | 65.82% |
| 2025 | 6.24% | 0.24 | 5.00% | 95.00% | 82.46% | 193.42% | 67.06% |
| 2026 | 11.74% | 0.81 | 5.00% | 95.00% | 73.21% | 182.32% | 67.79% |

## Risk policy and replay

A hard 5% Treasury ceiling supersedes the global ES/edge reductions to cash. Their diagnostic estimates remain reported. Individual peer-squeeze and run-up reductions still impose per-stock short bounds; capacity or sector infeasibility raises an error rather than relaxing the limits.

Months are constructed and settled in chronological order. Each month uses capital compounded through the preceding net return, including costs. An independent full-ledger replay verifies that every formation NAV matches within relative tolerance 1e-12; no whole-path convergence iteration is required.

The existing daily return gaps and daily/monthly reconciliation failures remain unresolved. Consequently the backtest does not have a certified daily volatility limit or submission-ready status.

Output: final_v3/treasury_5pct_output/. Checkpoint: final_v3/checkpoints/treasury_5pct_20261028/. Its archive contains the 36 existing neural models, 24 characteristic XGBoost models, 24 filing XGBoost models, valuation forecast caches, new historical allocations and source/configuration metadata.

Archive SHA-256: 63047df20b593254ab027e4ba1232fbf642a0b055f6b7f1f4337b18c5cd5b2fd
