# Treasury 5% and monthly total turnover 30% - 2026-10-28

Uses the existing valuation-enabled stock/graph forecasts with no agents or training. Rebuilds construction and accounting chronologically, using only preceding settled returns to determine pretrade holdings and fund capital.

## Constraints

- One-way rebalance turnover is half the sum of absolute signed trades, divided by current NAV. It includes exits, entries, resizing, sign changes and drift. The first deployment is exempt, as explicitly requested; all later months have a 30% total cap. Scheduled trades prefer a 25% budget, leaving five percentage points for involuntary exits. If that is infeasible, the solver minimizes necessary trades within the 30% hard limit, preserving as much headroom as possible. This is a reserve of trading capacity, not cash.
- Own-capital Treasury stays at or below 5% at each monthly rebalance. Short-sale collateral is separate.
- Exactly 100 longs and 100 shorts; gross at most 200%, absolute net at most 50%, positions at most 5%, sector net at most 20%, at least five sectors, and the $500m short market-cap floor.
- The original formation eligibility, monthly schedule, rank buffer, ADV capacity, peer-squeeze and run-up bounds remain. Membership and weights may depart from the preferred target to meet the turnover budget. The projection preserves both preferred long/short notional budgets and sector name counts.
- As in the preceding 5% Treasury run, the hard cash ceiling overrides global ES/edge reductions to cash; their diagnostics remain available.

## Results

See [monthly metrics and standalone pod breakdown](MONTHLY_POD_METRICS.md)
for CAGR, Sharpe, rank IC, drawdown and alpha. Dynamics has no full-period
standalone return series under the unchanged constraints: its February 2021
rebalance requires about 32.91% one-way turnover, above the 30% cap.

January 2021-August 2026. Net of modeled trading and borrow costs. Sharpe uses monthly returns above the reporting T-bill rate.

| Metric | Prior 5% Treasury run | Turnover-constrained run |
|---|---:|---:|
| Net CAGR | 18.25% | 13.64% |
| Sharpe | 1.13 | 0.86 |
| Information ratio | 0.82 | 0.52 |
| Maximum month-end drawdown | -8.79% | -12.77% |

Initial deployment one-way turnover: 88.58%. Subsequent scheduled rebalances average 25.0210%, maximum 25.8408%.

Maximum Treasury: 5.0000%. Mean long/short: 95.00% / 86.21%. Maximum gross: 196.82%; maximum absolute net: 27.81%. All 68 months retain 100 longs and 100 shorts.

## Forced liquidations and total turnover

The existing missing-return policy liquidates a holding at its last dated quote within the holding month. That future missing-return event is not known at the preceding formation date. It is not used to pre-screen candidates or retroactively shrink the rebalance budget. The preferred five-percentage-point reserve and formation-only fallback are fixed in advance; the runtime rejects any completed month whose total turnover still breaches 30%.
These involuntary liquidations add an average 0.0598% one-way turnover after initialization. Including them, total turnover after initialization averages 25.0808%, reaches 26.2485%, and exceeds 30% in 0 months. The 30% total realized turnover cap passes in every subsequent month. The report records both checks separately.

## Annual results

2026 covers January-August. Rebalance turnover averages below exclude the initial deployment.

| Year | Net return | Sharpe | Mean Treasury | Mean short | Mean rebalance turnover | Mean total turnover including initial |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | 29.80% | 1.77 | 5.00% | 86.50% | 25.00% | 30.40% |
| 2022 | 31.10% | 2.01 | 5.00% | 86.48% | 25.12% | 25.17% |
| 2023 | 1.72% | -0.43 | 5.00% | 88.22% | 25.00% | 25.04% |
| 2024 | 7.77% | 0.35 | 5.00% | 93.97% | 25.00% | 25.00% |
| 2025 | -0.40% | -0.40 | 5.00% | 84.30% | 25.00% | 25.00% |
| 2026 | 11.06% | 0.86 | 5.00% | 73.59% | 25.00% | 25.21% |

## Verification and preservation

Independent full-ledger replay matched every monthly net return within absolute tolerance 1e-12. Separately reconstructed pretrade drift and trades matched rebalance turnover within 1e-12. Actual compounded capital, name counts, ADV bounds, short safety caps, Treasury, gross/net and sector bounds were checked.

Mixed-integer optimization chooses membership within the turnover budget, then an LP polishes the weights with tight feasibility tolerances. The objective minimizes deviation from the preferred target; a time-limited feasible solution is accepted only after every hard constraint is checked.

Existing daily held-return gaps and daily/monthly reconciliation failures remain. The 12% daily volatility ceiling is not certified and submission_ready remains false. Exposure and Treasury limits are checked at monthly rebalances; daily price drift is not constrained by this replay.

Output: final_v3/turnover_30pct_reserved_output/. Checkpoint: final_v3/checkpoints/turnover_30pct_reserved_20261028/. The verified archive preserves 36 neural models, 24 characteristic XGBoost models, 24 filing XGBoost models, valuation caches, forecasts, allocations, source and configuration. Previous runs remain preserved.

Archive SHA-256: 1be52666fc841b539a91dbbac23041382edc14c86d48892ba39b09b3528d2838
