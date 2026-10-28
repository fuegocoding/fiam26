# Valuation enabled and Treasury cash sweep - 2026-10-28

No agents were used. Both gap_micro_best (valuation) and runway_micro (fundamentals) are enabled at $1bn. Actual positions retain the original per-name liquidity checks and $500m short floor. No risk, cost, or signal parameters were selected using these rerun results.

Fresh current-v3 valuation training on the A100 took 519.18 seconds. Existing graph forecasts, 36 neural checkpoints and 48 other XGBoost models were reused. The old micro-off run and checkpoint remain unchanged.

## Full-period net results

January 2021 through August 2026. Each standalone pod has its own $1bn endowment and converged compounding capital path. The graph overlay is present only in the combined book. Standalone pod results are not additive attribution.

| Book | Net CAGR | Sharpe | IR vs T-bill + 4% | Max monthly drawdown |
|---|---:|---:|---:|---:|
| combined | 9.10% | 0.97 | 0.28 | -4.12% |
| fundamentals | 5.03% | 0.36 | -0.46 | -5.60% |
| valuation | 3.62% | 0.08 | -0.78 | -5.66% |
| events | 3.02% | -0.06 | -1.07 | -5.60% |
| dynamics | 4.16% | 0.14 | -0.36 | -14.90% |

The previous micro-off book, before the Treasury change, had CAGR 8.55%, Sharpe 0.95 and IR 0.20. Both micro sleeves and cash accounting changed in the new combined result; this comparison does not isolate either change.

## Calendar-year return (Sharpe)

Sharpe uses monthly net returns less the same-month reporting T-bill rate, annualised with sqrt(12) and sample standard deviation. 2026 is January-August: its return is not annualised; its Sharpe is annualised from eight observations.

| Year | Combined | Fundamentals | Valuation | Events | Dynamics |
|---|---:|---:|---:|---:|---:|
| 2021 | 11.20% (1.95) | 4.16% (0.83) | 5.57% (1.32) | 1.07% (0.32) | 1.29% (0.19) |
| 2022 | 16.73% (1.87) | 2.32% (0.08) | 7.54% (0.93) | 5.93% (0.89) | 14.81% (1.52) |
| 2023 | 2.98% (-0.47) | 4.06% (-0.20) | 2.70% (-0.61) | 2.41% (-0.85) | 2.26% (-0.53) |
| 2024 | 10.07% (1.17) | 13.03% (2.17) | -0.28% (-1.20) | -0.96% (-1.58) | 6.84% (0.39) |
| 2025 | 4.45% (0.09) | 11.58% (1.22) | 0.79% (-0.65) | 2.17% (-0.76) | -5.79% (-1.12) |
| 2026 | 6.60% (0.73) | -5.60% (-2.64) | 4.35% (0.61) | 6.64% (0.94) | 5.25% (0.36) |

## Combined book by stock market-cap group

The labels are the supplied formation-date size_grp labels. Contributions are equity price P&L less trading and borrow costs, expressed per total fund NAV. Annual NAV contribution is mean monthly contribution times 12; it is not a standalone CAGR or Sharpe. Cash accounts are separate.

| Group | Mean gross / NAV | Mean names | Annual net NAV contribution |
|---|---:|---:|---:|
| Financing | 0.00% | 0.0 | 0.000 pp |
| Short collateral | 0.00% | 0.0 | 1.350 pp |
| Treasury | 0.00% | 0.0 | 1.700 pp |
| large | 41.20% | 99.7 | 4.101 pp |
| mega | 28.72% | 68.4 | 1.817 pp |
| micro | 0.19% | 0.6 | -0.173 pp |
| nano | 0.00% | 0.0 | 0.000 pp |
| small | 12.08% | 31.2 | 0.108 pp |

## Pod contributions by market-cap group

Annual net NAV contribution (percentage points) within each standalone pod:

| Pod | Nano | Micro | Small | Large | Mega |
|---|---:|---:|---:|---:|---:|
| fundamentals | -0.174 | -0.206 | 1.395 | 0.517 | 0.167 |
| valuation | -0.072 | -0.369 | 0.192 | 1.279 | -0.437 |
| events | -0.020 | -0.264 | 0.771 | -0.346 | -0.165 |
| dynamics | 0.000 | -0.069 | -1.748 | 0.867 | 2.623 |

## Cash and verification

Unused own capital, max(1 - long notional, 0), is swept to a 3-month Treasury bill accrual proxy. The monthly rate is FRED TB3MS / 1200, lagged two calendar months so the monthly average was published before formation. Daily accrual geometrically matches the monthly rate. Short proceeds remain separate collateral earning the existing panel-implied rate; financing above own NAV is charged that rate. The proxy omits Treasury price changes, bill trading costs and broker margin terms. It replaces existing cash interest rather than counting it twice.

Every monthly cap/cash/cost contribution is checked against the total NAV return. Yearly cap contributions are geometrically linked and sum to the compounded annual return. The archived portfolios were independently reevaluated and matched the stored returns to 1e-12.

The daily held-return coverage and monthly/daily reconciliation checks remain authoritative: these monthly results do not constitute a daily-volatility or submission-readiness certificate.

Artifacts: final_v3/valuation_treasury_output/report/ and research_audit/. The full CSV breakdown includes each pod, cap group and year. The checkpoint is final_v3/checkpoints/valuation_treasury_20261028/.

FRED series definition: https://fred.stlouisfed.org/series/TB3MS
