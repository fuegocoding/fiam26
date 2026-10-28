# Evidence from the 2026-10-28 no-agent run

These are attribution diagnostics on the already observed 2021-01 to 2026-08
window. They are not newly selected out-of-sample strategies. All construction
alternatives hold the completed run's capital schedule fixed to isolate
eligibility and construction effects. They are not self-consistent alternative
capital paths. The broad diagnostic violates the actual capacity/short-floor
requirements and cannot be presented as a deployable $1bn book.

| Controlled construction | Net CAGR | IR vs T-bill+4% | One-way monthly turnover |
|---|---:|---:|---:|
| Broad top/bottom 100; no capacity/short-floor screen | 25.74% | 1.56 | 53.89% |
| Actual capacity and short-floor eligibility | 10.69% | 0.36 | 48.85% |
| Plus sector quota/count matching | 10.28% | 0.38 | 50.23% |
| Plus existing retention buffer | 10.72% | 0.44 | 37.64% |
| Plus existing beta sizing, before risk overlays | 10.80% | 0.47 | 37.80% |
| Actual book, including risk overlays | 8.55% | 0.20 | 33.92% |

Average rank IC falls from 0.1123 in the full scored universe to 0.0482 in the
common capacity-eligible universe. The latter averages about 765 stocks per
month. For the characteristics XGBoost component, IC falls from 0.1661 to 0.0427.
IC is a ranking diagnostic, not factor-adjusted or independently verified alpha.

The broad diagnostic short leg contributes +16.24% of NAV per year before costs;
after capacity/short-floor eligibility, its contribution is -0.13%. In the actual
book the short contribution is -0.18%. The principal research problem is finding
short signals that survive the investable universe, not merely stronger ranks
in unrestricted small stocks.

The retention buffer helps in this comparison; removing sector matching also
does not produce a clear IR improvement. The risk layers need a closer audit:

| Exposure diagnostic | Net CAGR | IR | Month-end drawdown |
|---|---:|---:|---:|
| Before-risk weights, 100% gross | 10.80% | 0.47 | -3.26% |
| Before-risk weights at actual average gross, constant | 9.77% | 0.38 | -2.81% |
| Before-risk weights at actual month-by-month gross | 8.70% | 0.23 | -2.93% |
| Actual full risk adjustments | 8.55% | 0.20 | -2.93% |

The average and monthly exposure paths in this diagnostic are taken from the
completed run. The constant average gross is an ex-post comparator, not a
parameter chosen for deployment. This suggests that exposure timing, more than
within-book name adjustments, warrants investigation on separate validation.

Removing the 5% graph tilt while keeping construction/risk logic produced 9.61%
CAGR and IR 0.38 versus the actual 8.55% and 0.20. The paired return-difference
HAC t-stat was 1.84. This single-window diagnostic does not justify automatically
removing the graph; keep it frozen while testing in pre-period or unseen data.

Actual pre-cost IR is 0.42 versus 0.20 after costs. Annual modeled trading and
borrow costs are 0.80% and 0.33% of NAV. The actual four-factor alpha is only
1.35% per year (HAC t 0.64), using the runtime's S&P plus panel-built size,
value and momentum controls.

## Priorities

1. **Repair and reconcile accounting first.** Daily held-return coverage is
   incomplete and every monthly ledger reconciliation currently fails. Do not
   tune risk or certify the 12% volatility ceiling from that daily series.
2. **Research alpha inside the actual eligible universe, especially shorts.**
   Evaluate long and short spreads separately after the $500m short floor and
   actual ADV/participation constraints. Examine the membership/size biases
   behind broad-universe IC and use cost-adjusted spreads rather than IC alone.
3. **Audit exposure timing on separate validation.** Isolate ES scaling, edge
   throttling and within-book short adjustments. Compare matched risk budgets
   and preserve the mandate constraints. The table above is a diagnosis, not
   permission to turn risk controls off to improve historical IR.
4. **Develop turnover-aware selection using fixed archived forecasts.** The
   existing retention buffer helps; test expected return versus marginal
   transaction/borrow cost on training/validation data before new model work.
5. **Spend compute on the CPU preparation path and reuse, not a larger neural
   model.** The measured full graph stage was 31m51s; its neural fitting plus
   inference was only 2m03s. Cached prediction/weight replay skips those stages.
   Cache invariant risk features across NAV passes; retain strict content checks.

Do not choose final weights or graph/risk parameters by maximizing the already
observed 2021-2026 results. Use a pre-period protocol and genuinely unseen
follow-up evidence. Relevant primary research:

- [Frazzini, Israel and Moskowitz: Trading Costs of Asset Pricing Anomalies](https://www.aqr.com/Insights/Research/Working-Paper/Trading-Costs-of-Asset-Pricing-Anomalies)
- [Bailey et al.: The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

Detailed local artifacts are in `../output/research_audit/`: construction
ladder, matched-exposure decomposition, signal diagnostics, turnover definition,
cache validation and replay validation. Canonical run outputs were preserved.
