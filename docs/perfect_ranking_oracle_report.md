# Perfect-Ranking Oracle: Upper-Bound Backtest

## Purpose

This note answers the question: how much could the portfolio make if we cheated
and knew each stock's next-month return in advance? The result is an ex-post
oracle benchmark, not a tradable strategy. It is useful for checking whether the
current model is close to the mechanical ceiling implied by the portfolio rules.

## Method

- Evaluation window: January 2021 through August 2026, or 68 target months.
- Input: the existing stock-month panel in `output/preds_xgb_finbert.csv`.
- Future information used deliberately: `ret_exc_lead1m`, the realized next-month
  excess return attached to each characteristic row.
- For every month, sort stocks by that realized return.
- Go long the top 100 stocks and short the bottom 100 stocks.
- Equal-weight each leg, producing 100% long exposure, 100% short exposure,
  200% gross exposure, and 0% net exposure.
- Monthly oracle return: average realized return of the long leg minus average
  realized return of the short leg.

This is the same top-100/bottom-100 portfolio construction used by the current
backtests, with the prediction column replaced by the realized outcome.

## Results

The panel contains 268,733 eligible stock-month observations in the evaluation
window. The oracle produced the following results:

| Metric | Perfect-ranking oracle |
|---|---:|
| Average monthly long-short return | 132.8729% |
| Arithmetic annualized return | 1,594.4747% |
| Annualized Sharpe | 13.9783 |
| Positive months | 100.0% |
| Annualized long-leg average | 1,000.7922% |
| Annualized short-leg average | -593.6825% |
| Best month | November 2024: 220.7100% |
| Worst month | July 2021: 79.5157% |

The compounded total is astronomically large because the oracle repeatedly
selects extreme winners and crash names. That number is mathematically valid for
this backtest but economically meaningless as a live return estimate.

## Comparison with existing portfolios

The strongest existing result in the repository is the equal-weight TimesFM
portfolio. Metrics below use the same 68-month window and monthly long-short
returns.

| Portfolio | Avg. monthly | CAGR | Cumulative | Sharpe | Positive months |
|---|---:|---:|---:|---:|---:|
| Perfect-ranking oracle | 132.8729% | 2,271,285.7% | Astronomical | 13.9783 | 100.0% |
| TimesFM equal-weight | 3.0683% | 37.4345% | 506.0975% | 1.2409 | 73.53% |
| TimesFM volatility-scaled | 2.8148% | 33.3176% | 410.1472% | 1.1334 | 73.53% |
| XGBoost default | 1.1523% | 7.5162% | 50.7831% | 0.4138 | 63.24% |
| XGBoost + FinBERT | 0.8795% | 4.1573% | 25.9626% | 0.3179 | 60.29% |

Using average monthly return, the oracle is about 43.3 times the current
TimesFM equal-weight result, a gap of approximately 129.8 percentage points per
month. The current strategy is therefore nowhere near the ex-post ceiling.

## Interpretation and caveats

The gap is expected: the oracle uses the answer being predicted and is allowed
to select the most extreme realized winners and losers every month. The result
also concentrates the short book in crashing microcaps, where delistings,
borrow availability, buy-ins, liquidity, price limits, and transaction costs
would materially reduce or eliminate the apparent return.

The oracle is a diagnostic only. It must not be presented as an out-of-sample
strategy, a forecast, or a performance claim. No TimesFM rerun or GPU was needed
for this analysis; the repository's existing outputs were sufficient. RunPod is
only needed if we regenerate the TimesFM forecasts or run a new GPU-intensive
experiment.
