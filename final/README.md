# FIAM 2026 market-neutral book (`final/`)

Self-contained pipeline that scores U.S. stocks each month, builds a sector-
and beta-neutral 100-long / 100-short book, and applies the risk rules.
Nothing outside this folder is imported.

## Run

```bash
# Full pipeline: sleeves, graph model, blend, then the book.
HF_HOME=local/hf_cache PYTHONPATH=.deps local/timesfm-venv/bin/python \
    final/combiner.py --work-dir output/final

# Rebuild only the book from saved predictions (e.g. after changing NAV).
python3 final/optimizer.py --predictions output/final/predictions.parquet
```

Fund size is the `NAV` constant at the top of `optimizer.py` (default $10M).
Every dollar size and liquidity limit derives from it.

## Layout

| Path | Role |
|---|---|
| `combiner.py` | Runs the nine sleeves and the graph model, blends the scores, calls the optimizer |
| `optimizer.py` | Picks 100 longs and 100 shorts (sector quotas, rank buffer, liquidity limits, beta-neutral sizing), then calls `risk.protect` each month |
| `risk.py` | Risk rules; one public function, `protect` |
| `sleeves/` | The return signals (one file, or one folder with an abstraction, per sleeve) |
| `graph/` | Point-in-time economic graph and EPS model used as a 5% overlay |
| `council/` | Placeholder for the event council hook |

## Risk rules (`risk.py`, all on by default)

| Rule | What it does |
|---|---|
| Peer-EVT squeeze sizing | A GPD fitted to three years of industry x size peer returns gives each stock's chance of a +50% month; shorts in the riskiest 5% get half weight, freed weight goes to the sector's other shorts (capped at 1.5x an average short) |
| Run-up cap | A short that rallied last month is cut so its exposure stays within 1.5x target. Monthly in the backtest; live it runs daily at the close |
| Expected-shortfall limit | The book is replayed on its holdings' last 36 monthly returns; if its 95% expected shortfall is above the median the same book showed over the previous 24 month-ends, gross is cut in proportion (never up, never below half) |
| Edge throttle | Gross scales from 50% to 100%, in quarter steps, with the t-statistic of the last 12 months' realized top-minus-bottom 5% score spread |
| Own-EVT diagnostic | A GEV on each short's monthly maxima of daily returns gives its 1-in-100-month jump, reported in `portfolio.parquet` |

Every rule is a constant at the top of `risk.py` and can be switched off
there.

## Results by fund size

Out-of-sample test period January 2021 to August 2026 (68 months). Monthly
rebalancing, all risk rules on, 100 longs and 100 shorts every month. Returns
are on NAV and net of trading costs (5 to 60 bp per unit traded by size tier)
and borrow fees (25 to 1,500 bp a year by size tier); capital earns the
3-month T-bill. A stock with no next-month return exits at its last price.

| Metric | $10M | $100M | $250M | $500M | $750M | $1B |
|---|---|---|---|---|---|---|
| Information ratio vs T-bill + 4% | **1.53** | **0.87** | **0.78** | **0.52** | **0.19** | **0.11** |
| Sharpe ratio (excess of T-bill) | 2.12 | 1.44 | 1.40 | 1.16 | 0.86 | 0.89 |
| CAGR (compounded annual return) | **18.8%** | **14.0%** | **12.8%** | **10.8%** | **8.6%** | **8.0%** |
| Cumulative return (Jan 2021 – Aug 2026) | 166.1% | 110.0% | 97.8% | 78.9% | 59.2% | 54.4% |
| Annualized active return (vs T-bill + 4%) | 10.3% | 6.1% | 5.0% | 3.2% | 1.1% | 0.5% |
| Annualized volatility | 6.7% | 6.9% | 6.4% | 6.1% | 5.9% | 5.0% |
| Max drawdown | -3.5% (May 2026) | -4.2% (Jul 2025) | -3.2% (Jan 2023) | -3.0% (Aug 2022) | -4.0% (Jul 2025) | -3.1% (Jul 2025) |
| Worst / best month | -2.9% / +6.1% | -3.0% / +6.5% | -3.2% / +6.5% | -2.5% / +5.9% | -3.1% / +5.6% | -2.5% / +5.1% |
| Months beating the hurdle | 63% | 57% | 63% | 56% | 47% | 46% |
| Alpha vs S&P 500 (CAPM) | 15.5%/yr (t 5.1) | 11.1%/yr (t 3.5) | 9.9%/yr (t 3.8) | 8.1%/yr (t 3.4) | 6.1%/yr (t 2.8) | 5.5%/yr (t 2.9) |
| Beta vs S&P 500 | -0.11 (SE 0.064) | -0.10 (SE 0.062) | -0.08 (SE 0.057) | -0.09 (SE 0.048) | -0.09 (SE 0.050) | -0.09 (SE 0.043) |
| Correlation with S&P 500 | -0.25 | -0.21 | -0.19 | -0.21 | -0.23 | -0.27 |
| 4-factor alpha (market, size, value, momentum) | 7.7%/yr (t 3.1) | 0.9%/yr (t 0.3) | 1.4%/yr (t 0.5) | 0.8%/yr (t 0.3) | -1.2%/yr (t -0.5) | -0.6%/yr (t -0.3) |
| 4-factor loadings (mkt / size / value / mom) | -0.06 / -0.27 / +0.24 / +0.15 | -0.02 / -0.28 / +0.29 / +0.22 | -0.01 / -0.19 / +0.22 / +0.20 | -0.02 / -0.19 / +0.18 / +0.18 | -0.02 / -0.17 / +0.16 / +0.19 | -0.03 / -0.12 / +0.14 / +0.16 |
| Trading costs / borrow fees per year | 2.1% / 1.8% | 1.6% / 1.2% | 1.4% / 0.9% | 1.1% / 0.6% | 0.9% / 0.5% | 0.9% / 0.4% |
| Gross exposure, average (range) | 81% (25%–100%) | 80% (25%–100%) | 82% (25%–100%) | 81% (25%–100%) | 82% (25%–100%) | 83% (25%–100%) |
| Net exposure, average (range) | +3.2% (-3.6% to +14.6%) | +3.5% (-2.6% to +12.2%) | +3.5% (-2.8% to +10.6%) | +3.5% (-2.2% to +11.7%) | +3.6% (-3.8% to +10.8%) | +3.7% (-1.9% to +9.8%) |
| Monthly turnover (of NAV) | 76% | 73% | 72% | 68% | 66% | 66% |
| Holdings per month (long / short) | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| Position size, average / max (% NAV) | 0.40% / 0.76% | 0.40% / 0.72% | 0.41% / 0.76% | 0.41% / 0.66% | 0.41% / 0.56% | 0.42% / 0.59% |
| Micro/nano share of short positions | 46% | 21% | 11% | 5% | 3% | 2% |

| Year | $10M | $100M | $250M | $500M | $750M | $1B | Hurdle (T-bill + 4%) |
|---|---|---|---|---|---|---|---|
| 2021 | 12.7% | 15.2% | 13.4% | 12.6% | 10.6% | 8.1% | 4.1% |
| 2022 | 35.5% | 26.4% | 18.9% | 15.9% | 15.5% | 14.2% | 6.2% |
| 2023 | 19.5% | 7.6% | 8.7% | 3.0% | 0.7% | 1.9% | 9.5% |
| 2024 | 22.8% | 17.6% | 17.7% | 16.9% | 12.7% | 12.7% | 9.3% |
| 2025 | 16.0% | 10.8% | 11.3% | 10.1% | 6.4% | 6.2% | 8.4% |
| 2026 (to Aug) | 2.4% | 2.9% | 3.1% | 3.5% | 3.2% | 2.7% | 5.2% |

### Reading the table

- **Capacity.** A position must be tradable within five days at 5% (longs)
  or 2% (shorts) of daily dollar volume. As NAV grows, small and illiquid
  names drop out (micro/nano shorts fall from 46% to 2%) and the book drifts
  to large caps, where the signals are weaker.
- **Stock-specific alpha exists only at $10M.** From $100M up the 4-factor
  alpha is indistinguishable from zero: what remains is the book's tilt to
  value, momentum and away from small caps. The $10M book is the headline;
  the larger sizes are its capacity curve.
- **Larger sizes are, if anything, optimistic.** Costs are a fixed rate per
  unit traded by size tier and do not grow with order size, so the extra
  market impact of trading millions per name is not charged.
- **Beta** is not statistically different from zero at any size (|t| < 2).
- **Exposure.** Gross is at most 100% of NAV against the 200% limit; the
  risk rules hold the remainder as cash when tail risk is high or the
  signal's recent edge is weak.

Metrics were computed with the evaluation harness in
`output/time_series_overnight_20261001/` (`lab.evaluate`), from books built by
`optimizer.optimize` at each NAV on the official predictions
(`output/time_series_final_e2e_20261001/predictions.parquet`). The table is
also saved as `metrics_by_nav_all.csv` there.
