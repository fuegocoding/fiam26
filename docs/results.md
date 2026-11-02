# Final results

These figures describe the **local no-agent Arcana-beta hybrid replay**
presented in [StochasticPirates.pdf](presentation/StochasticPirates.pdf),
especially slides 2, 8, 12 and 14-18. The saved aggregate report agrees with
the presentation's headline metrics. Fourth place is the team's reported
competition finish; it is not a performance statistic from the backtest.

The window is January 2021-August 2026: 68 monthly observations, with
modeled trading costs and borrow fees deducted. Starting capital is $1 billion.
The hurdle is three-month T-bill + 4%; the S&P 500 comparison is the price
index without dividends.

## Performance

| Measure | Strategy | Hurdle | S&P 500 |
| :--- | ---: | ---: | ---: |
| CAGR | 15.65% | 7.53% | 13.47% |
| Cumulative return | 127.89% | 50.87% | 104.63% |
| Annualized volatility, monthly observations | 9.54% | 0.56% | 15.10% |
| Maximum drawdown, month-end NAV | -10.96% | 0.00% | -24.77% |
| Sharpe above T-bill | 1.22 | — | 0.70 |
| Information ratio vs hurdle | 0.80 | — | 0.44 |
| Best month | 8.92% | 0.78% | 10.42% |
| Worst month | -3.21% | 0.34% | -9.34% |

Ending strategy NAV is $2.279 billion. Monthly CAPM beta is -0.055
(standard error 0.079); annual CAPM alpha is 12.36% (HAC t = 3.11).
Four-factor alpha falls to 5.84% (HAC t = 0.94), so the evidence does not
support treating all CAPM alpha as distinct stock-selection skill. Size,
value and other factor exposures explain part of the return.

## Calendar-year returns

| Year | Strategy, net | T-bill + 4% | S&P 500 |
| :--- | ---: | ---: | ---: |
| 2021 | 35.0% | 4.1% | 26.9% |
| 2022 | 15.9% | 6.2% | -19.4% |
| 2023 | 4.8% | 9.5% | 24.2% |
| 2024 | 13.3% | 9.3% | 23.3% |
| 2025 | 3.1% | 8.4% | 16.4% |
| 2026, January-August | 18.9% | 5.2% | 12.3% |

Values are rounded from the saved CSV; the presentation rounds the 2023
hurdle to 9.4%. The final partial year is not annualized in this table.

## Where returns came from

The long leg contributed approximately **+16.2% of NAV per year**, with
positive price P&L in five of six years. The short leg contributed **-3.1%**
per year on average but added **+27.5% in 2022**, when longs lost 12.6%.
These leg figures are before trading costs, borrow and cash income; they
are not standalone compounded returns.

The short book lost 18.0% of NAV in 2023 and 11.9% in 2025. During the
May-October 2025 drawdown, longs added 13.0% while shorts lost 25.9%, before
costs and cash income. Speculative themes in names such as Oklo, Applied
Digital and Quantum Computing outran the long book. The presentation reports
recovery to a new high by July 2026.

Largest positive contributors included Comfort Systems USA (+3.10% of NAV),
Newmont (+2.92%) and DigitalOcean (+2.45%). Largest negative contributors
included Teradyne (-2.50%), Oklo (-2.34%) and GE Vernova (-2.24%). These are
cumulative price-P&L contributions while held, not stock total returns.

## Implementation and costs

The book held 200 names every month. Average/max gross exposure was
174.93%/193.93%; net ranged from -3.93% to 43.90%. The maximum individual
weight was 5.0% and the median short market capitalization was $15.5 billion.

Modeled annual trading and borrow costs were **0.76%** and **0.44% of NAV**,
respectively. Average one-way turnover per rebalance was 32.62%; annualized
observed turnover was 391.43%. Complete calendar years used approximately
395% under the separate below-400% annual budget. Initial deployment is
excluded from that budget. These turnover definitions should not be
interchanged with Arcana's platform turnover statistic.

## Interpretation and open checks

- **Local vs Arcana:** the headline table is the local monthly net replay.
  Arcana Portfolio 73 screenshots provide separate platform risk/attribution
  analytics, before local modeled costs. Their figures use different
  normalization/accounting and should not replace the net backtest table.
- **Daily accounting:** held-return coverage is incomplete and daily/monthly
  NAV do not reconcile. The daily beta and daily risk numbers in the deck
  are indicative. The saved compliance receipt reports `all_pass: false`
  and `submission_ready: false`; daily volatility is null rather than certified.
- **Input timing and bias:** historical vintages, actual source availability
  lags, I/B/E/S coverage, 8-K survivorship and universe/return bias need further
  measurement. Walk-forward splits alone do not resolve these issues.
- **Security mapping:** historical Arcana mappings, including Coherent and
  SVB exceptions described in prior audits, need independent reconciliation.
  Historical point-in-time availability of the exported Arcana betas is
  not independently established.
- **Research selection:** the final presentation is a selected hackathon
  result after experimentation. A new independent holdout, cost sensitivity
  and broader regime tests remain useful next steps.

The original aggregate reports and check receipts are preserved in
[results/final_strategy](../results/final_strategy/README.md), with file hashes
and source context. They were not relabeled as a fresh audited backtest.
