# Strategy

The economic idea is that business fundamentals, expectations, valuation and
corporate disclosures adjust at different speeds. Four research pods express
those mechanisms as stock rankings. Combining them aims to diversify the
sources of return while explicit constraints control the resulting book.

Source: final presentation, slides 3, 5-7 and 22; implementation in
[final_v3](../final_v3/README.md).

## Signals and blend

| Pod | Mechanism | Implementation |
| :--- | :--- | :--- |
| Fundamentals | Slow updates to expectations and operating performance | Earnings-surprise/breadth sleeves, momentum-gated I/B/E/S revisions, inventory vs sales, sales vs SG&A |
| Valuation | Differences between model-implied value and price; cash runway | Valuation-gap and runway sleeves, with the same liquidity and sizing checks as other signals |
| Corporate events | Information in the type and frequency of disclosures | Boosted-tree forecast using 54 counts of 8-K item families over 1/3/12 months |
| Market dynamics | Relative-return patterns across characteristics | Walk-forward XGBoost ranking across 147 company characteristics |

Available sleeve scores are blended and converted into monthly ranks. A
company-text/peer graph forecasts earnings information and supplies a **5%
rank tilt** where available. Stocks without a graph forecast retain the
sleeve rank. Pods describe the economic research organization; they are not
four separately allocated capital accounts.

```mermaid
flowchart LR
    A[Characteristics, filings and disclosed revisions] --> B[Four research pods]
    B --> C[Fixed sleeve blend]
    G[Company and peer graph] --> C
    C --> D[100 longs and 100 shorts]
    D --> E[Liquidity, beta, exposure and turnover sizing]
    E --> F[Monthly portfolio and accounting checks]
```

## Portfolio construction

The rank selects 100 long and 100 short names, with sector-balanced selection
and a 0.5 retention buffer for incumbents. Higher-beta shorts allow a smaller
short notional to offset the long book; beta neutrality does not imply equal
long and short dollars. Average exposure in the saved replay was 174.93%
gross and 15.07% net.

| Control | Rule at formation |
| :--- | :--- |
| Price and short eligibility | Minimum $5 price; short market capitalization at least $500 million |
| Trade liquidity | At most 5% long / 2% short of five days of observed dollar volume |
| Position size | At most 5% of NAV per name |
| Capital deployment | At least 95% own capital deployed to longs; cash/Treasury cap 5% |
| Gross and net | Gross at most 200%; net within ±50% |
| Sector exposure | At least five sectors; sector net exposure within ±20% of NAV |
| Formation beta | Absolute beta at most 0.025 under each of three estimators |
| Turnover | Calendar-year one-way turnover strictly below 400%; initial funding exempt and forced exits included |
| Short protection | Halve the riskiest 5% of shorts by peer-squeeze probability; cap rallied shorts at 1.5 times target |

The beta estimators are trailing 252-session local daily beta, trailing
36-month local monthly beta, and Arcana model beta to SPY with a local daily
fallback. The saved hybrid used 8,752 Arcana observations and 4,848 local
fallbacks across 13,600 holding-month observations. All three constraints are
checked after sizing; exposure can drift between monthly rebalances.

Unused own capital earns a lagged T-bill proxy. Short proceeds are treated
as separate collateral, and borrowing above own NAV incurs financing. Trading
and borrow estimates are deducted from the local reported returns.

## What the controls do and do not establish

Fixed rules determine weights and limits. Research agents can propose signals
before a test year, but cannot override risk controls. No agent proposal
entered the final historical book.

The saved formation checks passed name count, exposure, sector, position,
short-capitalization, beta and calendar-year turnover requirements. They do
not establish continuous neutrality or complete daily compliance. Missing
held returns and daily/monthly NAV disagreement prevent certification of the
12% daily volatility ceiling. See [results and limitations](results.md).
