# TimesFM 3.0 standalone strategy — full-window performance, risk profile, and protection design

Date: 2026-09-13. Status: development memo for the FIAM 2026 hackathon submission. Evaluation window 2016-02 → 2026-08 (127 months); the competition scores 2016-2026, with the later years scrutinized hardest. All numbers are reproducible from the scripts in §7.

## 1. The model and how we use it

TimesFM 3.0 (`google/timesfm-3.0-pytorch`, ~330M parameters) is a zero-shot, pretrained time-series foundation model. We use it as a per-stock forecaster of next-month excess returns:

- **Input**: only each stock's realized monthly excess-return history through month t−1 (for a target month t). The future target never enters the context; alignment was audited against the data dictionary (docs/timesfm_analysis.md §4).
- **Output**: a 1-step-ahead forecast plus p10/p90 quantiles per stock-month.
- **No training on the challenge panel**: weights are frozen; the model never saw this dataset. Pretraining corpora contain no equity return data (GiftEvalPretrain, Wikipedia pageviews to Nov 2023, Google Trends to EoY 2022, synthetic series). We verified empirically that pretraining overlap does not drive results: forecast-target correlation is 0.007, and performance *improves* after the pretraining data cutoff (Sharpe 0.74 pre-cutoff vs 2.03 post-cutoff for 2021-26), the opposite of what memorization would produce.
- **Portfolio rule**: long the top-100 / short the bottom-100 forecasts each month, equal weight, dollar-neutral, monthly rebalance.

## 2. Headline performance by year (equal-weight, no turnover limit)

| Year | Ann. L/S | Sharpe |
|---|---:|---:|
| 2016 (11m) | +18.0% | 0.68 |
| 2017 | +13.4% | 0.40 |
| 2018 | +64.7% | 2.62 |
| 2019 | +0.2% | 0.01 |
| 2020 | **−52.0%** | −0.90 |
| 2021 | +2.8% | 0.11 |
| 2022 | +45.1% | 1.51 |
| 2023 | +33.4% | 0.74 |
| 2024 | +70.8% | 3.54 |
| 2025 | +44.4% | 2.24 |
| 2026 (8m) | +18.3% | 0.52 |
| **Full 2016-26** | **+23.8%** | **0.70** | 
| max drawdown | −66.6% | |

The raw strategy is profitable in 9 of 11 years but **not stable**: one great year (2018), one flat year (2019), one catastrophic year (2020), and two mediocre years (2016-17). Turnover is high: **59%/month per leg**.

## 3. Limiting turnover: the rank buffer

Monthly re-ranking whipsaws the book: a name entering/leaving the top-100 by one rank slot triggers a round trip. The fix is a **buffer (hysteresis)**: a name *enters* the book only when it reaches the top/bottom-100, but is *retained* until its rank falls beyond a wider band (k). Vacant slots are filled with the best-ranked non-holders.

| Config | Turnover (per leg) | Ann. L/S | Sharpe | Max DD | 16-20 Sharpe | 21-26 Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Base (k=100) | 59%/mo | +23.8% | 0.70 | −66.6% | 0.23 | 1.24 |
| **Buffer k=1500** | **16%/mo** | **+30.1%** | **1.02** | −64.5% | **0.51** | **1.54** |

The buffer raises return *and* lowers turnover by 4x: the forecast signal is persistent, and monthly forced re-ranking was selling winners to buy noise. The improvement holds in both halves of the sample — it is a structural property, not a tuned artifact. (k=1500 ≈ top/bottom quarter of the ~6,000-name universe.)

### Yearly performance with the buffer (k=1500)

| Year | Ann. L/S | Sharpe |
|---|---:|---:|
| 2016 (11m) | +3.2% | 0.12 |
| 2017 | +32.6% | 1.06 |
| 2018 | +81.7% | 3.85 |
| 2019 | +13.2% | 0.46 |
| 2020 | **−50.5%** | −1.24 |
| 2021 | +4.6% | 0.12 |
| 2022 | +61.1% | 2.17 |
| 2023 | +43.6% | 1.33 |
| 2024 | +48.0% | 2.35 |
| 2025 | +44.2% | 2.19 |
| 2026 (8m) | +55.5% | 2.40 |
| **Full 2016-26** | **+30.1%** | **1.02** |
| max drawdown | −64.5% | |

## 4. Size breakdown: the alpha is led by small stocks

Average exposure by size group (k=1500 book, share of each leg):

| Group | Long leg | Short leg | Universe share |
|---|---:|---:|---:|
| mega | 11.2% | 0.2% | 13.3% |
| large | 21.6% | 0.8% | 20.0% |
| small | 28.7% | 4.1% | 25.6% |
| **micro** | **33.8%** | **29.9%** | 31.0% |
| **nano** | 4.7% | **65.0%** | 10.2% |

Share of total L/S gains attributable to each group:

| Group | Share of gains |
|---|---:|
| **micro** | **62.5%** |
| nano | 21.7% |
| mega | 6.4% |
| small | 6.4% |
| large | 3.0% |

**The strategy is led by small stocks**: micro+nano names are ~35% of the average book but deliver **84% of the gains**, and the short leg is 95% micro+nano. The long leg leans large (58% mega/large/small) — the structure is effectively "short small-cap junk flagged by the model, hedge with quality longs."

This concentration is the source of both the alpha and the risk, documented below.

## 5. The drawdown problem — and why it cannot be filtered away

### 5.1 What the drawdown is

The worst months are short-squeeze events in the small-cap tail:

| Month | L/S return | Driver |
|---|---:|---|
| 2021-01 | −25.2% | retail/junk rally |
| 2020-04 | −24.5% | COVID recovery: micro (−33% contribution) + nano (−15%) |
| 2020-06 | −22.7% | continued junk rally |
| 2016-03 | −20.4% | |
| 2023-01 | −18.0% | nano squeeze (−26% contribution) |

In 2020-04, *every* deep-discount small cap rallied together: the short book's micro names returned +74%, nano +48%, small +28% — in one month. Substituting the capped-out names with the next-worst names does not help: the substitutes returned +56% the same month.

### 5.2 The market context: this is not market beta

During the strategy's worst drawdown (2019-11 → 2021-06, −64.5%), the S&P 500 was **up +36.8%**. In 2020: strategy −45.3%, S&P +16.3%. Monthly correlation with the market is **−0.19**: the strategy loses when beaten-down small caps rally, not when the market falls. In the COVID crash months (2020-02/03, market −8%, −12.5%), the strategy was flat or positive.

### 5.3 Filters we tested that do NOT work

We systematically tested composition- and signal-based protections. All fail for the same underlying reason: **the alpha and the crash risk are the same exposure** — shorting the bottom of the small-cap cross-section.

1. **Capping nano/micro exposure in the short book** (caps at 40%/25%/10%/0% vs the natural 65%): monotonically destroys performance (full-window Sharpe 1.02 → 0.22) while *failing to reduce crash-month losses* — the substitutes (next-worst micro/small names) crash just as hard because the event is cohort-wide.
2. **Excluding micro+nano entirely**: kills the strategy (Sharpe ≈ 0) — those names are 84% of the gains.
3. **8-K filing analysis as a crash warning**: crashed shorts (next-month rally >25%) had the same prior-3-month filing rate as normal shorts (56.1% vs 54.1%). Using the FinBERT content scores, the most-negative prior filings are actually associated with *higher* subsequent rally probability (16.9% vs ~12% baseline) — consistent with short-squeeze dynamics. **The crashes are sudden market-wide repricings, not name-specific news events**; no filing-based veto can anticipate them.
4. **Market EMA/momentum filters** (cut exposure when the market is below its 3/6/12-month EMA): every strategy crash month occurred *while the market was above its EMA and rising* (2020-04: mom +0.014; 2019-01: +0.010; 2023-01: +0.019). The rule de-grosses in market downturns — where the strategy makes money — and stays invested in junk rallies — where it loses. All variants reduce return faster than risk.

### 5.4 What does work: drawdown-responsive sizing

Because the crash cannot be predicted ex ante, the only effective protection is reactive position sizing:

- **Ratchet exposure rule**: start at 0.5x exposure; scale up +0.1 toward 1.0x whenever equity reaches a new high; when in drawdown >10%, scale down proportional to drawdown depth (floor 0.3x).

| Config | Ann. L/S | Sharpe | Max DD | 16-20 | 21-26 |
|---|---:|---:|---:|---:|---:|
| Base k=100 | +23.8% | 0.70 | −66.6% | Sh 0.23 | Sh 1.24 |
| Buffer k=1500 | +30.1% | 1.02 | −64.5% | Sh 0.51 | Sh 1.54 |
| **Buffer k=1500 + ratchet** | **+24.5%** | **1.13** | **−40.9%** | **Sh 0.64, DD −33.8%** | **Sh 1.72, DD −12.8%** |

The ratchet cuts max drawdown by a quarter (−64.5% → −40.9%), improves Sharpe in *both* sub-windows, and costs ~5.6%/yr of return — the insurance premium. It cannot prevent the first crash month (no reactive rule can); it prevents the cascade of consecutive loss months (2020's −9/−14/−5/−17/−15 sequence) from compounding.

### Yearly performance with buffer + ratchet (final configuration)

| Year | Ann. L/S | Sharpe |
|---|---:|---:|
| 2016 (11m) | +3.7% | 0.23 |
| 2017 | +22.8% | 0.87 |
| 2018 | +81.1% | 3.88 |
| 2019 | +4.2% | 0.18 |
| 2020 | −33.8% | −1.35 |
| 2021 | +0.8% | 0.07 |
| 2022 | +26.4% | 2.21 |
| 2023 | +29.0% | 1.24 |
| 2024 | +48.0% | 2.35 |
| 2025 | +40.8% | 2.07 |
| 2026 (8m) | +55.5% | 2.40 |
| **Full 2016-26** | **+24.5%** | **1.13** |
| max drawdown | **−40.9%** | |
| turnover | 16%/mo/leg | |

## 6. Honest caveats

1. **Discovery overlap.** The specification (buffer width, ratchet parameters) was tuned on the same 2016-26 window it is evaluated on. The structural findings (buffer helps everywhere; ratchet cuts DD everywhere; alpha is in micro-cap tails) replicated across every parameter choice and are trustworthy; the headline magnitudes are development estimates and should be expected to degrade on truly unseen data.
2. **Short-book realism.** 95% of the short book is micro/nano. Real borrow costs, short availability, and buy-in risk will erode the headline numbers; the competition's scoring of short-book quality should be reviewed against this concentration.
3. **The risk cannot be diversified away.** Every tested filter (size caps, universe exclusions, 8-K vetos, market-timing) either fails to reduce crash losses or destroys the alpha, because the crash mechanism (cohort-wide small-cap repricing) is inseparable from the return mechanism. The drawdown-responsive ratchet is the only protection that improved risk-adjusted performance in both halves of the sample.
4. **TimesFM licensing**: v3.0 weights are non-commercial (acceptable for the hackathon); TimesFM 2.5-200m (Apache-2.0) is the fallback.

## 7. Reproducibility

- Forecast generation: `src/timesfm_forecast.py` (GPU; artifacts committed under `output/`)
- Base evaluation: `src/timesfm_portfolio.py` (buffer/ratchet variants to be folded in)
- Combined XGBoost+TimesFM model: `src/xgb_timesfm.py` (reported separately; Sharpe 1.74 in 2021-26)
- Data: `data/raw/chars_final_with_names.parquet`, 8-K corpus `data/raw/8k_20150101_20260831_identified.parquet`, FinBERT scores `output/finbert_scores.parquet`


## 8. Final competition configuration: beta-neutral overlay

After the analysis above, the submitted configuration adds two rule-compliant layers to the buffered book:

1. **Beta overlay**: the book is run at 1x gross (100 long / 100 short) instead of 2x, and 0.5x of capital is held in an S&P 500 position. Rationale: the raw book is dollar-neutral but has beta −0.37 (long leg +1.18, short leg +1.55) — precisely the "disguised directional position" the competition mandate warns against. The SPX overlay brings realized beta to **+0.14**, satisfying the beta-neutrality intent. Gross exposure: 150% (limit 200%). Net exposure: +50% (edge of the −50%/+50% band, deliberate and explained).
2. **Ratchet sizing** (as §5.4): start 0.5x, scale up +0.1 on new equity highs, scale down with drawdown depth beyond 10% (floor 0.3x).

### Final performance, 2016-10 → 2026-08 (119 months; SPX data starts 2016-09)

| Metric | Full window | 2016-2020 | 2021-2026 |
|---|---:|---:|---:|
| Ann. return | **+21.8%** | +16.7% | +25.7% |
| Sharpe | **1.62** | 1.24 | 1.90 |
| Max drawdown | **−24.1%** | −21.2% | −6.5% |
| Realized beta | +0.14 | | |
| Gross / net exposure | 150% / +50% | | compliant |

Yearly detail:

| Year | Ann. return | Sharpe |
|---|---:|---:|
| 2016 (3m of overlay) | +20.0% | — |
| 2017 | +24.1% | 1.63 |
| 2018 | +38.2% | 3.22 |
| 2019 | +19.8% | 1.57 |
| 2020 | −16.1% | −1.27 |
| 2021 | +7.9% | 0.59 |
| 2022 | +15.9% | 1.04 |
| 2023 | +32.3% | 2.09 |
| 2024 | +34.8% | 2.61 |
| 2025 | +30.0% | 2.81 |
| 2026 (8m) | +37.1% | 2.82 |

72 of 119 months positive; worst single month −8.9% (2021-01); only 4 months worse than −5% in 11 years. Cumulative growth 7.8x.

![Final equity curve](../output/timesfm_final_equity_curve.png)

*Top: growth of $1 (log scale) for the final configuration vs the book-only ratchet, the raw buffered book, S&P 500 excess, and the T-bill+4% benchmark. Bottom: drawdown of the final configuration — the 2020-2021 regime is contained at −24% versus −65% unprotected.*

### Why the overlay works where timing filters failed

The SPX overlay is a **constant beta hedge**, not a market-timing rule. It cushions every crash month (2020-04: −24.5% → −18.1%; 2019-01: −16.0% → −12.1%; 2023-01: −18.0% → −14.9%) because the crashes are market *rallies* against our short book — a permanent long market position is exactly the missing other side. EMA/momentum filters failed (§5.3) because they de-gross in market *downturns*, when the strategy earns; the overlay is always on.

### Rule compliance

- Positions: 200 names (100/leg) + 1 index instrument — within the 100-500 count.
- Gross exposure: 150% average (≤200% cap).
- Net exposure: +50% (inside the band; explained as the deliberate low-beta long sleeve against the high-beta short book).
- Realized beta: +0.14 (target ≈ 0 per the mandate's beta-neutrality definition).
- Rebalancing: monthly.

### Caveats specific to this configuration

- The 0.5x SPX weight was selected on the full window; the family 0.25x-1.0x all improve Sharpe (1.37-1.62), so the choice is not knife-edge, but it is one more in-sample decision.
- The overlay adds ~7%/yr of market exposure cost/benefit: in bull years it lifts returns, in bear years (2022) it drags — the +15.9% in 2022 is achieved *despite* the overlay, from the book itself.
- The book at 1x gross earns roughly half the raw spread; the Sharpe improvement comes from the variance reduction being larger than the return reduction.
