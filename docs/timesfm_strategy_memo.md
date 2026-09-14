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

The buffer raises return *and* lowers turnover by 4x: the forecast signal is persistent, and monthly forced re-ranking was selling winners to buy noise. The improvement holds in both halves of the sample — it is a structural property, not a tuned artifact. (k=1500 applies to the forecast cross-section, which averages ~2,600 names/month (range ~1,900-3,450) — so the buffer retains a name anywhere within the top/bottom ~45-58% of the ranked universe, wider than a "quarter".)

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

- **Ratchet exposure rule** (exact mapping, as reproduced in src/repro): month 1 runs at 0.5x exposure (the start counts as the first step); scale up +0.1 toward 1.0x whenever equity reaches a new high; while in drawdown >10%, exposure = max(0.3, 1 − 2×drawdown). The scale for month t uses equity information through t−1 only.

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
3. **The risk cannot be diversified away within a book — but it can across engines.** Every tested within-book filter (size caps, universe exclusions, 8-K vetos, market-timing) either fails to reduce crash losses or destroys the alpha, because the crash mechanism (cohort-wide small-cap repricing) is inseparable from the TimesFM return mechanism. The drawdown-responsive ratchet was the only within-book protection that improved risk-adjusted performance in both halves of the sample. However, the 2020 crash turns out to be largely engine-specific: an independently constructed XGBoost book (corr 0.31 with the TimesFM book) was *positive* in 2020 (+7.1% vs −50.5%), and blending the two engines at 50/50 halves the worst cascade months without any reactive rule (§8 v2).
4. **TimesFM licensing**: v3.0 weights are non-commercial (acceptable for the hackathon); TimesFM 2.5-200m (Apache-2.0) is the fallback.

## 7. Reproducibility

- Forecast generation: `src/timesfm_forecast.py` (GPU; artifacts committed under `output/`)
- **Reproduction of §2/§3, §5.4, §8**: `src/repro/`
  - `repro_lib.py` — shared primitives: rank-buffer book, ratchet sizing, FRED market data, metrics
  - `repro_01_base_and_buffer.py` — base k=100 and buffer k=1500 (§2, §3)
  - `repro_02_buffer_ratchet.py` — buffer + ratchet (§5.4)
  - `repro_03_final_config.py` — §8 final config v2 (blend) + superseded v1 + equity curve; artifacts under `output/repro/`
- XGBoost+TimesFM backcast folds 2018-2020 (§8 v2 extension): `src/xgb_timesfm_backcast.py` → `output/preds_xgb_timesfm_backcast.csv` (2018 thin-train, 2019-20 quasi-OOS; 2016-17 not foldable)
- The reproduction pinned three previously unstated conventions: the ratchet de-gross mapping `exposure = max(0.3, 1 − 2×drawdown)` with month 1 at 0.5x; the SPX sleeve earning the SPX price return; the v1 ratchet scaling both sleeves. All §2-§5.4 and both §8 configs reproduce exactly from committed code, except the original §8 beta (+0.14, corrected to +0.27/+0.44 in the v1 record).
- Combined XGBoost+TimesFM model (OOS folds): `src/xgb_timesfm.py` (Sharpe 1.74 in 2021-26)
- Data: `data/raw/chars_final_with_names.parquet`, 8-K corpus `data/raw/8k_20150101_20260831_identified.parquet`, FinBERT scores `output/finbert_scores.parquet`


## 8. Final competition configuration v2: two-engine blend, beta-neutral SPX sleeve

The submitted configuration (supersedes the single-engine v1, kept below):

1. **Alpha book at 1.2x gross, two engines blended 50/50**:
   - TimesFM 3.0 book with the k=1500 rank buffer, at 0.6x gross (legs 0.3/0.3);
   - XGBoost (147 characteristics + TimesFM forecast features, default hyperparameters) raw book at 0.6x gross (legs 0.3/0.3) — **no buffer**: tree ranks lack TimesFM's persistence and the buffer destroys the tree book (Sharpe 1.74 → 0.77 in 2021-26);
   - the two books correlate only **0.31** monthly: ~389 positions (372-400), ~11-name overlap. The DD reduction comes from engine diversification, not sizing.
2. **SPX sleeve at 0.25x of capital**: a structural beta anchor, not timing. Both books' 1x beta ≈ 0 (TimesFM buffered +0.02, XGB raw −0.03), so the sleeve alone sets realized beta; the weight is picked by the rule "beta ≈ 0", not by return maximization.
3. **No ratchet**: in the blend the drawdown-responsive ratchet is inert — Sharpe 1.80-1.84 and maxDD −16.1% to −17.4% across every ratchet setting *including off* — because the 2020 cascade is absorbed by diversification: the XGB book was **positive in 2020 (+7.1%)** while the TimesFM book lost −50.5%. Dropping it removes 5 parameters and the mandate's time-varying-exposure disclosure obligation: exposures are static (gross 145%, net +25%) except the monthly rebalance.

### Performance, 2018-01 → 2026-08 (104 months; XGBoost backcast starts 2018)

| Metric | Full window | OOS 2021-26 |
|---|---:|---:|
| Ann. return (arith) | **+21.8%** | +27.3% |
| CAGR | **+23.3%** (6.1x) | +30.1% |
| Sharpe | **1.82** | — |
| IR vs TB3MS+4% | 1.27 | 1.61 |
| Max drawdown | **−17.4%** | **−5.3%** |
| Realized beta | +0.09 (se 0.08) | +0.24 (se 0.11) |
| Worst month / months < −5% | −7.8% (2021-01) / 3 | |
| Hit rate | 70% | |

Yearly detail (ann. return / Sharpe):

| Year | Ann. return | Sharpe |
|---|---:|---:|
| 2018 | +28.5% | 3.50 |
| 2019 | +14.4% | 1.26 |
| 2020 | −8.5% | −0.84 |
| 2021 | +16.9% | 0.98 |
| 2022 | +33.4% | 2.05 |
| 2023 | +18.4% | 1.96 |
| 2024 | +30.2% | 2.71 |
| 2025 | +26.2% | 2.56 |
| 2026 (8m) | +44.7% | 6.49 |

![Final equity curve v2](../output/repro/timesfm_final_equity_curve_repro.png)

*Top: growth of $1 (log) for the v2 blend vs the superseded v1 single-engine config, the raw 1x blend book, S&P 500, and the T-bill+4% benchmark. Bottom: v2 drawdown — 2020 contained at −17.4% with static exposures.*

### Like-for-like vs the superseded v1 (both on 2018-2026)

| Config | CAGR | Sharpe | IR | maxDD (OOS) | beta (OOS) | Worst | Months < −5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v2 blend 1.2x + 0.25x SPX** | +23.3% | 1.82 | 1.27 | **−17.4%** (−5.3%) | +0.09 (+0.24) | −7.8% | 3 |
| v1 TFM-only 1.0x + 0.5x SPX + ratchet | +22.8% | 1.59 | 1.11 | −24.1% (−6.5%) | +0.27 (+0.44) | −8.9% | 6 |

v1 native-window (2016-10 → 2026-08) numbers as previously reported in this section: ann +21.8%, Sharpe 1.62, maxDD −24.1%, beta +0.27/+0.44, cumulative 7.8x. Reproducible via `src/repro/repro_03_final_config.py`.

### Why the blend works where the single book could not be protected

§5.3's conclusion — "the risk cannot be diversified away" — was true **within one book** and is false **across engines**. The TimesFM crash is engine-specific: in 2020-04 / 2020-06 / 2021-01 the TimesFM short book lost −24.5% / −22.7% / −25.2% while the XGBoost book returned −4.6% / **+12.7%** / +0.3% (calendar 2020: XGB +7.1% vs TimesFM −50.5%). The drawdown mechanism (§5.1) is a TimesFM-signal-specific small-cap squeeze, not a property of the data. Engine diversification (corr 0.31) delivers what the ratchet delivered reactively — without time-varying exposure and without its return cost. The SPX sleeve cushions what remains: it is a constant beta hedge, always on (the EMA/momentum filters of §5.3 failed exactly because they de-grossed in market downturns).

### Parameter-optimization disclosure (multiple-testing discipline)

| Item | v1 (superseded) | v2 (final) |
|---|---|---|
| Tunable scalars | ~8 (buffer k; ratchet start/step/gate/floor/depth; SPX weight; book gross) | **2** (buffer k; book gross) |
| Structural rules (not tuned) | — | engines 50/50 (symmetry); SPX weight = the beta≈0 anchor; ratchet dropped (ablation); no buffer on XGB (measured: hurts) |
| Rejected experiments disclosed | size caps ×4, exclusions, 8-K veto, EMA ×3, SPX family ×4-5 | same, plus buffer-k grid ×6 and the ratio grids used in sizing |

Sensitivity evidence (reproducible; all plateau-shaped except where noted):

- **Buffer k (TimesFM book)**: Sharpe by k = 100→0.70, 500→0.85, 1000→0.88, **1500→1.02**, 2000→0.89, 2500→0.68. The earlier claim of robustness "across every parameter choice" does not hold — k=1500 is the in-sample argmax. Disclosed; its influence is halved inside the blend and the blend's dominance does not depend on it.
- **Ratchet ablation (blend)**: Sharpe 1.80-1.84, maxDD −16.1% to −17.4% across start/step/gate/floor/depth settings and with the ratchet off → dropped.
- **Blend weight** (at 1.2x book): 0.3 → Sh 1.87 / −12.0%; 0.5 → 1.83 / −16.8%; 0.7 → 1.68 / −23.1% — Sharpe plateau over 0.3-0.5; 0.5 picked by symmetry.
- **Book gross**: monotone IR/DD trade-off, no interior peak — a stated policy pick, not an optimization.

§6.1's discovery overlap still applies to the specification itself (chosen after seeing 2016-26 data). Mitigations: the blend's dominance replicates out-of-design-window (2018-20 quasi-OOS: blend Sharpe 0.99 vs 0.60 TFM-only / 0.86 XGB), all surviving knobs sit on plateaus, and this disclosure. Expect OOS degradation regardless.

### Rule compliance

- Positions: ~389 names (372-400; two 100/leg books, ~11-name overlap) + 1 index instrument — within the 100-500 count.
- Gross exposure: 145% constant (1.2x book + 0.25x SPX; static — no time-varying exposure to explain or predict).
- Net exposure: +25% at every date — inside the band, off the edge.
- Realized beta: **+0.09 full window (se 0.08); +0.24 OOS 21-26 (se 0.11)** — materially closer to zero than v1's +0.27/+0.44; the residual OOS beta is the deliberate low-beta long sleeve and is explained as such.
- Rebalancing: monthly. Turnover/leg: TimesFM 22%/11%, XGB 56%/52% (blended ~39%/31%).
- Conventions (pinned by reproduction, §7): the SPX sleeve earns the SPX price return; book legs are excess over the pipeline RF.

### Caveats specific to this configuration

1. **2016-17 coverage gap**: XGBoost cannot be backcast before 2018 (panel data starts 2015-02; the first foldable test year under the spec's structure is 2018). The 2016-10..2017-12 segment of the narrative remains TimesFM-only (§2-§5); the blend window is 2018-2026.
2. **Backcast labeling**: the 2018 XGB fold trains on 11 target-months (thin); the 2019-20 folds are proper but those periods later serve as validation for the 2021-26 folds. The blend's dominance replicates in that segment (Sharpe 0.99 vs 0.60/0.86), but it is not OOS-pure on par with 2021-26.
3. **In-sample decisions**: buffer k (a measured in-sample peak — disclosed above), book gross (policy pick), the SPX weight family, and the blend design itself. Expect OOS degradation (§6.1).
4. The XGB short-leg composition (small-cap heavy) and its higher turnover (56%/52%) carry the same short-book realism scrutiny as §6.2.

### Superseded v1 configuration (single-engine, for the record)

TimesFM buffered book at 1x gross + 0.5x SPX (price return) + ratchet, 2016-10 → 2026-08: ann +21.8%, Sharpe 1.62, maxDD −24.1% (−21.2% in 2016-20; −6.5% in 2021-26), realized beta +0.27 (se 0.08) / +0.44 OOS, 83/119 months positive, worst month −8.9% (2021-01), cumulative 7.8x. Retired in favor of v2 on every metric — including the compliance-critical beta — with like-for-like numbers in the table above.
