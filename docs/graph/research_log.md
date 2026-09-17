# Economic graph research log (FIAM 2026): graph, sell-side expectations and valuation

Everything tried between 11 and 15 September 2026 by Claude Code and Codex, in one place: what was
built, how it was tested, the numbers, and the decision. The per-experiment scripts, tests and
reports were removed after this log was written. They are archived outside the repository in
`~/fiam26_research_archive_20260914.tar.gz` (code, tests, docs and small result files).

**Contents**
0. [Bottom line](#0-bottom-line)
1. [Validation rules and how to read the results](#1-validation-rules-and-how-to-read-the-results)
2. [Data](#2-data)
3. [Economic graph](#3-economic-graph)
4. [Is the graph useful?](#4-is-the-graph-useful)
5. [Sell-side expectations and the expectation gap](#5-sell-side-expectations-and-the-expectation-gap)
6. [Surprise model and monthly strategies](#6-surprise-model-and-monthly-strategies)
7. [Valuation: analyst-, price- and graph-implied value](#7-valuation-analyst--price--and-graph-implied-value)
8. [Alpha search with the existing monthly data](#8-alpha-search-with-the-existing-monthly-data-wf)
9. [Daily event study: was it already priced in?](#9-daily-event-study-was-it-already-priced-in-exploratory-yahoo-prices)
10. [Decisions by architecture step](#10-decisions-by-architecture-step)
11. [Next steps and data to request](#11-next-steps-and-data-to-request)
12. [Repository after cleanup](#12-repository-after-cleanup)

---

## 0. Bottom line

- **No tradable return edge has survived the challenge's walk-forward schedule.** The best monthly
  strategy, a factor-neutral small-cap book ranked on the predicted analyst surprise, earned 6.24% a year
  (t 0.89) under the challenge schedule and 8.06% (t 1.12) when training ran through the prior year.
  Neither is significant, and neither clears cash + 4% after plausible costs.
- **One predictive asset is solid.** A model of where the Street is wrong (the next analyst EPS
  surprise) has out-of-sample rank IC of 0.28–0.37 in every test year from 2021 to 2026.
  Analyst-error history is its most important input.
- **The economic graph is sound and useful for risk, not alpha.** Graph peers hedge better than
  GICS peers and transfer earnings information better, but graph-derived return signals do not
  survive factor controls in investable stocks.
- **Valuation divergences (analyst vs price vs graph) have no out-of-sample edge**, in either the
  simple sector models or the improved implied-cost-of-capital and warranted-multiple version.
- **A broad search found nothing investable either.** 465 signal × size × horizon candidates (graph return signals, Street
  variables, 8-K events, graph structure) went through the walk-forward selection. No small, large or mega book is significant.
  The only significant book is in micro caps: 26.0% [2.80], but 11.1% [1.31] without 2025. It is short-driven and loads on
  short-term reversal ([section 8](#8-alpha-search-with-the-existing-monthly-data-wf)).
- **Daily prices show the information is already priced in.** At the same actual surprise, stocks the model
  predicted to beat move 3.6% less at the announcement, and graph peers reprice within the peer's own 3-day
  announcement window. Trading after the announcement (forecast + surprise drift, 3-month hold) earned 7–11% a year
  with 4–12% factor alpha from 2021 in every size group, but nothing in 2017–2020. Under the competition document's
  exact procedure (strategy chosen on the two-year validation window, net of assumed costs), only micro caps
  (7.2% net, IR vs cash + 4% of 0.48) and large caps (5.5% net, IR 0.28) clear the hurdle; small, mega and the
  combined investable book do not ([section 9](#9-daily-event-study-was-it-already-priced-in-exploratory-yahoo-prices)).
- **Strongest compliant result: trade the surprise only after the Street confirms it**
  ([section 9.10](#910-overnight-pre-registered-edge-tests-the-street-confirms-the-surprise-wf)). The signal ranks reporters
  at the end of the following month by surprise plus next-quarter analyst revision breadth. It is pre-registered and chosen
  on validation.
  - Small + large + mega: 12.7% net, IR 1.00, alpha t 3.7.
  - Large + mega: 8.9% net, IR 0.78; with the ML forecast added, 11.9% net, IR 1.21.
  - It survives costs, tail/sector variations and a factor regression, and is mainly a long-side effect in large (not mega)
    caps.
  - Multiple testing: across all 190 books tested, two close variants clear the Bonferroni bound (t 3.65); E7 itself (t 3.56)
    falls just short.
  - Limits: it fails under value weighting and earned nothing in 2016–2020, so the post-2021 regime risk remains.
  - Graph additions, dispersion tilts, seasonality, non-earnings events and valuation tilts (ICC, reverse-DCF feasibility,
    warranted multiples, cheap/expensive gate) did not help ([9.12](#912-valuation--street-confirmation-and-timesfm-distribution-signals-pre-registered-batches-1112-wf)).
- **TimesFM's quantile outputs** (forecast per unit of range, downside q10, skew) found no tradable edge in any cap group. The
  only positive book is lottery-stock avoidance in sub-$5 micro caps (17% net, t 2.1), where shorting is impractical (9.12).
- **Best valuation model ([9.13](#913-which-model-values-companies-best-pricing-accuracy-unique-firms-a-learned-gate-and-the-value-gap-as-an-edge-wf--post-hoc-checks)):**
  gradient-boosted trees on non-price fundamentals and forward analyst EPS. They price 35% of firm-months within ±25%
  (median error 0.36 in log terms), against 24% for industry comparables.
  - Sector labels add nothing.
  - A mixture of experts gated on firm traits rediscovers banks/utilities and biotech on its own, but its linear experts are
    less accurate.
  - Nvidia-like firms with no close peers carry 4× the error, so their "overvaluation" is not a signal.
- **New candidate edge: the micro-cap value gap** (price ≥ $5, liquid). Buying names cheap vs the trees' fair value and
  shorting the expensive ones earned 31% net a year (t 4.2), positive every year 2021–2026.
  - It survives a one-month delay, controls for value, momentum, reversal, size, volatility and profitability, and the
    multiple-testing bound.
  - About three quarters of it comes from shorting expensive micro caps, so borrow data is required before believing it.
  - It does not carry over to small, large or mega caps.
- **A lagged rolling-beta overlay** is the only portfolio-level improvement found (return 8.06% → 10.12%, t 1.45,
  realized beta −0.27 → −0.10). It is a risk-control candidate, not proven alpha.
- **Next:** CRSP daily returns (announcement-window and drift tests of the surprise forecast) and
  I/B/E/S sales consensus. See [section 11](#11-next-steps-and-data-to-request).

## 1. Validation rules and how to read the results

**Challenge rule** (competition PDF, "Training Procedures"; `data/raw/_pdf_extract.txt` around line 516):
expanding training window, rolling two-year validation window, splits assigned by target return month,
annual refit, monthly forecasts. Test year 2021: train on target months 2015–2018, validate 2019–2020,
test 2021. Test year 2022: train through 2019, validate 2020–2021, test 2022. Continue to 08/2026.
Labels realized in the test period stay out of training and validation. Report OOS R² (zero benchmark).
This is implemented in `src/graph/walk_forward.py` (`split_masks`).

Labels used throughout this log:

| Label | Meaning |
|---|---|
| **[WF]** | Challenge walk-forward: every choice made on training/validation, results read only on the 2021–2026 test years |
| **[WF-0]** | Walk-forward without a validation window: training runs through December of the prior year (requested as a robustness variant; not the PDF specification) |
| **[FIXED]** | Older analyses with a fixed 2015–2020 "discovery" / 2021–2026 "confirmation" split. **Not compliant.** Several conclusions were drawn by reading 2021–2026. Treat as hypotheses only |
| **[CS]** | Purely cross-sectional or descriptive (no time-series fitting), e.g. data audits and graph construction statistics |

Values are shown as 2015–2020 / 2021–2026 for [FIXED] results (2017–2020 / 2021–2026 for the surprise-model ablation);
brackets hold Newey–West t-statistics.

Point-in-time conventions used everywhere:
- **Timing:** a feature dated month-end t uses only information available by t.
- **Returns:** `ret_exc_lead1m` (month t+1) or compounded excess returns over t+1..t+h.
- **"Investable":** the small, large and mega size groups (NYSE breakpoints).
- **"Factor controls":** 13 characteristics (be_me, ope_be, at_gr1, ivol_capm_252d, market_equity, qmj, beta_60m, ret_6_1, rmax1_21d, seas_1_1an,
  eqnpo_12m, gp_at, turnover_126d) plus ret1, ret_12_1, last surprise, 3-month revision and revision breadth.

## 2. Data

### 2.1 Supplied competition data [CS]

| Dataset | Content | Notes |
|---|---|---|
| `chars_final_with_names.parquet` | 529,082 security-months, 6,561 permnos (6,524 permcos), 01/2015–08/2026, 147 characteristics | Unique on (permno, eom); `ret_exc_lead1m` is the future return |
| `8k_20150101_20260831_identified.parquet` | 373,139 filings, 3,687 permnos | `items` complete (0.3% empty): 2.02 earnings releases 119,644; 7.01 Reg FD 87,932; 8.01 83,714; 5.02 officer changes 70,582; 1.01 agreements 48,376; rare 4.02, 2.05, 2.06, 3.01. Exhibits (press releases) are not guaranteed in the text; timestamps are date-level only |
| `output/finbert_scores.parquet` | FinBERT tone for all 373,139 filings | No predictive value in any test |
| `data/scoring/text_scores.db` | LLM filing scores | Only 2,310 filings, too few to use |

Size groups (median market equity in $M, 2025; typical firms per month): nano 14 (~400–570), micro 219 (~1,300),
small 1,737 (~900, effectively mid caps), large 7,689 (~600), mega 51,096 (~370).

### 2.2 Sell-side analyst data in `local/` (WRDS I/B/E/S summary files) [CS]

| File | I/B/E/S dataset | Content |
|---|---|---|
| `s365yejuuvprhila.csv.gz` (4.5M rows) | EPS consensus summary (statsum) | Mean/median estimate, dispersion, number of estimates, up/down revisions, high/low, fiscal period end, actual and announcement date/time. **Quarterly EPS only, FPI 6–9 (next four quarters)**; 01/2014–05/2026 |
| `b2tkpunzdr2n0s5t.csv.gz` (435k) | Surprise history (surpsum) | Announcement date, actual, pre-announcement consensus, dispersion, SUE score; quarterly EPS |
| `cln59hjwxfpwgvza.csv.gz` (2.7M) | Recommendation summary | Mean/median rating (1 strong buy … 5 sell), number of ratings, upgrades/downgrades, buy/hold/sell %; 25% US rows |
| `lhn8sdwvp01ym0q7.csv.gz` (2.7M) | Actuals and pricing summary | Last annual/interim actual, price, shares, 5-year growth; to 04/2026 |
| `monldfn04vjxfvwv.csv.gz` | Split adjustment factors | |
| `ntbqsjp1a1qhsn1n.csv.gz`, `gsba0tp9aa3eg8bk.csv.gz` | Identifier files (estimates, recommendations) | CUSIP, official ticker, name, exchange, country |
| `j2vz7zigsgxzab3h.csv.gz` | I/B/E/S–CRSP link (iclink) | TICKER, PERMNO, ncusip, date range, match score; last end date 01/2026 |

- **Not available:**
  - analyst names, individual estimates or ratings, broker identities, track records (need the I/B/E/S detail files)
  - price targets
  - sales/EBITDA consensus (SAL)
  - annual FY1/FY2 EPS and long-term growth consensus
  - commodity prices
- **Coverage:**
  - 5,928 of 6,561 panel permnos (90.4%) link to I/B/E/S, and 3,503 of the 3,687 8-K permnos (95.0%).
  - Stock-months linked by the kept loader in `src/graph/expectations.py` (score ≤ 2, latest links kept open past the 01/2026 end date): 90.4%;
    with an expectation state (next unreported quarter's EPS consensus): 79.5%.
  - By size (linked / with state): mega 99.8%/97.0%, large 99.2%/95.7%, small 97.7%/93.1%, micro 84.1%/70.0%, nano 74.7%/44.1%.
    Earlier drafts reported 86%/77% using raw link end dates.
- **The data ends early:** consensus snapshots end 05/2026 and links end 01/2026, so the last months of 2026 have no expectation state.

### 2.3 Audits [CS]

| Check | Result |
|---|---|
| Characteristics panel quarterly timing | `niq_su` changes in the same month as the I/B/E/S announcement for 96.9% of quarters and earlier for 0.2%. No look-ahead in the panel |
| I/B/E/S ticker → PERMNO collisions | 0 of 478,235 ticker-months |
| I/B/E/S split basis | Estimates, actuals and `PRICE` are all on the current split-adjusted basis (AAPL 2019 price 49.47 vs CRSP 197.92). Never mix with CRSP prices |
| Consensus rounding | statsum rounds MEANEST to cents on the adjusted basis (NVDA 2019: 0.03 vs 0.0286), so revisions are noisy for heavy splitters. Use surpsum for surprise labels |
| Surprise-file consensus vs last monthly snapshot | Median absolute difference 0.003; 82% within 1 cent; actuals match 99.4%; last snapshot typically 18 days before the announcement |
| Report timing | Median announcement 36 days after fiscal quarter end |
| `intrinsic_value` | A dollar level, not a ratio (divide by market equity before using it as a control) |
| `sale_gr3` | Cumulative 3-year sales growth (rank correlation 0.98 with compounded annual growth) |
| Risk-free rate | `data/external/TB3MS.csv` monthly; lag one month (published after month end) |

### 2.4 Environment

- System `xgboost` crashes on import (bus error). The working copy is in `.deps/`, so use `PYTHONPATH=.deps`.
- CUDA training works through the native API (1.5 s vs 5.7 s on CPU for a 200k×200 test).
- `sklearn`, `lightgbm` and `coverage` are not installed.

## 3. Economic graph

Built by Codex (13 September) in `src/graph/economic_graph.py` with `src/graph/business_text_profiles.py` and
`configs/economic_graph.json`. Artifacts: `output/economic_graph_refined/YYYY-MM/` (140 months).

### 3.1 Construction [CS]

- **Nodes:** one node per `permco` per month; the representative security is chosen by primary flag, then market equity, then PERMNO.
- **Candidates:** GICS gates the candidate set, using the narrowest tier (subindustry → industry → group → sector,
  FF49/SIC fallback) that supplies the requested number of peers.
- **Similarity components** (each in [0, 1]):

  | Component | Definition | Weight |
  |---|---|---:|
  | Industry | Hierarchical GICS | 20% |
  | Business | Cosine similarity of 60-month hashed TF-IDF profiles from filing-body business language (issuer names removed; only filings dated ≤ t) | 42% |
  | Fundamentals | Gaussian similarity of sector-relative z-scores of 18 characteristics (be_me, sale_me, ocf_me, ebitda_mev, gp_at, op_at, niq_at, ebit_sale, at_turnover, cash_at, debt_me, tangibility, rd_sale, capx_gr1, at_gr1, sale_gr1, emp_gr1, oaccruals_at) | 33% |
  | Events | Recent (3-month) filing excerpts and item codes | 4% (capped at 10%) |
  | Returns | Positive pairwise-complete 36-month correlation; cannot admit a candidate | 1% (capped at 5%) |

- **Edges:** keep 15 peers; weights A_ij = softmax((s_ij − max s)/0.08), so rows sum to one, with no self-loops.
- **Outputs:** sparse adjacency, raw similarity, edge components, and peer-relative features X_i − Σ_j A_ij X_j with coverage.
  The candidate cache (`candidates.parquet`: top 75 plus top 25 per component) allows cheap re-weighting.

### 3.2 Evolution and audit [CS]

- **Version 1** (industry 45%, fundamentals 45%, returns 10%, 20 peers):
  - 83% of edges within the exact subindustry.
  - Strongest neighbor weight 5.3% vs 5% under equal weighting.
  - Removing returns changed the top peer for about half the firms.
  - Odd matches such as Amazon–Dillard's and Microsoft–Dolby.
- **Version 2 (current)** adds business-text similarity, softmax concentration and a much smaller return weight. Audit over all months:
  140 months, 528,793 nodes, 7,928,035 edges; 86.5% of edges within the exact subindustry; median strongest-neighbor
  weight 11.6% (vs 6.7% equal); top-three weight 30.3%; about 13.6 effective neighbors; business profiles for 62.9% of
  company-months. Dropping the return component keeps 93.2% of top-1 peers and 95.5% of top-5 memberships.
  In 2025-12, Visa–Mastercard (rank 1) and JPMorgan–Bank of America (rank 1) are correct, and Amazon–Dillard's and
  Microsoft–Dolby fall to rank 8. Ranks change by month: in 2026-08 they are 1, 4, 10 and 8, so both odd pairs are still among the 15 peers.
- **Caveats:** characteristics are used at month-end t without release timestamps, and GICS classification vintages are not
  certified. Business-text coverage rises from 24% of company-months (2015) to 87% (2025); 62.9% is the pooled figure.
- **Weights:** set as economic priors, not tuned on returns.
- **Superseded:** `output/economic_graph/` (version 1) was deleted.
- **Re-weighting:** the command-line re-weighting tool was deleted; re-weighting from the candidate cache now means calling
  `reweight_candidates` in `src/graph/economic_graph.py` directly.

```bash
OPENBLAS_NUM_THREADS=1 python -m src.graph.business_text_profiles
OPENBLAS_NUM_THREADS=1 python -m src.graph.economic_graph --config configs/economic_graph.json --save-candidates
```

## 4. Is the graph useful?

### 4.1 XGBoost feature ablation (Codex) [WF]

Same 523,125 rows, 147 base characteristics, top/bottom-100 portfolios, OOS 01/2021–08/2026. Folds follow the challenge schedule
(`xgb_baseline.expanding_folds`: train ≤ Y−3, validation Y−2..Y−1 with early stopping, test Y). The graph features are 18 peer-state and 18 idiosyncratic features:

| Variant | OOS R² | Rank IC [t] | Ann. top/bottom-100 | Sharpe |
|---|---:|---:|---:|---:|
| Base | −0.00160 | 0.0366 [4.09] | 10.1% | 0.32 |
| + peer state | −0.00149 | 0.0398 [4.49] | 16.3% | 0.62 |
| + idiosyncratic state | −0.00124 | 0.0365 [4.11] | 17.8% | 0.72 |
| + both | −0.00149 | 0.0383 [4.33] | 14.5% | 0.55 |

- **Validation reading (2019–2020):** peer and both variants raised R² by about 0.0006 but slightly lowered rank IC;
  idiosyncratic state was unchanged. Nothing in validation justified adding the graph features.
- **No paired improvement is significant:** IC change t ≤ 0.99, long–short change t ≤ 0.76, MSE gain t ≤ 0.82.
- **Decile long–short Sharpe falls slightly** with graph features (1.15 → 1.07/1.04/1.00). The top-100 gains are tail-driven.
- **The XGBoost baseline itself is weak:** its prediction has a negative slope once factors are controlled (investable, 6 months, 2021–2026: −2.85 [−5.7]).

### 4.2 Model-free graph tests: graph vs GICS peers vs random same-GICS peers with graph weights [FIXED]

| Test (architecture step) | Result | Verdict |
|---|---|---|
| **Peer basket as hedge** (comps, risk, neutralization) | Residual variance / market-demeaned variance, investable: graph 0.887 / 0.878, GICS 0.902 / 0.910, placebo 1.004 / 1.021. Graph − GICS −1.5pp [−4.6] / −3.2pp [−11.0] | **Graph helps**: 15 graph peers hedge better than ~115 GICS peers |
| **Next fundamental surprise** (seasonal SUE, no analyst data) | Next sales SUE, investable: graph peer slope 0.035 [4.8] / 0.040 [5.3]; graph − GICS IC +0.041 [8.6] / +0.045 [6.0]. Earnings SUE: GICS catches up in 2021–2026 | **Graph helps**, strongest for sales |
| Peer surprise → t+1 return | Nothing robust (investable earnings 0.10 [0.4] / 0.17 [0.7]) | No |
| Peer return spillover | All stocks, 3-month peer return 1.30% [4.4] / 1.78% [4.0]; investable not significant | Small caps only |
| Peer-relative fundamentals X − AX → return | Investable composite IC 0.014 [1.9] / 0.001 [0.1]; beats GICS in 0 of 18 features | No |
| Multi-scale bands (local, meso, macro) | Sign flips across periods and universes | No |
| Peer 8-K sentiment | Neither own nor peer tone predicts returns | No (weak input) |

### 4.3 Re-weighting the graph for surprise transfer [FIXED; variant chosen on 2015–2020, non-compliant]

- **Twelve variants** were rebuilt from the candidate cache and ranked by the slope of peer analyst surprises on the firm's next surprise.
- **Weights barely matter within the production limits:** every constrained variant is within about ±0.3 slope units of the default.
- **Fundamental similarity carries the transfer:** fundamentals-only peers P_recent 3.54 [4.9], events-only 2.03 [3.9], business-text-only 1.90 [3.2].
- **New peer signal:** peers' 8-quarter beat rate, investable 4.46 [9.1] / 3.82 [5.5]. Peer consensus revisions flip sign (−1.80 [−2.7] / 1.32 [2.2]).
- **Best variant:** `fundamental_heavy` (industry 15%, business text 25%, fundamentals 55%), slope 4.54 vs 4.48 for the default. A negligible gain.
- **Where the tuned variant was used:** the section 5.3 ablation and the section 6 Codex runs.
- **Decision:** kept code uses the untuned default graph; it gives the same surprise-model accuracy (section 6.1).

## 5. Sell-side expectations and the expectation gap

### 5.1 Point-in-time expectation state [CS] (kept in `src/graph/expectations.py`)

For each (permno, month-end t):
- **Snapshot and target quarter:** the latest consensus snapshot ≤ t (≤ 45 days old, median 14). The target is the earliest fiscal quarter
  **not announced by t** (`ANNDATS_ACT` ≤ t is the only use of that column).
- **Street features:**
  - consensus ÷ price, dispersion ÷ price
  - revision breadth (up − down)/n
  - 1- and 3-month consensus changes for the same quarter
  - staleness, recommendation level and 3-month change
  - last announced surprise ÷ price
  - a seasonal AR(1) own EPS forecast gap
- **Label:** actual − last pre-announcement consensus (surpsum) ÷ price, released after t and within 180 days of the quarter end.
- **Coverage:** 420k firm-months: mega 97%, large 96%, small 93%, micro 70%, nano 44%.
- **Analyst-error history:**
  - mean surprise over the last 4 and 8 quarters, 8-quarter beat rate, surprise volatility
  - consensus-implied growth vs the same quarter last year
  - 3-month revision of the following quarter
  - days past the firm's usual report date

**Sanity** [FIXED], rank IC with the eventual surprise, investable: last surprise 0.230 [21.1] / 0.274 [19.2]; breadth
0.050 [7.0] / 0.083 [9.2]; 3-month revision 0.027 [2.1] / 0.089 [5.4]; recommendation change −0.020 [−4.8] / −0.015 [−3.9].

### 5.2 Peer information vs the Street [FIXED]

Peer shock = graph-weighted mean of peers' latest analyst surprises (announced within 92 days):
- `recent`: all fresh peers.
- `aligned` (eom): the peer's quarter ends within 46 days of the firm's unreported quarter.
- `snap`: aligned, and the peer announced before the firm's consensus snapshot.

| Question | Finding |
|---|---|
| Does the peer shock predict the firm's next analyst surprise beyond its own Street signals? | Yes. Investable slopes: snap 0.032 [3.0] / 0.019 [1.9]; aligned 0.057 [6.9] / 0.066 [6.4]; recent 0.048 [7.4] / 0.055 [5.7] |
| Graph vs GICS peers | For `recent` (previous-quarter errors) graph dominates: 0.043 [8.0] / 0.049 [6.8] vs GICS 0.010 [1.7] / 0.001 [0.1]. Same-quarter transfer: equal-weight GICS peers do as well. In large and mega caps GICS peers carry no surprise information; in nano and micro both do |
| Do analysts react to peer news? | Barely: correlation of peer shock with the firm's 3-month revision 0.02 (β ≈ 0.009), and no catch-up before release. The underreaction gap P − βR ≈ P. Partial reaction in 2021–2026: slope of past revision on the snap shock, investable, −0.014 [−1.2] / 0.048 [2.4] |
| Own seasonal forecast gap | Predicts surprises (investable 0.047 [6.1] / 0.037 [3.7]) and complements the peer shock, but has **negative** sector-neutral returns in investable stocks (−0.36 [−2.4] / −0.38 [−2.1]). The return sign flips by size: positive in nano and micro (nano, 6 months, 14.4 [1.9] / 14.8 [6.3]), negative in small, large and mega (mega, 6 months, −3.32 [−2.9] in 2021–2026) |
| Next-month returns | All stocks, snap shock: 2.10% [4.5] / 1.56% [2.6]. Investable: 0.85% [1.9] / 0.59% [0.7] |
| Longer horizons | Investable aligned-shock sector-neutral spread: 3 months 1.20 [1.9] / 1.86 [3.3], 6 months 1.90 [1.8] / 3.81 [3.9], 12 months 2.66 [2.5] / 2.58 [1.6]. Previous-quarter peer surprises predict the firm's surprise two quarters ahead (graph 4.25 [9.4] / 3.48 [6.8]; GICS ≈ 0) |
| By size | Surprise prediction works in every group except nano. Return effects concentrate in micro and small; large and mega vanish after controls |
| By year | Surprise prediction positive in 12 of 12 years. The investable 6-month aligned-shock return spread is positive in 10 of 12 years but uneven: 2020 alone +14.0%; excluding 2020 the average falls from 3.40% to 2.35%, and 2024–2025 average 1.62%. **The `recent` shock reversed in investable stocks:** 6-month Q5−Q1 2024 −2.41 [−2.9], 2025 −6.13 [−4.5] (Fama–MacBeth −3.99 [−4.3] and −7.27 [−4.7]), consistent with the 2024 and 2026 strategy losses |

**Caveats:** `snap` covers only 13% of firm-months (months with fewer than 100 observations are skipped); one placebo seed; no trading costs.
Open item: replace the seasonal AR(1) own forecast with a characteristic-based EPS model (the forecast gap is the weakest input).

### 5.3 Company representation ablation [FIXED; annual refits from 2017, no validation window]

Gradient boosting on the within-month rank of the next analyst surprise, with the tuned `fundamental_heavy` graph (section 4.3). Investable rank IC, 2017–2020 / 2021–2026:

| Features added | Count | Investable rank IC |
|---|---:|---:|
| Street state | 13 | 0.261 / 0.303 |
| + analyst-error history | 20 | 0.299 / 0.342 |
| + graph peer signals | 24 | 0.306 / 0.345 |
| + 147 characteristics and seasonal forecast gap | 173 | 0.312 / 0.357 |
| + 8-K item counts, days since 2.02, Reg FD after 2.02, FinBERT | 194 | 0.313 / 0.356 |

- **Analyst history is the largest gain:** +0.038 [9.7] / +0.039 [9.6], positive in every size group. The 4-quarter mean surprise has 22% of total gain and the last surprise 12%.
- **Graph peers add a little:** +0.007 [5.3] / +0.003 [3.7] (vs GICS peers +0.005 [3.9] / +0.002 [2.1]; none in mega caps).
- **Fundamentals add a little:** +0.006 [2.8] / +0.012 [6.4] (micro +0.021; mega ≈ 0).
- **8-K events add nothing:** +0.001 / −0.001.
- **Returns vanish after factor controls** (investable, 6 months): the predicted surprise earns 3.34 [1.1] / 6.23 [3.2] with Street controls,
  but 0.41 [0.4] / 0.66 [0.4] after the factor controls. It survives only in micro (12.13 [4.2] in 2021–2026), nano, and large caps in 2021–2026.
  These were read from the test period and are not established.

## 6. Surprise model and monthly strategies

Codex, 14 September. Monthly data only.

### 6.1 Walk-forward surprise model [WF] and [WF-0]

- **Features:** the section 5.3 representation without 8-K events (Street state, analyst history, graph peers, 147 characteristics and forecast gap).
- **Model:** XGBoost with fixed parameters, refit once per test year.
- **Training rows:** holding month ≤ the training end and surprise label announced by then.

| Test year | Train end [WF] | Validation IC [WF], tuned graph variant | Test IC [WF], tuned graph variant | Test IC [WF], kept module (default graph) | Test IC [WF-0], tuned graph variant | Test IC [WF-0], kept module |
|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 2018-12 | 0.288 | 0.356 | 0.351 | 0.371 | 0.370 |
| 2022 | 2019-12 | 0.313 | 0.294 | 0.293 | 0.291 | 0.291 |
| 2023 | 2020-12 | 0.330 | 0.287 | 0.285 | 0.294 | 0.292 |
| 2024 | 2021-12 | 0.286 | 0.291 | 0.293 | 0.304 | 0.303 |
| 2025 | 2022-12 | 0.296 | 0.311 | 0.312 | 0.315 | 0.315 |
| 2026 | 2023-12 | 0.308 | 0.312 | 0.311 | 0.313 | 0.316 |
| Mean | | | 0.308 | 0.307 | 0.315 | 0.315 |

The Codex runs (tuned `fundamental_heavy` graph variant) and the kept `src/graph/expectations.py` (untuned default graph) agree within
0.01 every year. The kept-module columns were rerun on 14 September after two timing fixes found in the cleanup review:
- the training label is now ranked within month among that fold's training rows only (before, 4–8% of training rows were
  ranked alongside labels announced after the training end);
- validation rows now also need their label announced by the validation end (before, about 5% of validation labels were
  announced in the test year).

Test ICs moved by at most 0.004. Validation ICs of the kept module after the fixes: 0.286, 0.322, 0.337, 0.289, 0.298, 0.317.

### 6.2 Monthly surprise portfolio [WF], January 2021–May 2026

- **Universe and screens:** the small size group; `common`, `primary_sec` and `exch_main` flags; price ≥ $5; dollar-volume percentile ≥ 0.20 within the month.
- **Signal:** the predicted surprise, residualized each month by OLS on the ranks of the 13 factor characteristics plus `ret1` and
  `ret_12_1`, and sector dummies. Unlike the section 1 "factor controls", last surprise, revision and breadth are not included.
- **Portfolio:** 50 longs / 50 shorts with sector-matched seat quotas, 100% long / 100% short (zero net, 200% gross), rebalanced monthly.
  Months that cannot fill 50 names per side count as zero return (this matters only for the event sleeves).
- **Costs:** (long turnover + short turnover) × cost rate each month, at 10, 25 and 50 bp.
- **Evaluation:** 65 of the 68 test months (I/B/E/S ends 05/2026).
- **Not a clean holdout.** The small-cap universe and factor-neutral design were motivated by [FIXED] readings that included
  2021–2026 (small caps "clear both periods"; small-cap aligned shock with factors 0.91 [0.6] / 5.49 [3.0]), and the model uses
  the tuned `fundamental_heavy` graph. Even 6.24% is optimistic.

| Specification | Active / evaluated months | Gross annual return | HAC t | Sharpe | After 25 bp turnover cost |
|---|---:|---:|---:|---:|---:|
| Raw aligned peer shock, small | 39 / 65 | 10.64% | 2.27 | 0.84 | 5.17% |
| Raw aligned peer shock, investable | 51 / 65 | 9.33% | 1.62 | 0.70 | 3.59% |
| Unneutralized surprise, small | 65 / 65 | 12.71% | 1.57 | 0.73 | 10.29% |
| **Factor-neutral surprise, small (primary)** | **65 / 65** | **6.24%** | **0.89** | **0.41** | **3.57%** |
| Factor-neutral surprise, investable | 65 / 65 | 0.87% | 0.13 | 0.06 | −1.69% |
| Factor-neutral small, expected-report months only | 44 / 65 | 0.85% | 0.22 | 0.08 | −5.15% |

**Primary portfolio:**
- cumulative return 31.7%, maximum drawdown −22.9%
- cash + 4% information ratio 0.15 (a flat 4% hurdle, valid only if the capital earns T-bills)
- turnover 42% (long) / 47% (short); after costs 5.17% at 10 bp, 3.57% at 25 bp, 0.91% at 50 bp
- ex-ante beta gap 0.004, but realized beta −0.25 against the S&P 500 price index's monthly change with an annualized intercept of 9.75% [1.55].
  On the challenge's excess-return basis (S&P change minus TB3MS/12) the intercept is 8.9% and beta −0.24.

**Other specifications:**
- **Unneutralized:** carries an ex-ante beta gap of 0.224.
- **Raw aligned peer-shock sleeve:**
  - cannot fill 100 names in 26 months
  - turns over about 90% per leg
  - was strongest in 2021 (4.86% a year from 2022)

**No-validation rerun [WF-0], primary portfolio:**
- 8.06% [1.12], Sharpe 0.57, cash + 4% information ratio 0.29
- cumulative 46.7%, drawdown −21.4%
- 5.42% after 25 bp costs
- beta −0.27 against the price index; on the excess basis intercept 11.0%, beta −0.26

By year: 2021 +33.6%, 2022 +19.5%, 2023 +4.4%, 2024 −10.4%, 2025 +19.9%, 2026 (to May) −23.4%.
2022–2025: 8.36% [1.22]; January 2022–May 2026: 2.29%.

**Standing rule from this run:** do not optimize this monthly design further on these test years. Daily announcement-window
analysis can inform a new frozen design, but cannot retroactively promote this monthly out-of-sample result.

### 6.3 Graph hedge, return heads, holding horizon, beta overlay [WF-0]; diagnostic, test years already seen

| Idea | Result | Decision |
|---|---|---|
| Direct graph-peer hedge (half of each position against its peer basket, ≤ 500 names, 200% gross) | 1 month: return 8.06% → 2.41%, volatility 14.25% → 8.41%, beta −0.272 → −0.121; after 25 bp −0.39% | Use graph peers in the covariance/risk model, not as a 50/50 hedge |
| Return head on factor-residual targets | Rank IC 1 month 0.0194 [3.16], 3 months 0.0371 [5.45], 6 months 0.0369 [2.79], but every real portfolio lost money | Reject as built |
| Return head on graph- and factor-residual targets | 6-month IC 0.0428 [5.93]; best continuous factor-neutral portfolio −3.18% [−0.70] | Reject as built |
| 3- and 6-month staggered holding | 4.63% [0.78] and 3.78% [0.74] vs 8.06% at 1 month | Reject |
| Lagged 24-month rolling-beta overlay (≥ 12 prior months): hedged return r_t − β̂·m_t, β̂ = OLS cov/var of the book on the S&P 500 price-index change over the prior 24 months, estimated through t−1 | 10.12% [1.45], volatility 13.80%, beta −0.099 | Keep as a risk-control candidate; hedge costs not included; first 12 months unhedged; recheck on the excess-return basis |

Winsorized residual targets hid adverse small-cap tail returns; real portfolio accounting restored them.

## 7. Valuation: analyst-, price- and graph-implied value

### 7.1 Sector divergence models, version 1 [WF]

Fixed parameters: cost of equity 9%, WACC 8%, long growth 2.5%.

| Bucket (GICS) | Analyst-implied | Price-implied | Graph-implied |
|---|---|---|---|
| Banks (4010) | Residual income on forward ROE (next-4Q consensus EPS / book) | ROE priced by P/B | Peer P/B × book; peer ROE |
| Software (4510) | 2-stage FCF DCF at analyst growth | Growth priced by EV/FCF | Peer EV/Sales × sales; peer growth |
| Industrials (20) | FCF DCF | Implied growth | Peer EV/EBITDA × EBITDA |
| Consumer (25, 30) | FCF DCF | Implied growth | Peer forward P/E × forward EPS; margin vs peers |
| Energy (10) | FCF DCF | Implied growth | Peer EV/EBITDA; FCF yield vs peers (no commodity data) |

- **Candidates:** 312 (universe × sector × signal × 1/3/6/12-month horizon).
- **Selection each test year:** training |HAC t| ≥ 2 on the factor- and value-controlled slope, with validation agreeing in sign.
  15, 26, 22, 19, 17 and 21 were selected in 2021–2026.

| Out of sample, 2021–08/2026 | Controlled slope | Raw long–short spread |
|---|---:|---:|
| Investable, 1 / 3 / 6 / 12 months | 0.48 [0.94] / 0.84 [0.77] / 4.94 [2.26] / 4.45 [0.89] | 0.03 [0.11] / −0.96 [−1.28] / 0.56 [0.53] / −0.23 [−0.12] |
| Micro, 1 / 3 / 6 / 12 months | ≈ 0 | 1.53 [3.62] / 2.22 [2.84] / 2.71 [2.94] / 6.49 [2.64] (mostly the short leg; factor exposure) |
| Mega (margin gap) | negative | −6.94 [−3.82] (3 months), −13.11 [−7.35] (6 months) |

- **Consumer comps reversal:** the only candidate re-selected every year (cheap vs peers' forward P/E underperforms).
  Controlled slope 10.01 [3.90] at 12 months, but raw spread −0.81 [−0.32]. At most a feature in a factor-neutral model.
- **Fixed-split favorites not selected:** the peer growth-gap signal, industrials and a bank ROE short, which looked good in the
  earlier fixed-split reading, were never selected in any test year (training |t| < 2, too few labels, or validation disagreed in sign).
- **Large caps:** one candidate, 6 months: controlled 7.77 [4.00], raw spread 5.81 [2.25] over only 15 months; at 12 months −8.33 [−2.71].
  **Small caps:** never selected.

### 7.2 Why version 1 measured Street expectations badly

1. **Fixed discount rates.** Solving for growth at a common 8–9% discount rate turns "price-implied growth" into mostly risk and
   rate differences. Practice holds the analyst earnings path fixed and solves for the implied expected return.
2. **Mixed cash-flow basis.** Free cash flow was grown at one year of EPS growth. The DCF was unavailable for about 40% of firms
   (63% in energy) because FCF was negative or missing, and growth rates barely persist.
3. **Raw comps are the value factor.** Peer multiple × own metric, without adjusting for growth, profitability and risk.
4. **Consensus taken at face value**, despite long-horizon optimism and short-horizon walk-downs.
5. **Levels, not changes.**
6. **Thin inputs:** no FY1/FY2 or long-term growth, no revenue/EBITDA estimates, no price targets, no commodity curves.

How practitioners do it:
- **Sell-side:** 3-statement models, 2–3 years of explicit forecasts, a 12-month price target on a sector-convention multiple (banks P/TBV vs ROTCE, software EV/NTM revenue vs growth, energy NAV at the strip).
- **Fundamental buy-side:** variant perception on key drivers, and reverse DCF checked against base rates.
- **Quants:** risk-neutralized earnings yield, revisions, SUE, implied cost of capital, warranted multiples and bias-corrected forecasts.

### 7.3 Street-implied expectations, version 2 [WF]

Built from current data with no time-series fitting: every month uses only its own cross-section and fixed priors.

- **Implied cost of capital (ICC):** residual income with forward ROE fading to the industry ROE by year 12 (analyst ICC);
  a fundamentals-only ICC with trailing ROE mean-reverting at 38% a year; analyst ICC minus the beta-required return
  (T-bill lagged one month + β × 5%); analyst minus fundamentals ICC (optimism gap); ICC vs graph peers.
- **Warranted multiples:** monthly regressions of log EV/Sales, P/B, forward P/E and EV/EBITDA on margin, growth, ROE,
  volatility, beta, R&D and size with GICS fixed effects. The residual is the signal; there is also a composite and a graph-relative version.
- **Base-rate feasibility:** growth implied by EV/FCF at a firm-specific WACC, scored against the same month's realized 3-year sales growth by size and sector.
- **Changes:** 3-month changes of the main signals.

**Diagnostics:**
- **Panel:** 513,157 firm-months; investable coverage analyst ICC 70%, fundamentals ICC 92%, warranted composite 92%, feasibility 65%.
- **ICC tracks rates:** median analyst ICC 6.5–8.0% in 2015–2021, 9.0% in 2022, 7.7–8.8% afterwards.
- **Correlated with value:** analyst ICC vs book-to-market ρ = 0.47; warranted composite vs book-to-market ρ = 0.61.

**Walk-forward selection:** 384 candidates per year. Investable selections: 2, 5, 0, 7, 3 and 3 in 2021–2026.

| Universe | Horizon | Months | Controlled slope | Raw spread |
|---|---:|---:|---:|---:|
| Investable | 1 | 53 | −0.09 [−0.39] | 0.28 [0.65] |
| Investable | 3 | 24 | −0.56 [−0.64] | −1.60 [−3.39] |
| Investable | 6 | 12 | 0.99 [0.54] | −2.27 [−2.39] |
| Investable | 12 | 45 | 6.56 [1.07] | −9.08 [−2.88] |
| Micro | 1 | 65 | 0.28 [0.51] | 0.80 [2.21] |
| Micro | 3 | 65 | 2.72 [1.40] | 3.49 [5.58] (short leg 2.81 [7.45]) |
| Micro | 6 | 63 | −0.26 [−0.11] | 3.19 [2.26] |
| Micro | 12 | 57 | −1.04 [−0.24] | −0.79 [−0.17] |

**Verdict:**
- **Better description, no alpha.** The version 2 models describe market expectations better, but produce no investable alpha;
  longer-horizon raw spreads have the wrong sign out of sample.
- **Where it still helps:** structured context for the research/committee agents, or candidate inputs to a factor-neutral model.
- **What more data would add:** richer forward estimates would improve the measurement, not necessarily returns.

## 8. Alpha search with the existing monthly data [WF]

Claude Code, 14 September. The question: with the data already in hand, is there a return signal the earlier sections did not test?
The scripts and small result files are archived outside the repository in `~/fiam26_alpha_search_20260914.tar.gz`
(`run.log` is the discarded first run, see 8.5).

### 8.1 What had not been tested

- **Return signals from the graph:** residual reversal (1 and 3 months) and residual momentum (12 months) relative to graph peers,
  compared with GICS-6 peers. Also peer returns as lead–lag signals.
- **Street variables as direct return signals.** Earlier they only predicted surprises. Tested: dispersion, revision level and breadth,
  recommendation changes, last surprise, 8-quarter beat rate, days overdue, announcement expected next month,
  recommendation vs 3-month price move.
- **8-K events as return signals.** Earlier they were only surprise-model features. Tested: item counts over months t−2..t for
  3.02 dilution, 2.03 new debt, 5.02 officer change, 1.01 material agreement, 2.05 restructuring, 4.02 non-reliance,
  3.01 delisting notice, 7.01 Reg FD, plus filing intensity.
- **Graph structure:** peer-set churn (1 − Jaccard overlap with the peers of 12 months earlier) and centrality (in-strength).
- **Confirmation:** last surprise kept only when graph peers surprised in the same direction; peer surprise and peer beat rate.
- **Size:** every signal separately in micro, small, large, mega and investable, at 1, 3 and 6 months.
- **Tradability:** raw (unwinsorized) returns, decile books, turnover costs, and a factor-neutral construction.

### 8.2 Method

- **Panel:** 354,681 stock-months after contemporaneous screens (price ≥ $5, top 80% by dollar volume).
- **Candidates:** 31 signals × 5 universes × 3 horizons = 465.
- **Monthly statistics:**
  - Fama–MacBeth slope of the within-sector signal rank on winsorized h-month returns. Controls: the factor controls,
    five value multiples (intrinsic value/ME, EBITDA/EV, sales/ME, FCF/ME, NI/ME) and the return/Street controls.
  - Sector-neutral quintile long, short and spread legs on raw returns.
- **Selection:** `src/graph/walk_forward.py` for each test year:
  - trade direction from the training mean
  - training |HAC t| ≥ 2
  - validation mean agrees in sign
- **Selected per year:** 28, 34, 36, 50, 43 and 42 of 465 in 2021–2026. If candidates were independent, about 11 a year would pass
  by chance. They are strongly correlated across horizons and universes, so the real false-positive count is unknown.
- **Decile book:**
  - composite of the signals selected for the test year (training direction, within-sector ranks, all horizons or 1 month only)
  - top and bottom sector-neutral deciles, equal weight
  - monthly rebalance, raw next-month returns
  - cost = 2 × turnover × cost rate; realized beta on the S&P 500 price-index change (the section 6.2 convention)
- **Factor-neutral book:** the same composite traded as the pure-characteristic portfolio of a monthly OLS on the score, sector
  dummies and all controls. It has zero exposure to every control and sector, 200% gross, raw returns.
- **Ties:** most firms have no event, so event counts are tied at zero. Ties are broken by a fixed hash of permno
  (see 8.5 for why row order failed).

### 8.3 Results

**Signals selected in at least 5 of 6 test years.** Direction − means high values underperform. Values are % over the horizon,
top vs bottom rank (controlled slope) or top vs bottom quintile (raw spread); brackets are HAC t-statistics on the test months.

| Universe | Horizon | Signal | Direction | Years | Controlled slope | Raw spread |
|---|---:|---|:---:|---:|---:|---:|
| Large | 6 | Non-reliance on past financials (4.02) | − | 6 | 5.51 [5.58] | 0.88 [1.20] |
| Large | 1 | Graph-peer beat rate | − | 6 | 0.51 [1.82] | 0.24 [0.74] |
| Small | 1 | Delisting notice (3.01) | − | 6 | 0.72 [1.86] | 0.10 [0.47] |
| Small | 3 | Delisting notice (3.01) | − | 6 | 1.79 [2.11] | 0.34 [0.62] |
| Micro | 3 | Residual 3-month reversal vs graph peers | + | 6 | 0.83 [0.63] | 2.10 [2.17] |
| Micro | 6 | Residual 3-month reversal vs graph peers | + | 6 | 0.18 [0.08] | 1.41 [0.84] |
| Large | 3 | Days overdue | − | 5 | 1.86 [2.44] | −0.49 [−1.12] |
| Large | 6 | Days overdue | − | 5 | 2.46 [3.40] | 0.01 [0.02] |
| Large | 3 | Delisting notice (3.01) | − | 5 | 2.20 [2.03] | 0.07 [0.18] |
| Large | 6 | Delisting notice (3.01) | − | 5 | 2.38 [1.78] | 0.61 [0.69] |
| Large | 1 | New debt (2.03) | − | 5 | 0.39 [1.44] | −0.23 [−1.36] |
| Mega | 6 | Delisting notice (3.01) | − | 5 | 1.16 [1.09] | 0.98 [1.40] |

Strong micro-cap candidates selected in 4 years: 8-quarter beat rate at 6 months, 4.91 [2.42] (raw spread 7.43 [5.12]);
graph-peer 3-month return at 3 months, 4.47 [3.32] (raw spread 2.68 [1.82]).

**Average of selected candidates by universe.** Raw spreads are significant only in micro caps:
3 months 2.81 [3.39], 6 months 3.57 [4.57]. Investable, large and mega raw spreads have |t| ≤ 1.43; small caps at 6 months 3.72 [1.51].

**Books built from the selections** (2021–2026 test months; annual returns; "all" = signals selected at any horizon):

| Universe | Signals | Decile book | After 25 bp | Turnover | Beta | Factor-neutral book | After 25 bp |
|---|---|---:|---:|---:|---:|---:|---:|
| Investable | 1 month | 0.5% [0.16], 32 months | −0.1% | 10% | −0.10 | 0.4% [0.08] | −4.7% |
| Investable | All | 2.3% [0.75] | −0.2% | 41% | 0.20 | 2.0% [1.13] | −2.0% |
| Large | 1 month | 2.9% [0.63] | 0.7% | 36% | −0.04 | 2.1% [1.37] | −1.2% |
| Large | All | 4.1% [1.09] | 1.2% | 48% | −0.16 | 1.3% [0.88] | −3.1% |
| Small | 1 month | 0.5% [0.16] | −0.6% | 19% | −0.06 | 4.3% [0.51] | −1.7% |
| Small | All | 7.8% [1.08] | 5.1% | 45% | 0.34 | 3.4% [1.14] | −0.9% |
| Micro | 1 month | 3.7% [0.32], 26 months | −1.4% | 84% | 0.20 | 2.2% [0.58] | −5.3% |
| **Micro** | **All** | **26.0% [2.80], 44 months** | **21.6%** | **74%** | **0.42** | **10.1% [2.48]** | **3.7%** |
| Mega | All | too few names | | | | 2.3% [1.55] | −2.4% |

Decile-book turnover is the share of names replaced per side each month.
Factor-neutral turnover is one-way weight turnover (54–125% a month); its costs use the weights actually traded.

### 8.4 Robustness of the two best books

**Micro caps, all signals (decile book, 26.0% [2.80]):**
- **One year:** 2025 +65.7%. Without 2025: 11.1% [1.31].
- **Short-driven:** the long leg returned 6.2% a year; the shorts fell 19.8% a year.
- **Coverage:** active in 44 of 68 months, because it needs about 500 scored names for 50 per side (5 months in 2021, 1 in 2022).
  With a $10 price floor there are never enough names.
- **Factor exposure:** regression on same-universe sector-neutral tercile spreads (size, value, momentum, short-term reversal,
  low idiosyncratic volatility, illiquidity) and the S&P. Short-term reversal loading 0.68, market 0.28, R² 0.25; alpha 15.9% [1.83].
- **Leave one signal out:** 21.7%–29.1% (t 1.93–3.27); without 2025, 6.0%–15.9% with no t above 1.8.
- **Factor-neutral version:** 10.1% [2.48] but 3.7% after costs and 5.2% [1.38] without 2025.
- The raw micro-cap short legs in the valuation work (section 7.3) showed the same pattern.

**Small caps, all signals (decile book, 7.8% [1.08]):**
- 2022 −23.4%, 2025 +31.6%. Without 2025: 2.7% [0.46]. Beta 0.34.
- **Factor exposure:** illiquidity 0.45, value 0.37, low volatility −0.56, size 0.20; R² 0.44; alpha 7.5% [1.58].

### 8.5 A tie-breaking artifact (first run discarded)

The first run broke tied ranks by row order. Rows are sorted by permno, which tracks listing age, so for sparse event counts the
quintile and decile legs were partly an age sort. With a stable random tie-break:
- small-cap 1-month book: 4.52% [1.35] (3.43% after costs) → 0.5% [0.16]
- large-cap non-reliance at 6 months, raw spread: −2.38 [−5.81] → +0.88 [1.20]
- large-cap days overdue at 6 months, raw spread: −1.96 [−3.81] → 0.01 [0.02]

A random tie-break re-drawn every month is also wrong: it made the small-cap 1-month book turn over 81% a month instead of 19%.
Any future sort on count or binary signals needs a fixed, age-independent tie-break.

### 8.6 Conclusions and what is missing

- **No investable edge.** No small, large, mega or investable book is significant, with either construction, before or after costs.
- **Controlled slopes do not become returns.** Non-reliance notices, overdue reports and delisting notices have significant
  factor-controlled slopes, but decile spreads and factor-neutral portfolios on raw returns do not pay. The factor-neutral book
  neutralizes the same controls, so the main remaining difference is winsorizing. That suggests the tails the slope ignores
  are real returns (the same lesson as section 6.3).
- **Events are sensible, rare shorts.** Restatements, delisting notices, late filers and new debt make economic sense but touch a
  few names a month. That is too sparse for a 100–500 name book; use them as risk flags or short vetoes in the committee.
- **The micro-cap book is not a strategy.** It is short-term reversal plus a 2025 episode, driven by shorts in $5–10 micro caps,
  with 74% monthly turnover and no borrow data.
- **Graph return signals stay weak.** Residual reversal vs graph peers, peer churn and centrality are selected only in micro caps.
  This agrees with section 4: the graph is for risk and information transfer, not return signals.

**Where the project lacks something, in order of importance:**
1. **Event-time data.** The only validated asset, the surprise forecast (IC ≈ 0.3), is about earnings announcements.
   A month-end sort mixes a few days of announcement reaction with about 20 days of noise. CRSP daily returns and
   announcement dates are needed to test and trade it (section 11, item 1).
2. **Trading-cost and borrow data.** Every result that looks strong is short-driven and in small or micro caps, exactly where
   spreads, borrow fees and short availability are not observed.
3. **Statistical power.** The first test year trains on four years of data, and hundreds of correlated candidates pass or fail a
   hard |t| ≥ 2 cut. Fewer, pre-specified hypotheses, or shrinking and combining candidates instead of selecting, would waste less signal.
4. **A broad cross-sectional signal.** A 100–500 name neutral book needs a score that ranks the whole investable cross-section.
   What has any power here is sparse (events) or confined to micro caps. The surprise forecast is the closest candidate;
   its return channel is still untested at event time.
5. **Sales and margin expectations** (I/B/E/S SAL) are still unmeasured.

## 9. Daily event study: was it already priced in? [exploratory, Yahoo prices]

Claude Code, 15 September. The question: do the graph and sell-side signals carry real information that the
market had already priced, which would explain why monthly returns showed no edge? Scripts and results are
archived in `~/fiam26_event_study_20260915.tar.gz`; the price cache is `local/yahoo_daily/` (git-ignored).

### 9.1 Data and method

- **Prices:** Yahoo Finance adjusted daily closes via `yfinance` for every panel ticker (7,450 symbols),
  mapped month by month using the panel's point-in-time tickers.
- **Validation against CRSP:** a stock-month is kept only if its compounded Yahoo return matches the panel's
  CRSP monthly return within 1 percentage point. This keeps 288,292 of 529,082 stock-months (54%).
  - In an 800-stock coverage check, Yahoo had 90% of months for stocks still trading in 08/2026 but only
    8.5% for stocks that had stopped trading, so **results are survivorship-biased**.
  - When Yahoo has a month, it matches CRSP within 0.1 pp 96.5% of the time.
- **Abnormal return:** stock return minus the equal-weight return of validated stocks in the same size group that day.
- **Events:** 113,681 I/B/E/S quarterly announcements with price ≥ $5. The forecast is the walk-forward
  out-of-sample predicted surprise (2021–2026) formed at the last month-end before the announcement
  (median 11 trading days earlier for investable stocks). Day 0 is the announcement date.
- **Windows:** before the forecast (62 and 20 trading days), forecast to 2 days before the announcement,
  announcement (days −1 to +1), and drift (+2 to +21, +2 to +63).
- **Statistic:** monthly cross-sectional slope of the window return on the signal's rank (% from bottom to
  top rank), averaged with Newey–West t-statistics.
- **Information transfer:** a firm's return in days −1..+1 around a graph peer's announcement, regressed
  on the peer's surprise rank, compared with same-GICS-6 firms that are not graph peers. Pairs where the
  firm itself announces within 3 trading days are dropped.
- **Nothing was selected or fitted.** The windows were fixed before the run, but the 2021–2026 test years
  have now been read, so any strategy suggested here must be validated elsewhere (CRSP, new design).

### 9.2 Results

**Investable stocks** (brackets: t-statistics; realized surprise uses 2015–2026, forecast rows use 2021–2026):

| Signal | 62 days before forecast | Forecast → day −2 | Announcement −1..+1 | Drift +2..+63 |
|---|---:|---:|---:|---:|
| Realized surprise (not known in advance) | 2.69 [4.5] | 0.87 [5.4] | **8.92 [26.8]** | 1.24 [2.0] |
| Predicted surprise | **3.71 [4.0]** | 0.65 [1.6] | 0.49 [1.2] | **2.92 [2.8]** |
| Predicted, holding the realized surprise fixed | 3.13 [3.8] | 0.21 [0.5] | **−3.61 [−7.6]** | 2.40 [2.4] |
| Predicted, holding realized surprise and the 62-day move fixed | | 0.29 [0.7] | −3.64 [−7.4] | 2.23 [2.2] |
| Graph peer shock, aligned quarter | 1.34 [1.9] | 0.20 [0.9] | 0.66 [2.5] | 1.48 [1.7] |
| Graph peer shock, recent | 0.88 [1.5] | 0.25 [1.4] | 0.42 [2.1] | 0.62 [0.8] |

**Announcement return (%) by forecast quintile, investable, 2021–2026:**

| Actual outcome | Q1 (predicted miss) | Q3 | Q5 (predicted beat) |
|---|---:|---:|---:|
| Beat | +1.42 | +1.25 | +1.22 |
| Miss | −2.45 | −3.84 | −5.51 |

By size, the "holding realized fixed" announcement slope is −3.59 [−6.9] in small, −2.83 [−6.5] in large and mega, and
−2.76 [−5.5] in micro caps. The unconditional announcement slope is positive only in 2021 (1.44 [2.8]).
The +2..+63 drift is 3.03 [2.8] in large and mega (2.79 [2.5] with both controls), 2.17 [1.3] in small and 3.10 [2.4] in micro.

**Information transfer on a peer's announcement days (−1..+1):**

| Followers | Graph peers | Same GICS-6, not graph peers |
|---|---:|---:|
| Investable | 0.197 [5.6] | 0.012 [0.3] |
| Large and mega | 0.197 [5.0] | −0.023 [−0.5] |
| Micro | 0.234 [3.3] | 0.092 [1.5] |

In the next 20 trading days, graph followers drift a further 0.17 [1.1] in investable stocks but 0.70 [2.6] in micro caps.

### 9.3 What this means

- **Yes, largely priced in.** The model predicts analysts' errors, not the market's.
  - **Before the forecast:** stocks had already moved 3.7% (bottom to top rank) in the forecast's direction in
    the quarter before it was formed. This is partly mechanical, because the model's inputs include past returns.
  - **At the announcement:** at the same actual surprise, a stock we predicted to beat moves 3.6% less.
    The market's own expectation already contains what the model knows beyond consensus: predicted beaters get
    little extra for beating and are punished hardest for missing (−5.5% vs −2.5%).
  - **Peer news:** graph peers react within the announcing peer's 3-day window, and in investable stocks
    there is no significant follow-through. Same-industry firms that are not graph peers do not react at all.
- **New, and worth a proper test:**
  - **Post-announcement drift linked to the forecast:** about +2.9% from bottom to top rank over the 63 trading days
    after the announcement, surviving controls for the actual surprise and the pre-forecast move, and present in
    large and mega caps. It could still be factor exposure (only size is adjusted), and the Yahoo sample lacks
    delisted stocks.
  - **The graph is a real information network:** it identifies the firms that reprice when a peer reports, which
    GICS does not. Use it for event-day risk and hedging, and for the committee's "who is affected" context.
  - **Asymmetric earnings risk:** predicted beaters carry a larger downside on a miss. A risk flag, not alpha.
- **What not to do:** trade "consensus is wrong" into the announcement; the market is already there.

### 9.4 Trading the drift and "not yet repriced" signals [WF] and [WF-0]

Monthly strategies on CRSP monthly returns (no Yahoo, so no survivorship filter). At the end of month T:
- **Reporters** (firms that announced in T), ranked within sector:
  - forecast drift: the forecast
  - surprise drift: the realized surprise
  - surprise not repriced: surprise minus the stock's month-T move
  - forecast not repriced: forecast minus the month-T move
  - forecast vs result: forecast minus surprise
  - forecast + surprise
- **Non-reporters:** graph peer shock minus the stock's own month-T move, for all non-reporters or only those reporting in T+1.
- **Portfolios:** top vs bottom 20%, equal weight, held 1 or 3 months as staggered sleeves; four size universes (64 candidates).
- **Forecast history for selection:** out-of-sample forecasts for 2017–2020, each year trained only on earlier
  data (rank IC 0.27–0.30), joined to the existing 2021–2026 walk-forward forecasts.
- **Selection:** `src/graph/walk_forward.py` on each candidate's monthly long-short return; selected candidates combined with equal weight.
- **Costs:** Σ|Δw| × cost per dollar traded.

**Walk-forward result (the honest number):**

| Run | Selected per year 2021–2026 | Annual return | After 10 bp | After 25 bp | Beta |
|---|---|---:|---:|---:|---:|
| Challenge schedule [WF] | 0, 2, 0, 3, 5, 4 | 1.6% [0.76], 44 active months | −1.8% | −6.9% | −0.03 |
| No validation [WF-0] | 1, 3, 4, 16, 15, 16 | 2.4% [1.09] | 0.2% | −3.1% | 0.00 |

**Why it fails: the drift only exists after 2020.** Strongest candidates (diagnostic, not a strategy; annual %, brackets t):

| Candidate | 2017–2020 | 2021–2026 | 2021–2026 after 25 bp | 2021–2026 factor alpha | By year 2021–2026 |
|---|---:|---:|---:|---:|---|
| Investable, forecast + surprise, 3-month hold | −1.4 [−0.6] | 10.0 [3.1] | 7.8 | 4.6 [2.7] | 27.7, 10.7, 3.3, 0.3, 5.2, 9.2 |
| Large + mega, forecast + surprise, 3-month hold | −2.5 [−1.4] | 7.6 [3.0] | 5.8 | 4.2 [2.7] | 10.8, 8.9, 2.9, 1.5, 9.3, 9.5 |
| Large + mega, surprise drift, 1-month hold | 2.1 [1.0] | 7.5 [3.8] | −0.7 (4.2 at 10 bp) | 7.3 [4.3] | 13.3, 6.3, 6.6, 3.6, 9.5, 3.0 |
| Small, forecast + surprise, 3-month hold | −0.1 [−0.1] | 7.6 [2.7] | 5.6 | 5.6 [2.9] | 21.8, 9.8, 1.1, 1.9, 3.3, 4.9 |
| Micro, forecast + surprise, 3-month hold | −0.8 [−0.3] | 11.3 [4.2] | 9.4 | 12.3 [5.2] | 13.9, 12.4, 13.3, 12.5, 7.7, 4.2 |
| Small, peer news not repriced, 1-month hold | 6.8 [1.6], alpha 12.4 [3.3] | 4.2 [0.8] | −5.9 | 3.1 [0.6] | 22.6, 3.4, 16.5, −2.1, −15.2, −1.4 |

- Factor alpha is on same-period tercile spreads for size, value, momentum, profitability and low volatility, plus the excess S&P return.
  The investable forecast + surprise sleeve loads on value (0.43) and profitability (0.41); its 2017–2020 factor alpha is −4.6% [−3.2].
- Three-month staggered sleeves trade about 0.6–0.8 of capital a month (Σ|Δw|, 200% gross); one-month sleeves trade 2.7–3.4.
- The forecast adds a little to the plain surprise drift (investable 3-month hold: surprise alone 12.2% [2.6] with factor alpha 4.1% [1.3]).
- The graph "peer news not repriced" signal worked in small caps from 2017 to 2023 and broke in 2024–2026.

**Verdict:**
- **The edge is real in 2021–2026 but was not detectable in advance.** Post-earnings drift (especially forecast + surprise,
  held three months) earned 7–11% a year with 4–12% factor alpha in every size group from 2021, but was flat or
  negative in 2017–2020 with genuinely out-of-sample forecasts. A rule-compliant process only starts selecting it in 2024–2026.
- **It cannot be presented as a walk-forward result for the challenge.** Including a drift sleeve would have to be justified by
  prior literature (post-earnings-announcement drift is long documented), with the disclosure that our own pre-2021 data showed none.
- **If kept for live use:** large/mega or small forecast + surprise, 3-month staggered sleeves, is the cheapest to trade
  (about 2% a year of costs at 25 bp); it needs a regime check, because the effect switched on in 2021 and could switch off.

### 9.5 Recheck under the competition document's procedure, by cap [WF]

**What the document specifies:**
- **Schedule:** expanding training and a rolling **two-year** validation window by target month.
  For 2021: train 2015–2018, validate 2019–2020, test 2021; refit annually, forecast monthly.
- **Costs:** no figure is given. The document asks for turnover, a gross/net label, and a reading of the short book
  for hard-to-borrow names.
- **Headline metric:** information ratio against cash + 4%, where active_t = total return − (TB3MS/1200 + 0.04/12).

**Procedure:**
- **Surprise model:** one per test year Y, trained on target months ≤ (Y−3)-12 with labels announced by then.
  That same model scores the validation years and the test year (test ICs 0.35, 0.29, 0.29, 0.29, 0.31, 0.31).
- **Candidates:** the 16 of section 9.4 (8 signals × 1- or 3-month hold).
- **Selection:** in each cap group, the candidate with the best validation information ratio net of the group's assumed
  cost is traded in year Y. If none is positive, the group holds cash.
- **Portfolio limits:** at most 500 names in total; dollar neutral; 200% gross when all sleeves are open.
- **Assumed one-way costs** (ours): micro 40 bp, small 20 bp, large 10 bp, mega 5 bp, large + mega 10 bp, investable 15 bp.

| 01/2021–08/2026, net | Micro | Small | Large | Mega | Large + mega | Investable |
|---|---:|---:|---:|---:|---:|---:|
| Net annual return (gross) | 7.2% (10.4%) | 3.7% (6.3%) | 5.5% (6.6%) | −0.8% (−0.1%) | 1.6% (3.0%) | 0.6% (1.4%) |
| t-statistic of net mean | 2.33 | 0.87 | 2.62 | −0.52 | 0.86 | 0.24 |
| IR vs cash + 4%, net (gross) | **0.48** (0.93) | −0.04 (0.29) | **0.28** (0.49) | −1.14 | −0.58 | −0.59 |
| Sharpe, net | 1.08 | 0.47 | 1.05 | −0.20 | 0.39 | 0.10 |
| Alpha vs S&P, annual [t] | 7.1% [2.5] | 3.9% [1.2] | 4.3% [2.0] | −1.1% [−0.6] | 1.9% [1.1] | 1.1% [0.4] |
| Beta (SE) | 0.01 (0.06) | −0.02 (0.06) | 0.11 (0.04) | 0.02 (0.03) | −0.02 (0.03) | −0.05 (0.05) |
| Max drawdown | −5.4% | −8.5% | −3.2% | −6.3% | −3.5% | −11.2% |
| Net by year 2021–2026 | 12.7, 1.0, 11.3, 8.8, 2.7, 4.4 | 20.8, 10.1, 0.1, −8.4, −1.9, 0 | 8.8, 2.4, 3.4, −1.9, 8.4, 9.8 | 0, 0.2, −3.7, −3.6, 2.4, 0 | 0, 8.0, −2.6, −5.3, 0, 9.0 | 0, 13.1, 2.4, −12.3, 0, 0 |
| Net at 10 / 25 / 50 bp | 9.6 / 8.4 / 6.4 | 5.0 / 3.0 / −0.3 | 5.5 / 3.8 / 1.0 | −1.5 / −3.7 / −7.3 | 1.6 / −0.6 / −4.2 | 0.8 / 0.0 / −1.3 |
| Long leg / short leg excess return | 11.4 / −2.3 | 7.1 / −0.3 | 9.5 / −0.8 | 3.0 / 3.1 | 3.9 / −3.3 | 0.0 / −3.5 |
| Average names long / short | 88 / 73 | 121 / 113 | 106 / 94 | 36 / 29 | 69 / 66 | 97 / 96 |
| Months within 100–500 names | 84% | 82% | 93% | 32% | 53% | 53% |
| Average gross exposure | 119% | 136% | 126% | 91% | 88% | 86% |
| Turnover (names replaced a month) | 27% | 30% | 29% | 33% | 32% | 13% |
| Short book: average market cap; share micro/nano | $0.33B; 97% | $1.7B; 3% | $7.8B; 0% | $115B; 0% | $47B; 0% | $25B; 2% |

- **Choices by year:** micro always picked a 3-month drift variant (surprise, surprise, forecast + surprise, forecast not repriced, forecast, forecast).
  Large changed every year (peer news before report ×2, forecast, surprise not repriced, forecast + surprise, surprise).
  Investable held cash in 2021, 2025 and 2026.
- **Only micro and large clear the hurdle net of costs** (information ratio 0.48 and 0.28), with positive net returns
  in 6 of 6 and 5 of 6 years. Small decays after 2022; mega, large + mega and the combined investable book fail.
- **Caveats:**
  - The candidate menu was designed after the daily event study had read 2021–2026, so the menu is not blind even though selection is compliant.
  - Six cap groups were compared; choosing one of them now is a further selection.
  - Costs are assumptions. The micro short book (97% micro caps, $330M average) is exactly what the document warns may be hard to borrow.
  - Gross exposure averages 119–136%, and some months fall below 100 names (quarterly reporting seasonality empties sleeves).
    This would need fixing, for example by carrying sleeves or filling to 100 names.
  - Delisted names without a return count as 0.

### 9.6 Holding period: decay curves and exit rules chosen on validation [WF]

The 3-month hold in 9.5 was not optimized; the menu was fixed at 1 or 3 months. Here the holding rule is a hyperparameter
chosen on the two-year validation window, with the same schedule, assumed costs and hurdle.

**Candidates:**
- **Signals:** surprise, forecast, forecast + surprise (reporters, within-sector ranks).
- **Exit rules:**
  - fixed hold of 1, 2, 3, 4, 6, 9 or 12 months;
  - **next report:** hold until the end of the month of the firm's next announcement (1–6 months);
  - **valuation target:** hold until the position's cumulative excess return reaches its expected 6-month repricing,
    b × (rank − 0.5), with b estimated by OLS on the fold's training window (at most 6 months).
- **Books:** active positions equal-weighted and scaled to 100% long / 100% short, so gross is 200% whenever a book exists.

**Decay: long-short return in month k after formation (% per month, forecast + surprise):**

| Group, period | k = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 | Sum of 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Large, 2017–2020 | −0.01 | −0.16 | −0.64 | 0.57 | 0.48 | −0.35 | 0.13 | 0.14 | −0.25 | 0.29 | 0.9 |
| Large, 2021–2026 | 0.93 [2.7] | 0.70 | 1.29 [2.9] | 0.94 | 0.57 | 1.07 [2.2] | 0.54 | 0.17 | 0.85 | 1.62 [3.8] | 9.6 |
| Investable, 2021–2026 | 1.02 [3.0] | 0.70 | 0.89 | 0.85 | 0.26 | 0.73 | 1.04 [2.6] | 0.58 | 0.50 | 0.64 | 8.4 |
| Micro, 2021–2026 | 1.78 [4.5] | 1.64 | 1.72 [2.9] | −0.15 | 1.58 | 1.41 | 0.55 | 0.79 | 0.26 | 0.56 | 10.5 |
| Small, 2021–2026 | 0.99 [2.7] | 0.67 | 0.83 | 0.75 | 0.69 | 0.76 [2.8] | 0.17 | 0.28 | 0.48 | 0.15 | 6.4 |

In 2021–2026 the drift is slow: roughly constant for six months and still positive afterwards, with a month-12 bump at the
next same-quarter report. In 2017–2020 it is absent at every horizon. With staggered sleeves a longer hold keeps most of the
monthly return while cutting turnover, so longer holds should win until the marginal month decays.

**Walk-forward result, 01/2021–08/2026, net of assumed costs (hold chosen on validation vs 3-month hold only):**

| Group | Chosen exits by year | Net annual (IR vs cash + 4%) | Fixed 3 months (IR) | Alpha [t], beta | Net by year |
|---|---|---:|---:|---|---|
| Large + mega | target, 4m, 12m, 4m, next report, 4m | **10.8% (0.77)** | 5.1% (0.22) | 10.7% [2.9], 0.00 | 16.7, 14.2, 3.8, −1.4, 15.5, 12.2 |
| Large | target, 4m, 12m, 4m, next report, 6m | **9.8% (0.63)** | 6.8% (0.46) | 10.1% [2.6], −0.03 | 6.9, 15.3, 5.2, −0.2, 12.7, 15.7 |
| Investable | target, 4m, 9m, 9m, 9m, next report | **12.8% (0.62)** | 4.7% (0.10) | 14.5% [2.4], −0.16 | 38.2, 15.2, 8.9, 2.3, −2.3, 10.0 |
| Micro | next report ×2, 9m, target, next report ×2 | 10.2% (0.43) | 10.3% (0.53) | 9.7% [1.6], 0.05 | 28.7, −7.4, −0.9, 19.0, 8.8, 9.6 |
| Small | cash, next report, 9m, 9m, 9m, cash | 1.4% (−0.32) | 0.6% (−0.55) | 2.2% [0.7] | 0, 13.1, 0.5, 6.0, −11.5, 0 |
| Mega | cash, 1m, 1m, target, cash, next report | 1.2% (−0.51) | 1.4% (−0.58) | 0.5% [0.2] | 0, 0.2, −3.7, −7.7, 0, 17.7 |

- **Implementation:** large, large + mega and investable books hold 100–500 names in every month, with 16–18% of names
  replaced per month. These fixed-3 figures differ from 9.5 because books are now scaled to 200% gross and use only the three drift signals.
- **The valuation-target exit underperforms fixed holds:**
  - its expected-repricing slope is estimated on training years, when the drift was small
    (b ≈ 0.00–0.02 in the 2021–2023 folds, 0.03–0.07 later);
  - so targets of about 1–2.5% close winners long before the drift ends.
  - With this data the right exit is time or information based, not a price target.
- **Caveats:**
  - **2021 luck:** the 2021 trades entered with validation IRs of only 0.03–0.09, and 2021 was the strongest drift year.
    Without 2021 the chosen-hold books earn about 9.5% (large + mega), 10.4% (large), 7.3% (investable) and 6.2% (micro) a year.
  - **Menu:** it is larger than in 9.5 and was designed after the daily study had read 2021–2026.
  - **Other:** costs are assumptions; the investable book's beta is −0.16.
- **Not yet tested, graph as an information clock:** keep a position while its graph peers are still reporting
  (their surprises confirm or reverse the news) and exit once most of the peer weight has reported.

### 9.7 The team's TimesFM forecast: what it is, and whether the graph or the drift book complement it [CS, diagnostic]

Zero-shot TimesFM forecasts (`output/preds_timesfm_all.parquet`), 2021–2026 target months, equal-weight top/bottom 100, no costs.

- **What it loads on** (average monthly rank correlation):
  - positive: last-month return 0.46, 12-month momentum 0.48, market equity 0.48
  - negative: idiosyncratic volatility −0.37, book-to-market −0.35
  - In short: long large, calm winners; short small, volatile losers.
- **It still adds something after those factors:** Fama–MacBeth slope 1.88 [3.4] alone, 0.83 [3.1] with 8 controls
  (last-month and 12-month return, max daily return, idiosyncratic volatility, size, turnover, beta, book-to-market).

| Book | Annual | Sharpe | Max drawdown | 2023-01 | Short book micro/nano |
|---|---:|---:|---:|---:|---:|
| TimesFM raw (all prices) | 36.8% [3.3] | 1.24 | −31.9% | −31.9% | 93% |
| TimesFM raw, price ≥ $5 | 11.0% [1.8] | 0.64 | −21.6% | −12.6% | 58% |
| Graph-relative TimesFM (forecast − graph-peer forecast) | 15.7% [1.7] | 0.60 | −25.0% | −24.2% | 90% |
| Graph-relative, price ≥ $5 | 2.3% [0.5] | 0.16 | −16.9% | −11.4% | 54% |

- **About 70% of the raw 2021–2026 return comes from stocks under $5**, whose short side the competition document warns may be untradeable.
- **Ranking against graph peers does not hedge the crashes.** 2021-01 got worse (−19.5% vs −7.5%) and returns roughly halve,
  because the squeezes are cohort-wide rather than within economic neighborhoods.
- **Low correlation with the drift books:** monthly return correlation of TimesFM raw with the drift books is −0.10 (large),
  0.03 (large + mega), 0.07 (micro) and 0.24 (investable). A sleeve-level combination should diversify.

### 9.8 Analyst dispersion and analyst revision lag: pre-registered tests [WF]

Hypotheses and directions were written in `dispersion_tests.py` before running.
- **Setup:** schedule, cost assumptions and evaluation as 9.6. Signal and exit are chosen on validation within each family
  (exits: 1, 3, 6, 9, 12 months or next report). Test window 01/2021–08/2026.
- **Data coverage:** pre-report dispersion 90% of reports, suescore 88%, post-report next-quarter snapshot by the end of
  the following month 99% (dispersion change 88%, revision 99.5%).
- **D4 and R1 timing:** formed at the end of the month after the report (Option A).

| Hypothesis (expected) | Large + mega: net, IR | Small + large + mega | Small | Micro | Verdict |
|---|---|---|---|---|---|
| D1 long low / short high dispersion | cash every year (validation IR −0.28 to −0.87) | cash every year | cash every year | cash every year | **Rejected** everywhere |
| D2 drift, high-dispersion half (> low) | high 5.5%, 0.22 · low 0.1%, −0.73 · all 8.0%, 0.60 | high 4.8% · low 2.9% · all 5.7% | high 1.1% · low 2.5% | high 3.2% (rarely enough names) · low 10.8% | Supported only in large + mega, and the high half still trails all reporters |
| D3 surprise ÷ dispersion (≥ ÷ price) | 2.9%, −0.20 vs 8.0%, 0.60 | 3.2% vs 5.7% | 2.8% vs 1.4% (both IR < 0) | 3.0% vs 4.6% | **Rejected**: price-scaled surprise is better |
| D4 unresolved news (> resolved) | unresolved 2.0%, −0.42 · **resolved 8.7%, 0.70** | 2.9% vs 1.9% (both IR < 0) | 0.1% vs −0.3% | −8.5% vs 6.2% | **Rejected, and reversed** in large + mega and micro |
| R1 surprise + analyst lag (> delayed surprise) | −3.4% vs 4.5% | −0.7% vs 1.8% | −2.1% vs 1.4% | −2.4% vs 6.4% | **Rejected** everywhere |
| R1 control: next-quarter revision momentum | 5.7%, 0.19 (DD −14%) | 2.5% | 2.5% | −0.4% | Weak |

**Reversed D4 in large + mega**, after reading the result: when analysts' disagreement about the next quarter
**falls** after the report, the surprise drifts. The book earns 8.7% net, IR 0.70, Sharpe 1.30, alpha t 2.98, beta 0.01,
max drawdown −4.2%, with net returns by year of 0, 12, 8, 0, 13, 17%. When disagreement rises, it does not.
A plausible reading is that converging analysts mark credible, persistent news, while rising disagreement marks noisy news.
Because the direction was reversed after seeing the test, this is a hypothesis for fresh validation, not an established edge.

**Other observations:**
- Every family held cash in 2021: the 2019–2020 validation windows showed no drift.
- The large + mega all-reporter drift book earns 8.0% here vs 10.8% in 9.6. The only difference is the exit menu
  (no 2- or 4-month holds, no target exit), which shows how sensitive the headline number is to the menu.

### 9.9 Which expert works when analysts agree or disagree? Evidence for a dispersion-gated mixture [CS, diagnostic]

Monthly rank IC by tercile of analyst dispersion (within month and cap), 2017–2020 / 2021–2026, brackets Newey–West t.

| Expert → target | Large + mega: agree / disagree | Small: agree / disagree | Micro: agree / disagree |
|---|---|---|---|
| ML surprise forecast → next analyst surprise | 0.40 / 0.27 · 0.44 / 0.36 | 0.39 / 0.25 · 0.40 / 0.27 | 0.30 / 0.27 · 0.33 / 0.23 |
| Graph peer surprises → next analyst surprise | 0.08 / 0.07 · 0.06 / 0.10 | 0.10 / 0.06 · 0.11 / 0.09 | 0.07 / 0.05 · 0.09 / 0.04 |
| Realized surprise → next 3 months' return (drift) | 0.01 / 0.01 · 0.03 [1.8] / **0.07 [4.1]** | 0.03 / 0.01 · **0.06 [3.2]** / 0.03 | 0.03 / 0.00 · 0.02 / 0.01 |
| TimesFM (price ≥ $5) → next-month return | 0.02 / 0.00 · 0.01 / −0.01 | 0.02 / 0.02 · 0.02 / 0.03 | −0.01 / 0.01 · 0.03 / 0.03 |

**Drift by post-report convergence** (surprise → returns T+2..T+4):
- Converged vs diverged shows no difference in large + mega (0.033 vs 0.035 in 2021–2026). The D4 book gap of 9.8 is not confirmed at the IC level.
- Large + mega is strongest when analysts disagreed before the report and converged after it: 0.062 [3.7] in 2021–2026, 0.020 in 2017–2020.
- Small caps show a different pattern.

**Verdict:**
- **The ML surprise model is most accurate when analysts agree,** in both periods and every cap. Consensus errors are
  systematic when analysts herd.
- **The return drift in large + mega is strongest when they disagree,** 2021–2026 only; small caps are the reverse.
- **Graph peers and TimesFM show no stable dispersion pattern.**
- **A single gate** ("agree → graph, disagree → ML") **is not supported.** The evidence points to the reverse mapping for
  the ML forecast, a cap-specific mapping for drift, and drift gating that can only be seen in the test years.

### 9.10 Overnight pre-registered edge tests: "the Street confirms the surprise" [WF]

Hypotheses were written before running (archived `overnight_hypotheses.md`); protocol as 9.8. Net annual return and IR vs cash + 4%
for 01/2021–08/2026; parentheses give the share of months within 100–500 names.

**Batch 1**

| Family (expected) | Large + mega | Small + large + mega | Small | Micro |
|---|---|---|---|---|
| Reference: surprise drift at T | 8.0%, 0.60 | 5.7%, 0.22 | 1.4%, −0.32 | 4.6%, 0.04 |
| E1 dispersion tilt (> reference) | 5.5%, 0.22 | **7.6%, 0.39** | 2.0%, −0.29 | 5.7%, 0.14 |
| E2a graph confirmation, added to the score (> reference) | 4.5%, 0.08 | 3.9%, −0.01 | 0.5%, −0.42 | 6.7%, 0.23 |
| E2b graph-confirmed news only (> reference) | 1.8%, −0.39 | 1.1%, −0.36 | 3.5%, −0.07 | −3.6%, −0.47 |
| E3 same-quarter seasonality (positive) | ≈ 0, cash most years | cash every year | −3.7%, −1.11 | −3.7%, −1.69 |
| E4 graph information clock exit (≥ reference) | 4.9%, 0.17 | 4.4%, 0.05 | 1.2%, −0.45 | **9.1%, 0.38** |
| E5 TimesFM, price ≥ $5, raw or residual | −3.0%, −0.54 | 4.9%, 0.05 | 7.9%, 0.27 | 2.2%, −0.23 |
| Reference: delayed surprise at T+1 | 4.5%, 0.11 | 1.8%, −0.35 | 1.4%, −0.54 | 6.4%, 0.13 |
| E6 high pre-report dispersion that converged (> delayed) | 10.1%, 0.72 (44%) | 1.1%, −0.43 | 2.5%, −0.19 (26%) | never enough names |
| **E7 surprise + post-report analyst revision breadth (> delayed)** | **8.9%, 0.78** (82%) | **12.7%, 1.00** (100%) | 2.4%, −0.26 | 11.5%, 0.42 (56%) |

- **Breadth definition:** (up revisions − down revisions) / number of estimates, for the next quarter, in the first I/B/E/S
  snapshot after the report; formed at the end of the month after the report.
- **E7, small + large + mega:**
  - Sharpe 1.47, alpha t 3.70, beta −0.09, max drawdown −5.1%
  - 63% of months beat the hurdle; best month +7.9%, worst −4.2%
  - 12% of names replaced a month
  - 12.0% net at 25 bp (IR 0.93), 10.3% at 50 bp (IR 0.73)
  - net by year: +30, +14, +2, +10, +8, +7; 2021 was traded because validation was positive
- **E7, large + mega:** Sharpe 1.42, alpha t 3.22, beta 0.02; net by year 0, +9, +9, +10, +15, +7.
- **Not supported:** graph confirmation, the graph clock (except micro), seasonality and TimesFM in large + mega.

**Batch 2** (written after seeing only large + mega batch 1 results)

| Check | Large + mega | Small + large + mega | Small | Micro | Verdict |
|---|---|---|---|---|---|
| F1a revision breadth alone (< E7) | 6.1%, 0.36 | 0.9%, −0.50 | 4.6%, 0.08 | 2.2%, −0.15 | Pass except small: E7 is more than breadth |
| F1b E7 + dispersion convergence | 6.5%, 0.44 | 5.1%, 0.16 | 2.7%, −0.16 | 2.0%, −0.20 | Worse than E7 |
| F1c ML forecast + E7 | **11.9%, 1.21** | 8.3%, 0.45 | −0.8%, −0.70 | 6.9%, 0.18 | Better in large + mega only |
| F2 fixed rule, 6-month hold, 2016–2020 net (E7 vs delayed surprise) | −1.8% [−0.5] vs −4.5% [−2.0] | +1.1% [0.3] vs +1.2% | −0.7% vs −0.7% | −6.8% vs −1.2% | **Fail**: E7 did not work before 2021 |
| F3 factor regression, large + mega E7 | alpha 7.2%, t 2.64, R² 0.19; profitability 0.24, momentum 0.20, value 0.14 | | | | Pass |
| F4 costs, large + mega E7 | 25 bp 7.9% (IR 0.62); 50 bp 6.3% (0.36); without 2025: 7.5% (0.55) | | | | Pass |

F1c large + mega: Sharpe 1.82, alpha t 4.22, beta 0.00, max drawdown −3.4%, net by year 0, +13, +12, +8, +19, +16.

**Verdict:**
- **The strongest compliant result so far:** trade the surprise only after the Street confirms it, i.e. analysts revise
  next-quarter estimates in the same direction. It works in small + large + mega and large + mega, survives costs and a factor
  regression, and is not driven by a single year.
- **It is still a post-2021 phenomenon,** like the plain drift. In 2016–2020 the same rule earned nothing, so the regime
  risk from 9.4 remains and cannot be validated on earlier data.
- **Adding the ML forecast** helps in large + mega (IR 1.21) but not in the broader universe.

**Batch 3: other forms of Street confirmation** (pre-registered; net annual, IR)

| Family (expected) | Large + mega | Small + large + mega | Small | Micro |
|---|---|---|---|---|
| Reference E7: surprise + next-quarter breadth | 8.9%, 0.78 | 12.7%, 1.00 | 2.4%, −0.26 | 11.5%, 0.42 |
| G1 surprise + rating upgrade (between delayed and E7) | 7.6%, 0.66 | 7.1%, 0.41 | 0.7%, −0.71 | −4.3%, −1.05 |
| G2 surprise + revision size (≈ E7) | **10.0%, 0.84** | 6.0%, 0.26 | 2.3%, −0.36 | 11.8%, 0.49 |
| G3 surprise + breadth of the next two quarters (≥ E7) | 8.2%, 0.70 | **14.1%, 1.13** | 3.3%, −0.12 | 0.3%, −0.25 |
| G4 E7, low analyst coverage (> high) | 6.0%, 0.36 | −0.2%, −0.63 | −4.9%, −0.97 | −2.7%, −1.13 |
| G4 E7, high analyst coverage | 6.7%, 0.45 | 5.8%, 0.31 | 3.4%, −0.09 | 3.1%, −0.05 |

- **The slow-diffusion (low coverage) hypothesis is rejected:** confirmation works in well-covered stocks.
- **Revision breadth over one or two quarters is the most robust confirmation measure.**

**Which leg earns E7?** Fixed 6-month hold, 2021–2026, gross, against the equal-weight return of stocks that were tradable
and in the cap group at the formation month-end.

| | Long leg vs universe | Short leg vs universe | Spread |
|---|---|---|---|
| Large + mega | +9.9% [6.1] | +4.5% [2.7] | 14.4% [6.1] |
| Small + large + mega | +10.7% [4.9] | +0.6% [0.2] | 11.3% [3.2] |
| Small | +8.0% [3.7] | +0.3% [0.1] | 8.3% [2.6] |
| Micro | +5.9% [2.1] | +0.3% [0.1] | 6.2% [0.9] |

- **E7 is mainly a long edge:** stocks with a beat that analysts then confirm keep outperforming.
- **A benchmark mistake:** a first version measured the legs against stocks in the cap group at the end of the return
  month. Winners migrate into larger size groups during the month, so that benchmark was biased up by about 11% a year
  (33% in micro) and made the short leg look like the whole edge. Strategy returns are unaffected, because
  membership and size group are fixed at formation.

**Batch 4: graph × Street confirmation** (pre-registered; micro was not completed because the machine ran out of memory)

| Family (expected) | Large + mega | Small + large + mega | Small |
|---|---|---|---|
| Reference E7 (T+1) | 8.9%, 0.78 | 12.7%, 1.00 | 2.4%, −0.26 |
| H1 E7 + graph-weighted peer revision breadth (≥ E7) | 5.7%, 0.28 | 6.0%, 0.31 | −0.3%, −0.79 |
| H2 peers' confirmed news → firms that have not reported (positive) | −2.5%, −1.75 | 1.9%, −0.48 | 4.0%, 0.00 |
| H3 E7 formed in the report month when a post-report snapshot exists (≥ E7) | 9.4%, 0.65 | 8.5%, 0.44 | 7.0%, 0.34 |

- **Graph additions dilute the Street-confirmation signal,** and peers' confirmed news does not spill over to firms that have
  not reported.
- **Earlier entry helps only in small caps.**

**Batch 5: E7 implementation robustness** (pre-registered; no selection across variants)

| Variant | Large + mega | Small + large + mega |
|---|---|---|
| Baseline: 20% tails, equal weight, within-sector ranks | 8.9%, 0.78 | 12.7%, 1.00 |
| 10% tails | 10.0%, 0.87 | 12.4%, 0.86 |
| 30% tails | 8.1%, 0.70 | 11.2%, 0.91 |
| No sector neutrality | 11.7%, 1.00 | 8.5%, 0.55 |
| Excluding financials and real estate | 9.7%, 0.93 | 7.3%, 0.43 |
| **Value-weighted sides** | **3.0%, −0.07** (max drawdown −18%) | 14.5%, 0.47 (volatile) |

- **Stable across tails, sector treatment and sector exclusions.**
- **Fails the pre-registered pass criterion under value weighting in large + mega.** The effect sits in the smaller names of
  the group rather than the mega caps; batch 6 tests this directly.

**Batch 6: where E7 lives** (pre-registered)

| Family (expected) | Net annual, IR | Sharpe, alpha t, max drawdown | Months within 100–500 names | Net by year 2021–2026 |
|---|---|---|---|---|
| J1 large group only (large > mega) | **7.9%, 0.62** | 1.27, 3.12, −6.1% | 82% | 0, 9, 8, 12, 12, 4 |
| J1 mega group only | 2.4%, −0.32 | 0.48, 1.03, −3.2% | 47% | 0, 10, −5, 0, 0, 9 |
| J2 large + mega without the smallest 30% (IR > 0.4) | **7.8%, 0.70** | 1.46, 3.38, −4.1% | 82% | 0, 11, 3, 10, 10, 10 |
| J3 two consecutive confirmed beats/misses, large + mega (≥ E7) | 7.5%, 0.41 | 0.89, 2.34, −9.3% | 35% | 0, 29, 6, 10, −3, 0 |
| J3 same, small + large + mega (≥ E7) | 8.6%, 0.55 | 1.02, 2.72, −6.5% | 82% | 0, 18, 3, 15, 8, 6 |

- **E7 is a large-cap effect,** not a mega-cap one, and it survives removing the smallest 30% of names. Value weighting fails
  because mega caps dominate the weights.
- **Streaks add nothing.**
- **Economic reading:** E7 is a monthly, point-in-time version of earnings momentum confirmed by analyst revision momentum
  (in the spirit of Chan, Jegadeesh and Lakonishok, 1996). It is a known family of anomalies, here validated under the
  competition schedule and active only since 2021.

**Batch 7: non-earnings 8-K events confirmed by analysts** (pre-registered). Firms with an 8-K in month T and no earnings
report in T or T+1, ranked at T+1 by target-quarter revision breadth.
- **Any non-earnings event:** large + mega −0.3% (IR −1.33), small + large + mega −2.0%, small +0.3%, micro −4.8%. Mostly cash,
  and the books rarely reach 100 names.
- **By item** (officer change, material agreement, restructuring/impairment, other events): too sparse or negative.
- **Control, no 8-K and no report:** −1.8% to +1.3%.
- **Verdict:** Street confirmation earns only after earnings reports. Revision momentum outside report months, with or
  without a corporate event, does not.

**Batches 8 and 9: what else to add to E7** (pre-registered; net annual, IR)

| Family | Large + mega | Small + large + mega | Small | Micro |
|---|---|---|---|---|
| Reference E7 | 8.9%, 0.78 | 12.7%, 1.00 | 2.4%, −0.26 | 11.5%, 0.42 |
| L1 E7 + ML forecast of the next quarter made after the report | 8.7%, 0.71 | 9.8%, 0.61 | 2.6%, −0.20 | 11.4%, 0.47 |
| L2 that forecast alone | 8.0%, 0.43 | 4.5%, 0.05 | −0.5%, −0.55 | 3.9%, −0.01 |
| M1 E7 + TimesFM forecast | 3.9%, −0.01 | 6.8%, 0.31 | **6.7%, 0.33** | 1.7%, −0.13 |
| M2 E7 − price move since the report ("confirmed, not yet repriced") | 6.6%, 0.35 | −2.6%, −1.19 | 1.9%, −0.38 | 9.0%, 0.30 |
| M3 E7 + price move since the report ("confirmed and moving") | 10.3%, 0.78 | 3.1%, −0.09 | 1.4%, −0.32 | −1.8%, −0.44 |

- **E7 on its own is the cleanest version.**
- **The post-report forecast does not help;** the pre-report forecast (F1c) helps only in large + mega.
- **Buying confirmed news that has not yet repriced is worse than E7** in three of four groups: this is confirmation
  momentum, not a catch-up.
- **TimesFM helps only in small caps.**

**Batch 10: placebo test** (pre-registered). Revision breadth is shuffled at random within month and sector (20 seeds) and the
whole walk-forward selection is rerun.

| Cap group | Real E7 IR | Placebo IRs: mean, 95th percentile, max | Placebos ≥ real |
|---|---|---|---|
| Small + large + mega | **1.00** | −0.33, 0.18, 0.28 | 0 of 20 |
| Large + mega | **0.78** | −0.29, 0.69, 0.74 | 0 of 20 |

- **Small + large + mega: clear pass.** Breadth carries real information beyond the surprise and the selection procedure.
- **Large + mega: pass, but narrow.** Two random placebos reached IR 0.68–0.74, so an IR near 0.7 in that group can arise from
  the procedure and the surprise alone about 10% of the time.

### 9.11 Smallest Qwen mixture-of-experts model: 8-K counterparty extraction and expert-count ablation [CS, exploratory]

- **Model:** Qwen1.5-MoE-A2.7B-Chat, the smallest Qwen MoE on Hugging Face (14.3B parameters, 60 routed experts, top-4 routing,
  2.7B active). Run as a 4-bit IQ4_XS GGUF (7.9 GB, SHA-256 verified) with llama.cpp b10970 (CUDA 12.8) on an RTX 3070 Ti
  Laptop (8 GB) with automatic GPU/CPU fit.
- **Files:** kept in the job's temporary folder, not the repository.

**Pilot: supply-chain edges from 8-K Item 1.01.** 300 random filings from 2019–2020 (7,872 in those years).
- **Prompt:** strict-JSON output of agreement type, counterparties (excluding the filer and financing agents) and the filer's role.
- **Speed and format:** 100% valid JSON (grammar-constrained); 0.9 s per filing (≈ 1,190 prompt and 95 generated tokens a second).
- **Agreement types:** credit financing 44%, securities offerings 25%, M&A 21%, licensing/collaboration 5%,
  commercial supply/customer 1%, other 4%. Free-text types sometimes escaped the enum.
- **Instruction following is weak:** banks and underwriters were still returned in financings (388 mentions, 40 matched to
  panel banks).
- **Edge yield:** 23 commercial or licensing counterparties, 4 matched to panel companies (BioMarin, GE, Ultragenyx, and one false
  self-match). Most counterparties are private, foreign or academic.
- **Verdict:** Item 1.01 is a poor supply-chain source. It would yield roughly one matched commercial edge per 75 filings,
  a few hundred edges over 2015–2026. Compustat customer segments remain the better route (section 11).

**MoE expert-count ablation.**
- **Setup:** the same 60 filings, changing the number of routed experts used per token (the model was trained with 4).
- **Patch needed:** this GGUF lacks explicit expert feed-forward sizes, so llama.cpp derives them from the expert count. A
  metadata-only copy with the sizes written out (`patch_gguf_expert_ff.py`) made the override work.

| Experts per token | Valid JSON | Same agreement type as 4 experts | Counterparty overlap with 4 experts (Jaccard) | Prompt tokens/s | Generated tokens/s |
|---|---|---|---|---|---|
| 1 | 80% | 63% | 0.41 | 1,467 | 145 |
| 2 | 100% | 92% | 0.77 | 1,295 | 118 |
| 4 (trained) | 100% | — | — | 1,342 | 95 |
| 8 | 100% | 93% | 0.87 | 1,181 | 68 |
| 16 | 100% | 85% | 0.83 | 945 | 40 |

- **Speed:** generation speed falls almost in proportion to the active experts, because per-token expert weights are the
  memory-bandwidth bottleneck. Prompt processing is batched and barely changes.
- **Fewer experts:** top-1 breaks format and content; top-2 keeps about 90% agreement for 24% faster generation.
- **More experts than trained:** does not converge to anything better and drifts away from the trained behaviour.
  Routing is a trained hyperparameter, not a free dial.
- **Limitation:** agreement is measured against the top-4 model, not against hand-labelled truth.
- **Lesson for the project's MoE step:** a small, trained top-k gate over specialised experts is how these models work.
  Averaging many experts is not a substitute; this matches 9.9, where one gate rule across caps was not supported.

**Multiple testing across 9.5–9.10.**
- **Books tested:** 134 distinct walk-forward books that took positions.
- **Signal vs chance:** 25% have t ≥ 2, against about 5% expected under independence.
- **Bonferroni:** the 5% two-sided threshold for 134 tests is t ≥ 3.56. Only three books reach it, all Street confirmation in
  large + mega: two-quarter breadth (G3) t 4.16, forecast + surprise + breadth (F1c) t 3.97, E7 t 3.56.
- **Reading:** the variants are strongly correlated, so they count as one effect confirmed several ways, not three
  independent discoveries.
- **What the bound does not cover:** the candidate menu was built iteratively on test-period diagnostics, so the real hurdle is higher.
- **The regime caveat stands:** the effect is absent in 2016–2020.

### 9.12 Valuation × Street confirmation, and TimesFM distribution signals: pre-registered batches 11–12 [WF]

Both batches were pre-registered before running. They use the same protocol as 9.10: the exit is chosen on the two-year
validation window by IR vs cash + 4% net of assumed costs, cash if nothing is positive, sector-neutral top/bottom 20%.
Cells show net annual return [HAC t], then IR vs cash + 4%.

**Batch 11: does "where" (valuation) sharpen "when" (the Street confirms)?**
- **Valuation signals:** section 7.3 definitions, recomputed at the formation month (T+1) from that month's cross-section only:
  - analyst implied cost of capital (ICC)
  - reverse-DCF feasibility (share of same size × sector firms whose realized 3-year sales growth beats the growth implied by
    EV/FCF)
  - cheapness vs warranted multiples
  - `value` = mean of their within-sector ranks
- **Coverage of delayed reporters:** ICC 75%, feasibility 52%, warranted multiples 90%, composite 77%.
- **Families:** N1–N4 add one rank to E7. N5 is a gate: longs only from confirmed beats that are cheap, shorts only from
  confirmed misses that are expensive.

| Family | Large + mega | Investable | Small | Micro |
|---|---|---|---|---|
| N0 E7 (reference) | 8.9% [3.56], 0.78 | 12.7% [3.01], 1.00 | 2.4% [0.80], −0.26 | 11.5% [1.19], 0.42 |
| N1 + analyst ICC | 10.2% [3.48], 0.80 | 9.3% [2.68], 0.58 | 0.2% [0.07], −0.51 | 0.7% [0.12], −0.17 |
| N2 + feasibility | 3.9% [2.38], −0.03 | 7.1% [2.70], 0.46 | −0.5% [−0.21], −0.86 | no book (too few names) |
| N3 + warranted multiples | 11.5% [2.94], 0.85 | 8.5% [2.26], 0.55 | 0.6% [0.18], −0.60 | 5.4% [1.08], 0.12 |
| N4 + value composite | 9.4% [3.07], 0.68 | 8.7% [2.70], 0.58 | 2.4% [0.90], −0.25 | 7.3% [1.05], 0.23 |
| N5 value gate | 10.0% [3.45], 0.88 | 7.1% [2.35], 0.43 | 5.1% [0.73], 0.09 (max drawdown −29.8%) | no book |

- **The reference reproduces 9.10 exactly**, so the pipelines agree.
- **The pre-registered expectation (≥ E7) is not met.**
  - Large + mega: N1, N3 and N5 are marginally above E7 on IR (0.80–0.88 vs 0.78), with equal or lower t. Over 5.6 years
    the IR standard error is about 0.45, so these gaps are noise.
  - Investable: every tilt had a negative 2019–2020 validation and sat in cash in 2021, the year E7 earned 30.5%. Over
    2022–2026 the tilts average 8.0–10.6% a year against E7's 8.2%. That comparison is post-hoc and does not rescue the test.
  - Small and micro: worse than E7.
- **Coverage confound:** a name without a valuation signal drops out of the additive books. Large + mega names per side fall
  from 115 to 81–101.
- **One thing to note, post-hoc:** in large + mega the gate earns about E7's return with lower turnover (Σ|Δw| 0.33 vs 0.54)
  and a higher alpha t (3.86 vs 3.22).
- **Verdict:** the confirmation carries the return, and cheapness does not add in a way that survives the protocol. This
  matches section 7.3: valuation describes expectations but does not predict returns here.

**Batch 12: TimesFM quantiles as risk and return signals.** All stocks in the cap group with a zero-shot forecast; exits h1/h3/h6.
Main universe: price ≥ $5 and dollar volume above the 20th percentile.

| Signal | Large + mega | Investable | Small | Micro |
|---|---|---|---|---|
| P0 point forecast | −3.0% [−0.52], −0.54 | 4.9% [0.72], 0.05 | 6.4% [1.20], 0.18 | −0.2% [−0.05], −0.40 |
| P1 forecast / (q90 − q10) | −1.9% [−0.52], −0.71 | 3.6% [0.67], −0.03 | 5.3% [1.00], 0.10 | 3.6% [1.04], −0.03 |
| P2 q10 (least downside) | −1.8% [−0.52], −0.42 | 0.3% [0.05], −0.20 | −2.6% [−0.38], −0.39 | −4.6% [−1.26], −0.80 |
| P3 −skew | −2.3% [−1.39], −1.02 | 0.1% [0.03], −0.77 | −2.8% [−0.70], −0.90 | 1.2% [0.23], −0.24 |
| P4 rank forecast + rank q10 | cash all years | −1.8% [−0.24], −0.30 | −1.4% [−0.24], −0.36 | −3.2% [−1.01], −0.68 |

- **No pre-registered expectation holds in the tradable universe.**
  - P1 does not beat P0.
  - P2 does not reduce drawdowns or crash months. In January 2023 it lost 22.3% in investable vs 13.0% for P0, and 14.8% in large + mega.
  - P3 is not positive. P4 does not beat P0.
  - P3's smaller drawdowns in large + mega and investable (−11% and −6%) mostly reflect four cash years out of six.
- **Without the price screen (the team's setting):** micro P3 earns 17.4% [2.10], IR 0.71, positive in every year. P4 earns
  13.7% [1.62] and P2 8.8% [1.09]. Worst months: P3 −17.8% in 2021-01; P2/P4 −19% to −23% in 2023-01.
- **Legs of that micro P3 book** (post-hoc, fixed 1-month hold, gross, formation-month benchmark):

  | Slice | Spread | Long leg | Short leg |
  |---|---|---|---|
  | All names | 15.1% [2.13] | 9.2% [2.19] | 5.9% [1.47] |
  | Price ≥ $5 and liquid | 9.4% [1.40] | | |
  | Below $5 or illiquid | 19.6% [2.25] | | |

  - 46% of short names trade below $5, against 25% of longs.
  - Chosen within the tradable universe, the same idea earns 1.2% net.
  - This is the lottery-stock (skewness/MAX) effect in penny micro caps: short names are hard to borrow and squeeze-prone,
    and t 2.1 is far from the multiple-testing bound.
- **Verdict:** TimesFM's quantiles give no tradable edge in any cap group. The point forecast stays what 9.7 found, a
  diversifying momentum/low-volatility tilt. The quantiles belong in the risk and portfolio steps (sizing, a veto on shorting
  lottery-like names), not in alpha.

**Multiple testing, updated:**
- **Books:** batches 11–12 add 56 books with positions (18 + 38), for 190 in total. The Bonferroni threshold rises to t ≥ 3.65.
- **Still above it:** two-quarter breadth (G3, t 4.16) and forecast + surprise + breadth (F1c, t 3.97).
- **Just below:** E7 (t 3.56). No batch 11 or 12 book reaches the bound; the best is N1 at 3.48.

### 9.13 Which model values companies best? Pricing accuracy, unique firms, a learned gate, and the value gap as an edge [WF + post-hoc checks]

Pre-registered as batch 13 before running. The scripts were rebuilt after a reboot wiped the temporary folder and are archived
in `~/fiam26_batch13_20260916.tar.gz`. The result files were lost; the numbers below are the values printed during the run.

**Setup.**
- **Target:** log(company market equity / total assets), predicted from non-price information only:
  - accounting levels scaled by assets (market value cancels out of every input)
  - analyst next-4-quarter EPS × shares
  - JKP residual-income value
  - about 60 profitability, growth, accrual, intangibility, leverage and earnings-quality ratios
- **Excluded inputs:** returns, volatility, beta, liquidity, share counts, and any score containing market value.
- **Pricing error** = predicted − actual. The **gap** is the same quantity, read as "cheap" when positive.
- **Training schedules:**
  - Cross-sectional models are fitted on each month with a 5-fold cross-fit by firm.
  - History models train on months ≤ Dec Y−3 and predict Y−2..Y.
- **Test window:** 2021-01 to 2026-08, 254,470 firm-months where every model has a prediction.

**Pricing accuracy.** MdAE is the median absolute log error; ±25% is the share of firm-months priced within 25%.

| Model | MdAE | Within ±25% | Monthly rank correlation |
|---|---:|---:|---:|
| M0 industry comparables (median multiples of book, forward earnings, sales, EBITDA, assets) | 0.543 | 24% | 0.59 |
| M1 JKP residual-income value (own coverage, 156k rows) | 0.622 | 19% | 0.60 |
| M2 ridge warranted value + sector/industry dummies | 0.450 | 27% | 0.78 |
| M3 agnostic dollar levels (Bartram–Grinblatt OLS) | 5.46 | 5% | 0.26 |
| **M4 gradient-boosted trees, no sector codes** | **0.358** | **34%** | **0.85** |
| **M5 M4 + sector and industry codes** | **0.355** | **35%** | **0.85** |
| M6 trees trained on history | 0.384 | 31% | 0.84 |
| M7 theory gate: separate experts with the anchors theory assigns each sector group | 0.520 | 24% | 0.67 |
| M8 learned mixture of experts (4 linear experts, gate on firm traits, no sector codes, EM) | 0.463 | 26% | 0.78 |
| M9 mean of M2, M4–M8 | 0.379 | 32% | 0.84 |

- **Best model on validation:** M6 in 2021, 2025 and 2026; M5 in 2022–2024.
- **Sector codes add almost nothing** once trees see the fundamentals (0.355 vs 0.358).
- **Hand-coded sector experts (M7) price worse than one linear model with dummies (M2).**
- **By cap (M5):** mega 0.29, large 0.31, small 0.32, micro 0.38, nano 0.53.
- **By sector (M5):** utilities 0.20, financials 0.21, materials 0.31, industrials 0.33, energy 0.34, staples 0.35,
  discretionary 0.38, tech 0.41, health 0.45, communication 0.48, real estate 0.48. Comparables are worst in exactly the
  intangible-heavy sectors: health 0.72, communication 0.71, tech 0.63.
- **Unique firms:** the error rises steadily with isolation, measured as the distance to the 10th-nearest firm in trait space.
  - Trees: 0.14 in the least isolated decile, 0.36 in the middle, 0.56 in the most isolated.
  - Comparables: 0.19 → 0.94.
  - Learned MoE: 0.29 → 0.70.

**Nvidia and other cases** (model value ÷ price; M5 trees / M6 history trees / M8 MoE / M1 residual income):

| Firm-month | Isolation decile | M5 | M6 | M8 | M1 |
|---|---:|---:|---:|---:|---:|
| NVDA 2021-01 | 7 | 0.50 | 0.46 | 0.35 | 0.32 |
| NVDA 2023-01 | 6 | 0.28 | 0.18 | 0.25 | 0.18 |
| NVDA 2024-01 | 9 | 0.22 | 0.15 | 0.21 | 0.33 |
| NVDA 2025-01 | 9 | 0.35 | 0.24 | 0.15 | 0.97 |
| NVDA 2026-07 | 10 | 0.61 | 0.28 | 0.67 | 1.43 |

- **Nvidia:** every fundamentals model called it 3–7× overvalued through 2023–2025 while it kept rising. For firms with no
  close peers, the gap measures growth expectations the inputs cannot see, not mispricing.
- **Tesla:** 0.1–0.5 throughout.
- **Apple:** trees 0.4–0.9; comparables 0.12–0.22.
- **JPMorgan and Exxon:** low isolation, trees 0.6–1.1.

**What the learned gate found without sector labels.** The mixture of experts' log-likelihood rose from −0.91 to −0.72 in
the 2021 fold, and the experts specialized. Its four experts:

| Expert | Share of rows | Profile | Where the gate puts it |
|---|---:|---|---|
| 1 | 26% | Levered, low-margin balance sheets (assets/book 3.9, gross profit/assets 0.07) | 66% of financials' weight, 61% of utilities'; JPMorgan 0.62–0.83 |
| 3 | 33% | Loss-making, R&D-heavy, cash-rich small firms without analysts (R&D/sales 0.52, cash/assets 0.59) | 71% of health care's weight, i.e. biotech |
| 0 | 24% | Profitable operating companies (ROA 8%, gross profit/assets 0.34) | Nvidia and Apple 0.9–0.99 |
| 2 | 17% | Loss-making levered firms (debt/assets 0.46) | — |

Exxon moved from expert 1 (2021–2022) to expert 0 (2023). The gate rediscovered finance's valuation regimes from firm traits
alone, but its linear experts price worse than trees.

**Returns** (pre-registered protocol; cells = net annual [HAC t], IR vs cash + 4%). S books use the gap alone on tradable
stocks (exits h1/h3/h6/h12). T books tilt E7 on delayed reporters (exits h3/h6/h12, within-sector percentile before the name
cap). "Best pricing model" is chosen per fold by validation MdAE.

| Family | Large + mega | Investable | Small | Micro |
|---|---|---|---|---|
| S M0 comparables | 8.6 [1.23], 0.25 | 8.2 [1.26], 0.24 | 5.5 [0.90], 0.12 | 29.6 [4.09], 1.40 |
| S M1 residual income | 0.4 [0.23], −0.52 | 4.6 [1.65], 0.07 | 5.0 [1.27], 0.11 | −3.8 [−1.73], −1.51 |
| S M2 ridge | −2.5 [−1.02], −1.02 | 1.8 [0.34], −0.17 | 2.6 [0.51], −0.12 | 25.4 [4.34], 1.54 |
| S M3 agnostic levels | 12.6 [2.50], 0.69 | 10.0 [1.60], 0.44 | 5.5 [0.89], 0.10 | 27.1 [3.12], 1.22 |
| S M4 trees | 1.5 [0.31], −0.20 | 7.6 [1.36], 0.24 | 7.6 [1.37], 0.27 | 30.4 [4.41], 1.76 |
| S M5 trees + sector | 1.1 [0.24], −0.24 | 6.3 [1.04], 0.15 | 7.8 [1.39], 0.27 | 30.9 [4.81], 1.80 |
| S M6 history trees | 4.3 [0.85], 0.02 | 6.1 [1.36], 0.18 | 11.5 [1.61], 0.42 | 27.0 [3.49], 1.40 |
| S M7 theory gate | −2.1 [−1.05], −1.08 | 5.0 [0.85], 0.07 | 3.3 [0.54], −0.06 | 26.9 [3.79], 1.37 |
| S M8 learned MoE | 3.9 [0.75], −0.01 | 3.7 [0.71], −0.02 | 3.3 [0.82], −0.07 | 24.4 [3.11], 1.26 |
| S M9 ensemble | 3.5 [0.64], −0.03 | 4.5 [0.85], 0.04 | 2.2 [0.60], −0.20 | 33.2 [4.48], 1.60 |
| **S best pricing model** | 1.1 [0.24], −0.24 | 2.9 [0.52], −0.08 | 9.1 [1.26], 0.28 | **31.1 [4.20], 1.74** |
| S model + exit chosen by validation IR | 12.6 [2.50], 0.69 | 10.7 [1.62], 0.44 | 8.5 [1.24], 0.26 | 28.5 [3.82], 1.49 |
| T0 E7 reference | 7.8 [3.84], 0.69 | 3.1 [1.31], −0.14 | 1.4 [0.57], −0.46 | 0.1 [0.02], −0.48 |
| T E7 + best-model gap, all sectors | 6.4 [1.77], 0.26 | 5.3 [1.61], 0.15 | 1.4 [0.41], −0.30 | 16.2 [3.56], 1.26 |
| T E7 + gap in book-anchored sectors only | 9.2 [4.03], 0.90 | 3.1 [1.15], −0.15 | 1.5 [0.54], −0.42 | 1.9 [0.61], −0.24 |
| T E7 + gap in sectors with positive validation IC | 6.0 [2.67], 0.33 | 6.0 [2.51], 0.33 | 1.2 [0.41], −0.40 | 8.7 [2.92], 0.60 |

- **Why T0 differs from 9.10:** the investable E7 reference (3.1%) is lower than 9.10's 12.7% because this batch offers only
  h3/h6/h12. Its validation was negative, so it held cash in 2021, the year 9.10's h9 exit earned 30.5%.
- **Micro, best-pricing book:**
  - exits h12 in 2021–2024 and h6 in 2025–2026; about 103 names a side
  - Σ|Δw| 0.36 a month; beta 0.08; worst month −6.9% (2021-01)
  - 30.5% net at 50 bp
  - by year: 30.5, 20.6, 42.5, 18.6, 52.4, 10.4%

**Post-hoc checks of the micro value gap** (not pre-registered; fixed 12-month staggered book, gross):

| Check | M5 trees | M4 trees | M0 comparables | M3 agnostic |
|---|---|---|---|---|
| Spread 2021–2026 | 33.3% [4.59] | 32.7% [4.74] | 41.1% [4.97] | 38.3% [4.00] |
| Long leg vs benchmark | 8.9% [2.15] | 10.0% [2.54] | 12.7% [2.42] | 12.3% [2.49] |
| Short leg vs benchmark | 24.3% [3.60] | 22.8% [3.56] | 28.4% [3.66] | 26.0% [3.41] |
| Entered one month later | 32.4% [4.39] | 31.6% [4.64] | 39.9% [4.71] | 37.8% [3.96] |
| 2019–2020 spread (long / short leg) | 18.4% [2.14] (20.1 / −1.7) | 17.7% [1.88] | 0.1% [0.01] | −9.5% [−0.74] |
| Fama–MacBeth next month, gap alone / with 6 controls | 3.87 [6.96] / 3.12 [5.73] | 3.85 [6.76] / 3.07 [5.68] | 4.18 [5.76] / 2.81 [5.74] | 5.64 [5.53] / 3.83 [4.01] |
| Fama–MacBeth next 12 months, alone / with controls | 27.1 [8.74] / 18.0 [3.55] | 27.0 [8.14] / 17.6 [3.71] | 34.2 [4.73] / 18.2 [6.84] | 48.5 [3.99] / 27.1 [2.68] |
| Fama–MacBeth 2019–2020, next month with controls | 1.69 [1.49] | 1.26 [1.24] | 0.55 [0.69] | 1.13 [1.09] |

- **Controls:** rank of book-to-market, 12-1 momentum, last-month return, size, idiosyncratic volatility and gross
  profitability. Slopes are % return per unit of centred rank.
- **How the M5 gap relates to known signals** (monthly rank correlation): book-to-market 0.52, momentum −0.25,
  last-month return −0.14, size −0.04, volatility −0.04, profitability 0.06.
- **What the two sides hold:**
  - Shorts: median price $12.2, book-to-market 0.12, daily volatility 4.2% (expensive, glamorous micro caps).
  - Longs: median price $9.9, book-to-market 1.40, daily volatility 3.6%.
- **Reading:**
  - The gap is not short-term reversal or bid-ask bounce: it survives a one-month delay, and the books hold 6–12 months.
  - It is not plain value: it survives the controls.
  - The tree gap also worked in 2019–2020 on the long side, while comparables and agnostic levels did not.
- **Caveats:**
  - About three quarters of the 2021–2026 spread comes from shorting expensive micro caps, during the 2021–2023
    speculative bust. Borrow fees and availability for those names are unknown. The break-even fee is about 33% a year of
    short notional, but hard-to-borrow micro caps can cost that much.
  - Returns are JKP monthly excess returns; delisting-return treatment should be confirmed on CRSP.
  - The variants are one effect, not independent discoveries.

**Post-hoc check of the sector split** (fixed 6-month exit including 2021, net, fold's best model; difference vs E7 [HAC t]):

| Cap | E7 | Tilt all sectors | Tilt book-anchored only | Tilt growth only (tech, communication, health, discretionary) |
|---|---|---|---|---|
| Large + mega | 13.9% [5.59] | 11.2% [2.52]; −2.7 [−0.80] | 14.9% [4.82]; +1.0 [0.60] | 10.0% [2.76]; −3.8 [−1.71] |
| Investable | 8.6% [2.37] | 11.6% [2.22]; +2.9 [0.73] | 10.5% [2.28]; +1.9 [0.81] | 9.8% [2.59]; +1.2 [0.49] |
| Small | 6.4% [1.76] | 5.9% [1.18]; −0.5 [−0.13] | 7.0% [1.68]; +0.6 [0.30] | 5.4% [1.27]; −1.0 [−0.33] |

The micro row was lost in the reboot. In large + mega the direction supports "use valuation only in book-anchored sectors"
(the growth-only tilt costs 3.8% a year), but no difference is significant, so the pre-registered 9.2% vs 7.8% gain is noise.

**Against the pre-registration.**
- **Pricing:**
  - Trees most accurate: yes.
  - Sector codes add little: yes.
  - M8 ≥ M7: yes. M7 ≥ M2: no.
  - Ensemble best: no; the linear members drag it down.
  - Error rises with isolation: yes.
  - Nvidia valued below price 2023–2026: yes through 2025; mixed in 2026.
- **Returns:**
  - "No standalone large + mega or investable book with t > 2": fails once, for M3 in large + mega (12.6%, t 2.50, below
    the multiple-testing bound).
  - "M3 strongest in small/micro": no; trees are strongest in micro.
  - "Most accurate pricing model is not the best return signal": true in large + mega, false in micro.
  - "No tilt beats E7 beyond noise": true outside micro.
  - "Walk-forward sector gate ≥ all-sector tilt": only in investable.
- **Multiple testing:** batch 13 adds 60 books with positions (about 250 in all; Bonferroni t ≥ 3.72).
  - Above the bound: the micro value-gap books (M5 4.81, M9 4.48, M4 4.41, M2 4.34, best-pricing 4.20, M0 4.09, IR-selected
    3.82, M7 3.79) and the large + mega book-anchored split (4.03, whose own reference E7 book here has t 3.84).
  - Earlier variants still above it: two-quarter breadth (G3, t 4.16) and forecast + surprise + breadth (F1c, t 3.97).

**Decisions.**
- **Valuation engine:** gradient-boosted trees on non-price fundamentals + forward analyst EPS (M5/M4) are the best
  "what should this be worth" model; sector labels are not needed.
- **Report isolation with every valuation.** For the most isolated decile (Nvidia-like firms) the model error is 4× the
  least isolated, so the gap there is a statement about expectations, not a trade.
- **MoE step:** a gate on firm traits rediscovers balance-sheet, biotech, profitable-operating and distressed regimes by
  itself; the experts should be nonlinear.
- **New candidate edge: the micro-cap value gap.** It is the strongest standalone result so far, but it is short-driven in
  expensive micro caps. Borrow-fee/availability data and CRSP delisting returns are required before trusting it. The long
  leg alone (9–13% a year vs the micro benchmark, t 2.2–2.5) is the part that needs no borrow.

## 10. Decisions by architecture step

Steps from [*FIAM 2026 Strategy — Team Overview*](../fiam2026_strategy_team_overview.pdf).

| Step | Status | Evidence | Decision |
|---|---|---|---|
| 1 Data layer | Built | Section 2 | Add CRSP daily and I/B/E/S sales data (section 11) |
| 2 Company representation | Built | Section 5.3 | Street state + analyst history + graph peers + characteristics; drop 8-K item counts and FinBERT |
| 3A Market expectations | Built, validated | Sections 5.1, 6.1 | Keep (`src/graph/expectations.py`) |
| 3B Corporate evidence | Weak | Sections 4.2, 5.3, 8 | 8-K item counts and tone add nothing as alpha; restatement, delisting-notice and late-filer flags are candidate short vetoes; LLM scores too sparse |
| 4 Dynamic economic graph | Built, validated for risk and information transfer | Sections 3, 4 | Keep for comps, covariance/risk and peer surprise features |
| 5 Graph-spectral context | Not supported | Section 4.2 | Do not use as alpha |
| 6 Expectation gap | Mechanism supported [FIXED]; surprise forecast validated [WF] | Sections 5.2, 6.1 | Surprise forecast is the core predictive asset; returns unproven |
| 7 MoE predictor | Not built; gate prototype tested | Sections 6.2–6.3, 9.13 | Start from the surprise head; a gate on firm traits (no sector codes) finds valuation regimes by itself; use nonlinear experts |
| 8 Research/valuation agent | Valuation engine chosen | Sections 7, 9.12, 9.13 | Trees on non-price fundamentals + forward EPS as the fair-value model, always reported with the firm's isolation; ICC and warranted multiples as thesis context; valuation does not improve the Street-confirmation book outside micro caps |
| 9 Adversarial committee | Not built | | |
| 10 Robustness-adjusted alpha | Not built | | |
| 11 Risk model | Partial | Section 6.3 | Lagged rolling-beta overlay candidate; graph peers for covariance; Kalman after daily data |
| 12 Optimizer | Mechanics tested (sector-matched 50/50, ≤ 500 names, 200% gross) | Section 6.2 | Rebuild within the final pipeline |

**Rejected ideas:**
- XGBoost + graph features as alpha
- peer-relative fundamentals and spectral bands
- 8-K sentiment and item counts
- sector valuation divergences (version 1 and 2)
- direct graph pair hedge
- 3/6-month staggering
- residual return heads as built
- expected-report catalyst sleeve
- graph weight tuning (no gain)
- 8-K events, Street variables and graph residual reversal/momentum, peer churn and centrality as monthly return signals (section 8)
- micro-cap composite book (2025- and short-reversal-driven, section 8)

## 11. Next steps and data to request

1. **CRSP daily returns** (with delisting returns) and daily market returns. The exploratory Yahoo study (section 9)
   already answered the announcement-window question; CRSP should confirm the post-announcement drift without
   survivorship bias and with factor-adjusted returns, and give a Kalman/daily beta and daily risk marks.
2. **Exact announcement timestamps**, if available beyond I/B/E/S dates.
3. **I/B/E/S sales consensus (SAL):** sales surprises were the graph's strongest transfer channel.
4. **I/B/E/S FY1/FY2 EPS, long-term growth, EBITDA/revenue estimates and price targets:** better Street-implied valuation for the research agent.
5. **I/B/E/S detail files:** only if analyst-level track records are needed.
6. **Trading-cost and short-borrow data** (daily bid–ask spreads, borrow fees, short availability): every result that looked
   strong in sections 7.3 and 8 was short-driven in small or micro caps.
7. **Method:** fixed, age-independent tie-breaks for count/binary signals (section 8.5); fewer pre-specified hypotheses, or
   shrink and combine candidates instead of hard selection.
8. **Standing rule:** every future model follows `src/graph/walk_forward.py` (section 1). Nothing is selected by reading 2021–2026.
   The section 6.2 monthly design is frozen: daily analysis can inform a new design but cannot promote the old result.
9. **Street confirmation (9.10) is the lead.** Before relying on it:
   - confirm it on CRSP monthly/daily returns including delisting returns;
   - check it in the daily data around the post-report I/B/E/S snapshot, to see how much drift a mid-month entry captures;
   - decide deliberately how to live with the post-2021 regime risk, since no pre-2021 validation exists.
10. **Graph edges:** 8-K Item 1.01 is too sparse for supply-chain links (9.11). Request Compustat customer segments and the
    I/B/E/S detail files (analyst co-coverage) for richer directed edges.
11. **Micro-cap value gap (9.13):**
    - Get short-borrow fees and availability for the short leg's names (expensive micro caps above $5).
    - Rerun on CRSP monthly returns with delisting returns.
    - Test the long leg alone as a no-borrow version.
    - Regenerate the lost outputs from the rebuilt scripts; `expectation_panel.parquet` must be rebuilt first for the T families.

## 12. Repository after cleanup

**Kept** (all other research scripts, tests and per-experiment docs from this work were deleted):

| Path | Purpose |
|---|---|
| `docs/graph/research_log.md` | This log |
| `src/graph/economic_graph.py`, `src/graph/business_text_profiles.py`, `configs/economic_graph.json` | Monthly economic graph builder (section 3) |
| `src/graph/expectations.py` | Point-in-time I/B/E/S state, analyst-error history, default-graph peer surprise features, walk-forward surprise model |
| `src/graph/walk_forward.py` | Challenge schedule (`split_masks`), train/validation signal selection and out-of-sample summaries (used by sections 7 and 8) |
| `output/economic_graph_refined/`, `output/business_text_evidence.parquet` | Graph artifacts (expensive to rebuild) |
| `output/expectations/` | Surprise predictions and fold ICs from the kept module (challenge schedule and `_no_validation` files) |

```bash
PYTHONPATH=.deps OPENBLAS_NUM_THREADS=4 python -m src.graph.expectations                       # challenge schedule
PYTHONPATH=.deps OPENBLAS_NUM_THREADS=4 python -m src.graph.expectations --validation-years 0  # no-validation variant
```

Unchanged: earlier committed work (agentic council, XGBoost/TimesFM/FinBERT baselines, their tests and docs)
and the licensed data in `local/` (now git-ignored).

**Archived code to reuse** (`~/fiam26_research_archive_20260914.tar.gz`):
- `src/monthly_surprise_strategy.py`: `screen_universe`, `residualize_score`, `allocate_sector_quotas`, `build_month_portfolio` (section 6.2 book).
- `src/graph_hedged_strategy.py`: `lagged_rolling_beta_hedge`, `constrain_holdings`, `build_graph_hedged_holdings` (section 6.3).
- Alpha search (section 8): `~/fiam26_alpha_search_20260914.tar.gz`.
- Daily event study and Yahoo downloader (section 9): `~/fiam26_event_study_20260915.tar.gz`; price cache `local/yahoo_daily/` (git-ignored).
- The archived scripts import the pre-move module paths (`src.expectations`, `src.walk_forward`, `src.economic_graph`);
  the kept modules now live in `src/graph/`, so use `src.graph.<module>`.

**Optional cleanup:** the 140 `candidates.parquet` files in `output/economic_graph_refined/` (819 MB of 1.7 GB) serve only graph
re-weighting, which was rejected. Delete them and drop `--save-candidates` if disk space matters.
