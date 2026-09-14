# Economic graph research log (FIAM 2026): graph, sell-side expectations and valuation

Everything tried between 11 and 14 September 2026 by Claude Code and Codex, in one place: what was
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
9. [Decisions by architecture step](#9-decisions-by-architecture-step)
10. [Next steps and data to request](#10-next-steps-and-data-to-request)
11. [Repository after cleanup](#11-repository-after-cleanup)

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
- **A lagged rolling-beta overlay** is the only portfolio-level improvement found (return 8.06% → 10.12%, t 1.45,
  realized beta −0.27 → −0.10). It is a risk-control candidate, not proven alpha.
- **Next:** CRSP daily returns (announcement-window and drift tests of the surprise forecast) and
  I/B/E/S sales consensus. See [section 10](#10-next-steps-and-data-to-request).

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
   announcement dates are needed to test and trade it (section 10, item 1).
2. **Trading-cost and borrow data.** Every result that looks strong is short-driven and in small or micro caps, exactly where
   spreads, borrow fees and short availability are not observed.
3. **Statistical power.** The first test year trains on four years of data, and hundreds of correlated candidates pass or fail a
   hard |t| ≥ 2 cut. Fewer, pre-specified hypotheses, or shrinking and combining candidates instead of selecting, would waste less signal.
4. **A broad cross-sectional signal.** A 100–500 name neutral book needs a score that ranks the whole investable cross-section.
   What has any power here is sparse (events) or confined to micro caps. The surprise forecast is the closest candidate;
   its return channel is still untested at event time.
5. **Sales and margin expectations** (I/B/E/S SAL) are still unmeasured.

## 9. Decisions by architecture step

Steps from *FIAM 2026 Strategy — Team Overview*.

| Step | Status | Evidence | Decision |
|---|---|---|---|
| 1 Data layer | Built | Section 2 | Add CRSP daily and I/B/E/S sales data (section 10) |
| 2 Company representation | Built | Section 5.3 | Street state + analyst history + graph peers + characteristics; drop 8-K item counts and FinBERT |
| 3A Market expectations | Built, validated | Sections 5.1, 6.1 | Keep (`src/graph/expectations.py`) |
| 3B Corporate evidence | Weak | Sections 4.2, 5.3, 8 | 8-K item counts and tone add nothing as alpha; restatement, delisting-notice and late-filer flags are candidate short vetoes; LLM scores too sparse |
| 4 Dynamic economic graph | Built, validated for risk and information transfer | Sections 3, 4 | Keep for comps, covariance/risk and peer surprise features |
| 5 Graph-spectral context | Not supported | Section 4.2 | Do not use as alpha |
| 6 Expectation gap | Mechanism supported [FIXED]; surprise forecast validated [WF] | Sections 5.2, 6.1 | Surprise forecast is the core predictive asset; returns unproven |
| 7 MoE predictor | Not built | Sections 6.2–6.3 | Start from the surprise head; return heads on residual targets were not tradable |
| 8 Research/valuation agent | Inputs available | Section 7 | Use ICC, warranted multiples and price-implied growth as thesis context only |
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

## 10. Next steps and data to request

1. **CRSP daily returns** (with delisting returns) and daily market returns: test the surprise forecast around
   peer announcements, before and on the firm's announcement, and in post-announcement drift; estimate a Kalman/daily beta.
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

## 11. Repository after cleanup

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
- The archived scripts import the pre-move module paths (`src.expectations`, `src.walk_forward`, `src.economic_graph`);
  the kept modules now live in `src/graph/`, so use `src.graph.<module>`.

**Optional cleanup:** the 140 `candidates.parquet` files in `output/economic_graph_refined/` (819 MB of 1.7 GB) serve only graph
re-weighting, which was rejected. Delete them and drop `--save-candidates` if disk space matters.
