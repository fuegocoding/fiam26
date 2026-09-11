# McGill-FIAM Asset Management Hackathon — First Challenge 2026
## Complete Rule Extraction & File Audit (from official PDF + package readme)

Source: `data/raw/McGill-FIAM Asset Management Hackathon First Challenge 2026.pdf` (27 pages)
+ `data/raw/readme.md`. Nothing omitted. These rules are absolute for the competition.

---

# PART 1 — COMPLETE COMPETITION RULES

## 1. Format
- Two stages:
  - **Stage One (Training Challenge)**: develop portfolio trading strategies with provided datasets using ML/big-data. External data allowed but provided datasets are part of evaluation criteria.
  - **Stage Two (Final Hack)**: top **10 teams**. Additional training, 2-day mentorship from real-time PMs, expanded text coverage, alternative datasets (Carbon Arc), exposure to industry leaders.

## 2. The Task (verbatim objectives)
Design and test an investment strategy leveraging ML, LLMs, agentic workflows, or a combination to:
1. Identify which stock-level factors are most predictive of future returns.
2. Construct a **market-neutral U.S. equity portfolio**, long and short legs balanced so the book carries little or no net market exposure.
3. Evaluate performance through rigorous backtesting.
4. Demonstrate predictive power using **strictly out-of-sample tests**.
5. **Document the workflow** (any agent's plan, tools, prompts) so a third party could reproduce it.

## 3. Hard Trading Criteria (absolute constraints)
- **Positions: 100–500 names** counting long + short together. (60 longs + 60 shorts = 120 ✓; 40 + 40 = 80 ✗.) Floor prevents concentration; ceiling prevents closet indexing.
- **Gross exposure ≤ 200% of capital** (max 2x leverage; longs + |shorts| ≤ $200 per $100 capital).
- **Net exposure ∈ [−50%, +50%] of capital at every rebalancing date** (longs − shorts; $125L/$75S = +50% allowed; $75L/$125S = −50% allowed).
- **Market neutral mandate**: dollar neutrality is the natural centre of the band. Definitions (be explicit which you enforce):
  - Dollar neutrality: $long = $short at each rebalance. Simple, easy to verify.
  - Beta neutrality: beta-weighted long exposure = short exposure → portfolio beta ≈ 0. "Harder, and more honest" — dollar-neutral long high-beta/short low-beta is not neutral.
  - Sector/factor neutrality (optional refinement): neutralize within sectors or vs size/value/momentum — what pro desks do.
- Organizers **will estimate realised beta themselves**. A strategy that sits at the edge of the band month after month, or whose beta to the S&P 500 is materially ≠ 0 over OOS, **is treated as a directional strategy** regardless of description.
- Worked example within rules: $110 long across 150 names, $90 short across 130 names → gross $200 (200%), net +$20 (+20%), 280 positions — all three constraints hold.
- **Rebalancing**: monthly (data are monthly/quarterly). May rebalance less often, but passive is not sustainable for market-neutral. At least **some (not all) holdings must be rebalanced once every half-year**. Monthly rebalancing + monthly turnover reporting is the toolkit default.
- **No trading restrictions beyond exposure limits and rebalancing.**
- Must report: average & maximum gross exposure; average & extreme net exposure; realised beta to S&P 500 over OOS.
- If gross/net exposure is varied over time, must explain what drives the switch and predict it in advance; limits hold throughout; must be able to say why the book was wherever it was.

## 4. Benchmark (absolute)
- **Benchmark = 3-month U.S. T-bill rate (TB3MS) + 4% per annum.**
- TB3MS monthly series from FRED: https://fred.stlouisfed.org/series/TB3MS (annualized % units).
- Monthly decimal conversion: **TB3MS / 100 / 12 + 0.04 / 12**.
- Benchmark is a **time series — compute month by month** across the whole OOS window (hurdle ≈ 7.7% ann. in 2026, ≈ 4% in 2021).
- **Do not subtract the risk-free rate twice**: the RF inside `ret_exc`/`ret_exc_lead1m` is the pipeline's RF, not necessarily TB3MS. Do not assume they cancel.
- Active return: `active_t = R_portfolio,total,t − (TB3MS_t/100/12 + 0.04/12)`.
- **Information Ratio (headline metric) = √12 × mean(active_t) / stdev(active_t)** — applied to active returns, NOT to regression intercept / residual vol.
- Must state: signed position weights, capital denominator, cash/collateral convention. If portfolio total return = benchmark cash return + portfolio excess return, then active = excess − 0.04/12; otherwise retain the difference between cash-rate series.
- Show S&P 500 alongside (context + beta verification via regression), but beating it is not the objective; high beta is not a strength.
- **Sharpe ratio** computed from portfolio returns in excess of the stated risk-free return (no credit for earning the risk-free rate alone).
- Also read: Sharpe, max drawdown, alpha, turnover, how close realised beta sits to zero.
- ⚠ **Shorting is not free**: borrow costs real, hard-to-borrow names can be recalled, concentrated small/illiquid/expensive shorts may be untradeable. No explicit borrow modeling required, but the short book will be scrutinized. Do not chase headline returns by letting net exposure drift long — the beta will reveal it.

## 5. Universe & Sample Periods
- Universe: U.S. public equities.
- Characteristics panel: **01/2015 – 08/2026**, 6,561 distinct PERMNOs.
- Linked filings: 3,687 distinct PERMNOs (all present in the panel).
- **Out-of-sample evaluation: 01/2021 through 08/2026** (post-COVID rebound, 2022 drawdown, AI concentration 2023–2025).
- Text is not available for every stock/month. **Keep stock-months without filings unless the strategy explicitly restricts the universe — and document any restriction.**

## 6. Prediction Target
- Primary: **next-month stock returns**. Long highest expected (e.g. top 100), short lowest expected.
- Trades executed at **beginning of month**, evaluated at **month-end**.
- Label convention (critical):
  - Row labelled by `eom` in month t: characteristics belong to month t; `ret_exc_lead1m` = stock's excess return in **t+1**. Use it as target **without shifting a second time**.
  - `ret` and `ret_exc` describe month t (contemporaneous), not next month. Do not include `ret_exc_lead1m` in predictors.
  - **Do not use current-month `ret`/`ret_exc` as a forward target.** A current month's returns/prices/month-end characteristics are not available at the beginning of that month.
- Alternative targets: forecast fundamentals driving returns (e.g. **ebit_sale, niq_at, niq_be** accounting ratios available in the panel). Predictions must be forward-looking, use only information available at the forecast date; for a quarterly fundamental target, **explicitly identify the future reporting period and release date**. A 3-row shift in a carried-forward accounting panel does NOT identify the next fiscal-quarter report.

## 7. Training Procedure (absolute specification)
- **Expanding training window + rolling 2-year validation window**, all splits assigned by **target return month**.
- First forecast year: train on available target months **2015–2018**, validate **01/2019–12/2020**, test **01/2021–12/2021**.
- Next forecast year: expand training through **12/2019**, validate **01/2020–12/2021**, test **01/2022–12/2022**.
- Continue through **08/2026**; **refit annually, forecast monthly**.
- Because characteristics begin Jan 2015, first target month supplied by `ret_exc_lead1m` is **February 2015**; do not invent December 2014 predictors.
- **A January 2021 return forecast uses December 2020 characteristics.**
- **Labels whose realization falls in the test period must be kept out of training AND validation**, even if their characteristic month precedes the test boundary.
- Tune hyperparameters on validation; estimate coefficients on training; reserve testing for OOS evaluation.
- Fit learned preprocessing on admissible training information only; **do not allow target availability to define the prediction universe**.
- Templates: `penalized_linear_hackathon.py` is a **legacy teaching template** — replace `sample_data.csv` with parquet/model table, map `stock_exret`, update old 2000–2024/8-year settings to the 2015 schedule above. Its dates and line references are NOT the 2026 spec.

## 8. Performance Evaluation (absolute)
- **OOS R² (Gu-Kelly-Xiu)**, benchmark = zero (no predictability):
  `R²_OOS = 1 − Σ_test (r − r̂)² / Σ_test r²` — do NOT subtract the historical mean in the denominator.
  Positive (even tiny) = capturing predictability. Typical stock OOS R² ≈ 1–2%.
  **Must report OOS R² of the methodology in the deck.** Apply to correctly aligned target/forecast pairs.
- Portfolio stats to evaluate: portfolio returns, Sharpe ratio, alpha, beta, information ratio vs cash+4%, maximum loss, drawdown, turnover.
- Alpha/beta: OLS of **portfolio excess-over-RF returns on S&P 500 excess-over-RF returns** for same holding months:
  `R_p,t − r_f,t = α + β(R_mkt,t − r_f,t) + ε_t`. Intercept = alpha (monthly; annualize ×12); slope = beta. **Report standard errors on both alpha and beta.**
- For market-neutral: slope = evidence of neutrality (should sit ≈ 0 over full OOS; far-from-zero beta = mandate failure); intercept should account for nearly all return.
- **Competition IR ≠ template IR**: `portfolio_analysis_hackathon.py`'s printed IR is alpha-to-residual-volatility — NOT the competition's benchmark IR. Reconcile drawdown and turnover definitions with the submission.

## 9. Portfolio Strategy Basics (from PDF)
- Simple strategy: rank by predicted returns, 10 deciles; long decile 10 / short decile 1, equal dollar weights; evaluated month-end. By construction dollar-neutral; **beta still must be checked**; zero cost (shorts finance longs).
- Alternative: buy 100 highest-predicted, short 100 lowest-predicted.
- Position sizing freedom: equal weight, conviction weight, volatility scaling. Neutrality definition choice, rebalance frequency, gross/net scaling are team decisions (with switch logic explained).

## 10. Look-Ahead Rules (absolute, includes agents)
- Cannot use actual events from month t+1 to make decisions in month t. Codes will be checked.
- **Agent-side look-ahead**: a frontier model trained through 2026 already "knows" what happened in 2021 — asking it for a view on that period is **recall, not forecasting**. Constrain the agent to documents and characteristics available at time t, and **state explicitly in the deck how the constraint was enforced. Teams that cannot explain how they ruled out model-side look-ahead will be treated as having used it.**
- Purely replicating the provided template codes is **not sufficient** — build on them with improvements.

## 11. Text Data: 8-K Filings (Quantillium)
- Cleaned 8-K current reports, 01/2015–08/2026, one Parquet (373,139 linked filing observations). Most reportable events have a **4-business-day deadline** (item-specific exceptions).
- Item codes (event taxonomy):
  - **2.02** — results of operations / earnings announcements (release may be an exhibit, not in main text).
  - **1.01 / 1.02** — material agreements: contracts signed/terminated, credit facilities, partnerships, supply deals.
  - **2.01 / 2.05 / 2.06** — transactions and write-downs: completed acquisitions/disposals, restructuring charges, impairments.
  - **5.02** — officer and director changes: CEO/CFO/board turnover (including abrupt).
  - **4.01 / 4.02** — auditor changes and non-reliance: "two of the strongest distress signals in the entire corpus".
  - **7.01 / 8.01** — Regulation FD / other events: guidance, buybacks, litigation, products, anything else.
- A filing may cover several items; main text may reference exhibits NOT included in the package.
- Treat extracted items/dates as **annotations to verify**, not guarantees of single event or exact publication time.
- Timing: `filing_date` = provider date; `filing_timestamp_utc` = date/midnight placeholder; `sec_accepted_at_utc` = null; `content_report_date` = the reported event, NOT a release date. **State a conservative availability convention**; get independent acceptance-time evidence before intraday/same-day execution claims.
- Monthly-frequency recipe: accumulate news + extract sentiment within current month t → forecast month t+1.
- A filing dated the 3rd cannot be used to form a portfolio on the 1st. **Filing timing must respect the portfolio's information cutoff.**
- Join to chars via **permno + dated month key**, cross-checking gvkey/iid. No separate cik_gvkey_linktable needed. Some verified links have no chars row in the exact filing month — retain/exclude those cases deliberately. readme.md has the many-to-one join example + identifier-quality flags.
- LLM usage ideas: encode filing bodies with pretrained LLM (BERT/FinBERT) or summarize into structured records; use embeddings/fields as features (XGBoost/LightGBM); predict next-month returns, next-quarter EPS, or sign/size of post-event drift. Reference: *Can AI Read the Minds of Corporate Executives?* (SSRN). ⚠ Start simple; text modeling escalates fast.

## 12. Agentic AI (optional, showcased)
- Not required. "A well-executed classical ML strategy will beat a poorly conceived agentic one." If built: document what it did, show prompts + tool definitions, be honest about failures (candid failure > polished success).
- Distinction: a model that scores text produces a **number**; an agent produces a **process** (inspectable, replayable, criticisable).
- Requested use cases:
  1. **Analyst agent**: ticker + that month's 8-Ks + characteristic history → structured verdict (what happened, expected?, signed conviction score). Run across the cross-section; treat output as a factor; does it survive OOS?
  2. **Event-triage agent**: read item code + body; distinguish routine dividend declaration from auditor resignation → cleaner event set.
  3. **Signal-discovery agent**: propose/code/backtest combinations of the 147 characteristics. Judged on how top idea holds up in an unseen period. ⚠ Free search finds spurious patterns faster than any human; **multiple-testing discipline is your responsibility**.
  4. **Devil's-advocate agent**: attacks the first agent's thesis (disconfirming filing, crowded position, factor exposure). Disagreeing-agent ensembles are promising.
  5. **Portfolio-construction agent**: forecasts → positions under constraints (100–500 names, 200% gross, ±50% net, turnover budget, sector limits); agent writes why each trade was made. **Explainability is a deliverable.**
  6. **Reproducibility agent**: re-runs pipeline hunting look-ahead bias, survivorship, leakage before submission. "We will be checking; you should check first."
- ⚠ Warnings: fluency ≠ accuracy (every agent claim must trace back to provided data); **agents crowd** (same models + same filings → same names → correlated unwind; "ask a frontier model what it likes" is an edge several hundred teams have). The interesting work: what you feed it, how you constrain it, how you verify it.

## 13. External Data (optional)
- WRDS allowed if university has access (e.g. van Binsbergen, Han & Lopez-Lira 2023 style: 67 ratios predicting EPS).
- If merged, include: **data-cleaning code + clear source description**.
- Join keys: preferably **permno + month**, or **gvkey + iid + month**. CIK is issuer-level; CUSIP can change — verify effective periods; never join on undated company name/ticker.

## 14. Daily Measurement (optional, differentiator)
- Monthly selection is the decision + backtest requirement. **Monthly results considered equally with daily**; daily analysis is not required and won't rescue a weak strategy; careful monthly analysis is not penalised.
- But daily marking is strongly encouraged: monthly returns hide intramonth drawdowns/vol/VaR; margin is calculated on the daily path.
- Sources: CRSP daily files via WRDS; or Alpha Vantage API (free key, TIME_SERIES_DAILY; check limits, cache results).
- If reporting daily-based figures alongside monthly, label clearly which is which.

## 15. Deck Guidelines (absolute submission format)
PowerPoint slide format, converted to PDF. **8 pages + appendix ≤ 10 pages.**
- **Page 1 — Executive summary**: strategy, ML/agentic choice, performance vs market-neutral benchmark (T-bill + 4%), S&P 500 for context, market/benchmark data covering complete 01/2021–08/2026 window.
- **Page 2 — Investment strategy**: long/short leg construction, which neutrality definition enforced, predictive signals. **Top 10 long + top 10 short average holdings over OOS, every position by ticker AND full company name.** Cumulative return chart vs benchmark and vs S&P 500 for 01/2021–08/2026. Label every holding/chart/table row by ticker + company name ("NVDA, NVIDIA Corporation"). Use permno + dated gvkey–iid for merging. If a holding lacks a period label, verify historical display labels from an authoritative source and record the source; do not substitute a later company name without verification.
- **Page 3 — Data & methodology**: justify the ML/LLM choice; training structure; any new architecture/approach; optional extra data + whether valuable; **OOS R² statistics for the overall sample for ML algorithms used**. If agentic: what the agent could see, tools callable, step budget, model used, how model-side look-ahead was prevented. Prompts + tool definitions in Appendix.
- **Pages 4–7 — Performance pack** (OOS 01/2021–08/2026, strategy vs benchmark vs S&P 500):
  - *Return stats*: avg monthly return + annualized (arithmetic AND geometric CAGR); cumulative OOS return; calendar-year returns year-by-year (strategy/benchmark/S&P); best & worst month with dates; hit rate (share of months with positive active return); **long leg and short leg returns reported separately**.
  - *Risk-adjusted*: **IR annualized (headline)**; Sharpe annualized; annualized alpha vs S&P 500 **with t-stat**; beta vs S&P 500 **with standard error**; correlation of monthly returns with S&P 500.
  - *Exposure/concentration/implementation*: avg number of holdings (long/short split); avg gross & net exposure each with range; avg & max single-position weight; share of book in top-10 names; avg monthly turnover with range; short-book characteristics (avg market cap, avg dollar volume, share of shorts in small-cap/hard-to-borrow names).
  - *Charts (minimum)*: cumulative return vs benchmark & S&P; underwater (drawdown) chart with S&P overlay; rolling 12-month active return; rolling 12-month IR; **rolling 12-month beta to S&P 500** (clearest neutrality picture); histogram of monthly returns with benchmark hurdle marked; **10 largest positive + 10 largest negative P&L contributors by ticker and company name**.
  - Headline table + 3–4 charts on pages 4–7; rest in Appendix. Label every table/chart with its period; state gross or net of trading costs.
- **Page 8 — Discussion**: did it perform as trained/expected? Main fundamental signals driving performance. Most profitable positions and why. Macro events that contributed. Potential improvements. Every stock named by ticker + full company name.
- What the committee wants: the pack an investment committee would actually be handed.

## 16. FINAL SUBMISSION PACKAGE (5 items, absolute)
1. **Deck + appendix in ONE PDF** (PowerPoint slide format → convert to PDF).
2. **CSV file with monthly stock holdings for long and short positions for EACH month of testing period 01/2021–08/2026.** Columns: `Date` (beginning of month, e.g. Sept 01 2026), `PERMNO`, `TICKER`, `COMPANY NAME`, `WEIGHT` (% of portfolio NAV invested in that stock for that month; positive = long, negative = short). **Plus a separate CSV with the time series of total portfolio monthly (or daily) returns** for the testing period.
3. **Python code: ONE file named `MAIN.py`** clearly describing the process. **DO NOT submit the data** to run it. The only data submitted is the holdings CSV from #2.
4. **Licence: public domain / free** (no licence; idea + backtest performance is what's evaluated).
5. **CVs for each team member** (used for recruiting by industry participants; keep up to date).

## 17. Evaluation Criteria
- Committee: industry experts + finance academics.
- Two main things: **the investment idea** (which factors you focus on/predict) and **the ML tools chosen** to bring it to life.
- Originality rewarded (e.g. predict earnings surprises, P/E, price-sales instead of raw returns; or novel technology: feed-forward NNs, gradient boosting, transformer text models, agentic pipeline).
- What is rewarded: **quality of reasoning — why this technology, for this signal, on this data** — not sophistication of the tool.
- Performance compared to market-neutral benchmark (T-bill + 4%); **neutrality is checked before anything else**. Main focus: originality, technology choice, execution of the backtest — not just returns.

## 18. Looking Ahead: Carbon Arc (finals only)
- Cannot use in Stage One; do not design the Stage One submission around it.
- Alt-data categories: transaction/receipt panels, web & app activity, healthcare claims, point-of-sale/supply-chain, labour & weather.
- Prep habits: (1) build pipeline so a new feature block merges on a **firm-month key** without rewriting downstream ("most teams lose their first day in the finals to a join"); (2) know your weakest signals + where outside observation would help most.
- Possible extra prizes for most creative use of their data (judged separately; originality over Sharpe).

## 19. Compute Resources
- LightningAI: free account, ~35h free GPU per account per month. Google Cloud trial, AWS free tier for deep learning.

## 20. Key References
- Gu, Kelly, Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS 33(5):2223–2273.
- Goyenko & Zhang (2022), joint cross-section options/stock returns with big data + ML (McGill WP).
- Chopados, Fan, Goyenko, Laradji, Liu, Zhang (2023), *Can AI Read the Minds of Corporate Executives?* (McGill WP).
- Jensen, Kelly, Pedersen (2022), *Is there a replication crisis in finance?* JF.
- Tibshirani (1996) LASSO; Zou & Hastie (2005) Elastic Net; Hoerl & Kennard (1970) Ridge.
- van Binsbergen, Han, Lopez-Lira (2023), *Man versus Machine Learning*, RFS 36(6):2361–2396.
- FRED TB3MS series for the benchmark.

## 21. Appendix A (PDF) — Characteristic glossary
The PDF appendix contains "Table 1: Stock-specific Features" (feature → acronym → academic reference), split over 3 pages:
- Page 25: `age` → `emp_gr1` (44 rows: firm age, liquidity measures, leverage, betas incl. Dimson/BAB/downside, bid-ask spread, CAPEX growth, accrual-related changes, profitability ratios, Amihud measure, dollar volume, distress/debt ratios, hiring rate…).
- Page 26: `eq_dur` → `ocf_at_chg1` (49 rows: equity duration, issuance/payout, Piotroski F-score, gross profits, intrinsic value, inventory growth (Belo & Lin 2011), idiosyncratic skew/vol variants, KZ index, market equity, mispricing factors (Stambaugh & Yuan 2016), earnings surprise/persistence/volatility, quarterly ROA/ROE, O-score, accruals, OCF ratios).
- Page 27: `tangibility` → `zero_trades_252d` (8 rows: tangibility, tax surprise, turnover, Z-score, zero trades).
- ⚠ **Source-document gap**: the PDF table jumps from `ocf_at_chg1` to `tangibility`, omitting ~46 acronyms present in `factor_char_list.csv` (ocf_me, ocfq_saleq_std, op_at, op_atl1, ope_be, ope_bel1, opex_at, pi_nix, ppeinv_gr1a, prc, prc_highprc_252d, qmj, qmj_growth, qmj_prof, qmj_safety, rd_me, rd_sale, rd5_at, resff3_12_1, resff3_6_1, ret_1_0, ret_12_1, ret_12_7, ret_3_1, ret_6_1, ret_60_12, ret_9_1, rmax1_21d, rmax5_21d, rmax5_rvol_21d, rskew_21d, rvol_21d, sale_bev, sale_emp_gr1, sale_gr1, sale_gr3, sale_me, saleq_gr1, saleq_su, seas_1_1an, seas_1_1na, seas_2_5an, seas_2_5na, sti_gr1a, taccruals_at, taccruals_ni). All 147 acronyms ARE in the CSV + readme; the omitted descriptions match the canonical open-source asset-pricing characteristic table (momentum/residual-momentum lags, max-return lottery signals, QMJ variants, R&D ratios, sales ratios, seasonality Heston-Sadka, accruals). Verified against the full-page renders — no hidden images in the PDF.

---

# PART 2 — FILE AUDIT: `data/raw/`

| File | Size | Status |
|---|---:|---|
| `chars_final_with_names.parquet` | 412.6 MB | ✅ NOT opened (per instruction). SHA-256 **verified** vs readme. |
| `8k_20150101_20260831_identified.parquet` | 358.1 MB | ✅ NOT opened (per instruction). SHA-256 **verified** vs readme. |
| `McGill-FIAM ... 2026.pdf` | 2.4 MB | ✅ Fully extracted (27 pages; Appendix A pages are images — viewed + transcribed). |
| `factor_char_list.csv` | 1.6 KB | ✅ Audited: 1 column `variable`, exactly 147 case-sensitive characteristic names (matches readme list). |
| `readme.md` | 27 KB | ✅ Audited: full schema/join/timing documentation (summarized below). |
| `penalized_linear_hackathon.py` | 7.5 KB | ✅ Audited: legacy ML training template (needs adaptation — see below). |
| `portfolio_analysis_hackathon.py` | 5.0 KB | ✅ Audited: legacy portfolio evaluation template (needs adaptation — see below). |

Derived artifacts created during audit: `_pdf_extract.txt` (full PDF text), `_pageNN_imgN.png` + `_fullpageNN.png` (Appendix A table images). Can be deleted later.

## 2.1 `chars_final_with_names.parquet` (per readme — metadata only)
- 529,082 rows × 198 columns; observation unit = security × calendar month; unique on **(permno, eom)** and **(gvkey, iid, eom)**.
- 6,561 distinct PERMNOs; 6,811 GVKEY–IID pairs. Coverage: `date` 2015-01-30 → 2026-08-28; `eom` 2015-01-31 → 2026-08-31.
- Columns = **147 selected characteristics** (the CSV list) + **51 identifiers/auxiliary** columns. The 51 (explicit list): id, permno, permco, gvkey, iid, excntry, exch_main, common, primary_sec, bidask, primaryexch, conditionaltype, crsp_nyse, shrcd, exchcd, crsp_shrcd, crsp_exchcd, comp_tpci, comp_exchg, curcd, fx, date, eom, adjfct, shares, me, me_company, prc_local, prc_high, prc_low, dolvol, tvol, ret, ret_local, ret_exc, ret_lag_dif, source_crsp, ret_exc_lead1m, obs_main, gics, sic, naics, ff49, size_grp, ret_exc_wins, me_lag1, ticker, company_name, ticker_name_reference_date, ticker_name_source, ticker_name_status.
- Key columns:
  - `date` = last source return observation date in the stock-month (**can precede calendar month-end**); `eom` = month-end label.
  - `ret` = current-month return (decimal; 0.01 = 1%). `ret_exc` = current-month excess (pipeline RF). **`ret_exc_lead1m` = NEXT-month excess return = the target**; missing in terminal month & where outcome unavailable. Never a feature. Never replace terminal null with zero.
  - `ret_exc_wins`: CRSP values unchanged; Compustat values clipped at period-specific 0.1%/99.9% cutoffs.
  - `source_crsp`: 1 = CRSP, 0 = Compustat (**all 2026 rows are Compustat**).
  - `prc`, `shares` (millions), `me`/`me_company` (USD millions), `me_lag1`, `adjfct` (split adjustment factor), `size_grp` (NYSE breakpoint size categories), `gics`/`sic`/`naics`/`ff49` industry codes.
  - Ticker/name labels: `ticker_name_status` — 498,706 verified in observation month; 15,731 verified via filing; **14,645 rows have NO verified label (null, not backfilled)**; 514,437 rows labeled.
  - Numeric missingness preserved from source; missing ≠ zero. Preprocessing/imputation is each team's job.
- Built from WRDS CRSP/Compustat, methodology associated with Jensen-Kelly-Pedersen (2023) pipeline (JKP factor site); separately constructed U.S. extract.
- **Timing facts**: characteristics row for month t pairs with target return in t+1. Jan-2021 forecast ← Dec-2020 characteristics row. First target month: Feb 2015.

## 2.2 `8k_20150101_20260831_identified.parquet` (per readme — metadata only)
- 373,139 rows × 94 columns; unit = retained filing observation × security; unique on `document_id` and on **(permno, filing_date, text_sha256)**. 373,054 Form 8-K + 85 Form 8-K/A.
- Coverage: `filing_date` 2015-01-02 → 2026-08-31; 3,687 PERMNOs. Filings/year: 2015: 23,426 · 2016: 24,039 · 2017: 25,362 · 2018: 26,944 · 2019: 28,167 · 2020: 32,973 · 2021: 35,124 · 2022: 35,946 · 2023: 38,181 · 2024: 39,411 · 2025: 39,608 · 2026 (Jan–Aug): 23,958.
- Every retained filing security appears in the characteristics panel; 6,092 filings for 42 securities with no chars observations were excluded.
- Same-month link: 370,201 filings; 2,699 link outside month; 239 span history gaps → consult `characteristics_link_basis`, `characteristics_link_date`.
- `text` = cleaned provider main text (HTML/whitespace cleaned; substantive text NOT summarized/stemmed/stopped). Exhibits/attachments NOT guaranteed present. `text_chars`, `word_count`, `text_sha256` quality fields.
- `items` = detected 8-K item labels (extraction metadata, not guaranteed exhaustive).
- Date fields (careful): `filing_date` (provider, unverified vs SEC acceptance), `content_report_date` (event date, not availability), `filing_timestamp_utc` (midnight placeholder), `sec_accepted_at_utc`/`accession_number`/`isin` (all null), `provider_added_at_utc`/`retrieved_at_utc`/`processed_at_utc` (processing times, not availability). `days_filing_after_content_report` = filing_date − content_report_date.
- `quality_flags`, `source_quality_flags`, `cleaning_flags` — lists of caveats.
- Identifiers: every row has nonmissing permno, gvkey, iid, cusip, cik, ticker, company_name. CUSIPs from dated CRSP records (349,181 monthly-snapshot precision; 23,958 rows for 2026 inferred continuity). FIGI alt-links: 898 rows; CIK-vs-Compustat-header differs: 1,081 rows. Keep identifiers as strings; tickers/names not permanent join keys.
- Source: Quantillium provider 8-K archives for NYSE (`UN`) / Nasdaq (`UW`) exchange codes.

## 2.3 `penalized_linear_hackathon.py` — LEGACY template, required adaptations
What it does: loads `sample_data.csv` (date = first day of **return** month t+1), reads `factor_char_list.csv`, keeps rows with non-missing `stock_exret`; per month: cross-sectional **median-fill missing → dense rank → scale to [−1, 1]**; StandardScaler fit on train; **OLS / Lasso / Ridge / ElasticNet** (fit_intercept=False, Y de-meaned by train mean, Lasso alpha grid 10^-4..10^4, Ridge 10^-1..10^8 ×0.5, EN −4..4) tuned on validation MSE; expanding window; writes predictions; prints **OOS R² = 1 − Σ(y−ŷ)²/Σy²**.
Legacy settings that MUST change for 2026 spec:
- Input: CSV → our parquet/model table; construct `stock_exret` = `ret_exc_lead1m`, and template `date`/`year`/`month` = **target month t+1** (preserving original `eom`/`date` separately).
- Windows: old 2000 start / 8y train / 2y val / 1y test through 2024 → new 2015-based expanding schedule (train available targets 2015–2018, val 01/2019–12/2020, test 01/2021–12/2021; expand annually through 08/2026).
- Splits by **target return month**; test-period realizations excluded from train/val.
- Its `._append` usage is deprecated (use concat); scaler must be fit on train only (it is).
- OOS R² output block is fine as reference (correct GKY formula), but apply to aligned pairs.

## 2.4 `portfolio_analysis_hackathon.py` — LEGACY template, required adaptations
What it does: reads predictions CSV (year, month, permno, stock_exret, model); decile rank (rank×10/(N+1) floor); equal-weight portfolio returns per decile; long-short = port_10 − port_1; Sharpe = mean/std ×√12; merge market factor; **Newey-West CAPM regression** (HAC maxlags 3); prints alpha, t-stat, "IR" = α/√mse_resid ×√12 (**NOT competition IR**); max 1-month loss; log-cumsum max drawdown; turnover = % of names replaced per month (start vs next-month start merge on MonthBegin-shifted date).
Adaptation requirements (from PDF):
- `stock_exret` must refer to the **target holding month** (map from ret_exc_lead1m; date/year/month = holding month).
- Enforce 100–500 position count and exposure limits (template does NOT enforce them).
- State capital normalization; compute **competition IR vs TB3MS+4%** (formula in §4 above), not α/resid-vol.
- Reconcile drawdown and turnover definitions with those reported in the deck.
- Add: benchmark series (TB3MS+4%), S&P 500 for context regression, long-leg/short-leg separate returns, exposure reporting, net/gross tracking.

## 2.5 Repository state
- Git repo initialized; `.gitignore` = `data/raw/*.parquet` (parquets not tracked — correct, they're 358/413 MB and redistribution is prohibited in submission anyway).
- No other project structure yet (no src/, no docs/, no notebooks).

---

# PART 3 — INITIAL PLAN SKELETON

## Phase 0 — Data infrastructure (next step)
- Read **metadata only** of both parquets (pyarrow `ParquetFile` schema) — no full loads. Convert CSV list → typed feature selector.
- Build small DuckDB/pyarrow-based loaders (lazy, column-pruned, eom-filtered).
- Verify: row counts, distinct permnos, date ranges, missingness map of the 147 chars, target coverage (rows with non-null `ret_exc_lead1m` by month).
- Download external series: FRED TB3MS (benchmark), S&P 500 total/excess returns + RF (FRED St. Louis) covering 01/2021–08/2026 (and training window for context).
- Design the **model table**: key = (permno, eom), features = 147 chars (+ optional filing aggregates), label = `ret_exc_lead1m`, target-month key explicit.

## Phase 1 — Baseline reproduction (the required starting point)
- Implement 2026-spec expanding-window pipeline: train 2015–2018 targets / val 01/2019–12/2020 / test 01/2021–12/2021 → expand annually to 08/2026.
- Run OLS/Lasso/Ridge/EN on the 147 rank-transformed chars. Record OOS R² per year and overall.
- Monthly portfolio: rank → long top / short bottom (decile spread AND 100-vs-100 variant), equal weights, dollar-neutral, within all constraints (100–500 names, ≤200% gross, ±50% net).
- Compute full metric pack with COMPETITION definitions (IR vs TB3MS+4%, Sharpe excess-RF, alpha/beta with SEs, drawdown, turnover, exposures).

## Phase 2 — Improvements (creativity axes)
- Better models (GBM, NN), better preprocessing, feature engineering, hyperparameter discipline.
- Text: 8-K event signals (item-code triage, sentiment/LLM encoding) accumulated within month t → month t+1 features. Conservative availability convention.
- Neutrality upgrades: beta-neutral and/or sector-neutral legs (professionally standard).
- Optional daily-risk layer (CRSP daily / Alpha Vantage).
- Consider agentic components (analyst/event-triage/devil's advocate) with documented look-ahead guards.

## Phase 3 — Deliverables
- 8-page deck + ≤10-page appendix → single PDF.
- Monthly holdings CSV (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT signed) 01/2021–08/2026 + returns CSV.
- `MAIN.py` single-file pipeline.
- Team CVs.
- Reproducibility pass (look-ahead/survivorship/leakage hunt) before submission.
