# What the time-series search is testing

The FIAM challenge is to build a reproducible, market-neutral U.S. equity research process that beats Treasury bills plus 4% after realistic implementation costs. Its released panel contains 529,082 stock-months, 147 baseline characteristics, and linked 8-K evidence. A good forecast alone is insufficient: the portfolio needs 100–500 positions, gross exposure at most 200%, net exposure within ±50%, little realized market beta, and credible liquidity and short-book quality. The submission also needs a deck, monthly holdings and monthly/daily returns, and a reproducible entry point and agent/tool record.

The official evaluation period is January 2021–August 2026. For test year Y, fit through Y−3 and validate on Y−2/Y−1, then refit annually. Labels must have become observable by the respective fit/validation cutoffs. The team has already studied these years, so the new tests are retrospective hypothesis screens. Neither annual walk-forward fitting nor a multiple-testing adjustment creates a fresh holdout after discovery.

## Where the project stood

The strongest established predictive asset is the analyst-earnings-surprise head. The research log reports materially better surprise forecasting than return forecasting. Economic graphs transfer reporting information and capture common risk; they have not established a stand-alone investable alpha engine. Most static filings, valuation, expectation-gap and graph monthly-return strategies were rejected.

The later research log matters: §9.6 reports a large/mega earnings-drift strategy with validation-selected exits at roughly 10.8% net and IR 0.77, with discovery and implementation qualifications. Newer §9.10 reports the stronger **Street-confirmation** lead: an earnings surprise followed by next-quarter analyst revision breadth in the same direction. It reports broad small/large/mega IR 1.00, positive cost robustness, and all months within the position-count range; adding the pre-report ML forecast improves large/mega IR to 1.21. These are the log's reported results, not independent reproductions by this runner. Its placebo and implementation studies support further work, but the effect was absent before 2021 and the discovery menu was iterative.

The opening conclusions predate those findings. The strongest economic use of time series already reported is therefore **the sequence from earnings news to analyst confirmation**, rather than a price forecast by itself. Later dispersion studies support testing information resolution and event clocks rather than simply buying low dispersion. Cached TimesFM paper returns are heavily concentrated in difficult micro/nano shorts and stocks below $5. §9.11 also rejects a sparse 8-K Item 1.01 supply-chain extraction route; it should not motivate an expensive new inference sweep here.

The new search therefore tests several roles: price and fundamental trajectories; analyst revision, error and disagreement paths; peer-news arrival; direct return versus fundamental forecasting; risk sizing; and information-based exits. [The full menu and commands](time_series_search.md) define 40 hypotheses. Supplementary boosted-tree and full-147-characteristic controls test whether nonlinear interactions or a weak compact baseline explain the results. The latter are attribution checks, not new economic hypotheses.

## What “snapshot versus trajectory” means

The snapshot models already contain historical summaries: momentum, lagged returns, historical surprise averages and beat rates, and risk characteristics. The comparison measures the value of **additional changes and reporting paths**, not the value of all historical information versus none. The full-characteristic arm retains all 147 supplied factors plus the same compact contemporaneous Street/peer controls in both arms.

Do not confuse three different outcomes:

1. Forecasting next announced earnings surprises or future analyst updates.
2. Ranking next-month return risk.
3. Generating incremental, implementable net stock returns over the cash-plus-premium hurdle.

The same data can be useful for the first two without proving the third.

## Execution and data repairs

The runner forecasts the full formation universe independently of whether future outcomes exist. Exact calendar lags preserve gaps; estimate changes require the same fiscal quarter. Historical peer graphs and public announcement clocks supply contemporaneous features. Annual holding-horizon selection reads only validation performance; failed validation holds cash and is reported as a position-rule failure, not a compliant submission.

The existing TimesFM generator selected rows using availability of future return targets. Among otherwise eligible test stock-months with at least twelve observed returns, 178 unknown-target rows also lacked cached forecasts. Its code is repaired and covered by timing tests, and safe forecasts were regenerated on 2026-09-21 with the safe availability policy (`output/preds_timesfm_all.parquet`; manifest under `output/timesfm_safe_v2/`). Tests 26/27/28/33/34/B3 cite the pre-regeneration cache and must be re-scored on the safe file before they can clear the implementation screen.

Portfolio P&L uses each supplied formation row's forward return. There are 694 valid forward outcomes without a next-month characteristic row; requiring that next row would create artificial missing outcomes. Actual position drift uses total stock returns, including the cash rate, and costs include entry, annual changes, and liquidation. Unknown held outcomes invalidate the month and suppress full-window CAGR/drawdown. The supplied readme identifies CRSP/Compustat provenance and a return-source switch to Compustat for all 2026 rows; daily/delisting reconciliation is still required. Superseded development results were discarded.

The interactive session stopped before the corrected run finished. Checkpoints survived and were resumed with fewer simultaneous heavy jobs. The global XGBoost installation also crashes while loading its native library; project-local `.deps` XGBoost works. No TimesFM inference, data downloads, GPU rentals, external messages, or live trades were launched.

## Reading the final results

Final numerical results are assembled in `output/time_series_batch/summary.md`, with combined cost results and paired comparisons beside it. Per-cost JSON files record annual validation choices and fixed-horizon diagnostics; book CSVs and holdings parquet files record actual portfolios. Prediction attribution and calibrated-EWMA risk attribution are separate from portfolio performance.

## Completed predictive findings

These are matched observations, not portfolio returns. Surprise-label comparisons cover 65 months with sufficient observed labels; return/risk comparisons cover all 68 test months. Unknown future labels remain explicitly counted.

| Model/target | Universe | Snapshot IC | Trajectory IC | Reading |
|---|---|---:|---:|---|
| Compact Ridge, return | Large + mega | 0.0327 | 0.0385 | Small, statistically unresolved improvement |
| Compact Ridge, surprise | Large + mega | 0.3468 | 0.3474 | Existing history summaries capture most information |
| Compact trees, surprise | Large + mega | 0.3765 | 0.3799 | Modest improvement |
| Full-factor trees, surprise | Large + mega | 0.3807 | 0.3817 | About 0.001 extra IC; not significant after adjustment |
| Full-factor trees, surprise | Small + large + mega | 0.3406 | 0.3403 | No improvement |
| Compact trees, log squared-return risk | Large + mega | 0.3005 | 0.3068 | Modest extra risk information; not demonstrated net alpha |

The full-factor return models also fail to establish a repeatable liquid-stock predictive gain. Risk-ranking improvement is modest, and the liquid-universe incremental risk comparisons do not survive the predictive family's approximate Holm adjustment. The all-formation results are stronger, but include stocks excluded by the implementation screen.

Calibration matters: large/mega Ridge risk MSE is 3.0604 versus 4.2898 for raw log EWMA variance, but **3.0723** for an EWMA baseline whose scale/intercept is learned on the training window. Most apparent loss improvement over the raw baseline is log-square scale calibration. Additional histories also barely change large/mega Ridge risk IC (0.2974 snapshot versus 0.2972 trajectory).

The other new targets are weaker. Raw current-consensus-error regression has negative rank IC despite a huge MSE improvement over the seasonal forecast; that is not reliable ranking skill. Next-month consensus revision forecasts mostly fail to beat no-change. Dispersion-change rank IC is positive, but loss improvements over no-change are very small and do not establish a trading benefit.

## How the time series can be most useful

**First, preserve the information sequence.** The latest existing lead is the public earnings surprise followed by analyst confirmation for the next quarter. Represent what changed, when it became public, whether it persists, and whether analysts confirm it. This is the clearest reported economic mechanism. Its post-2021 regime dependence and return-data provenance still need confirmation.

**Second, use history summaries as a fundamental forecasting prior.** The earnings-surprise head has useful relative ranking accuracy. Added trajectories bring little incremental information once all 147 factors and public Street/history state are included. A larger sequence model needs to beat this strong baseline before earning additional compute. A surprise rank forecast is not an EPS-dollar forecast or a proven return signal.

**Third, use time series for risk and research control.** They can update risk rankings, track information age and changes, check quarter consistency, and reveal forecast/data availability problems. Apply any risk overlay to the established candidate alpha book only through a separate net-cost validation test. The current momentum/risk tests do not establish that overlay's benefit on Street confirmation.

Clean TimesFM regeneration, complete delisting/return reconciliation, borrow data, and a compliant fallback/optimizer matter more for a final submission than selecting the highest partial IR from this search.

## Completed portfolio screen

All **40 hypotheses plus four baselines** finished at 10/25/50 bp: 792 comparisons. Six compact nonlinear controls and four full-characteristic controls add 180 comparisons. Total: **972 saved portfolio comparisons and 2,916 logical holding configurations**, with annual validation selection among the three holding horizons.

| One-way cost | Comparisons | Cash for all test years | Books with unknown held outcomes | Positive-IR books clearing implementation screen |
|---|---:|---:|---:|---:|
| 10 bp | 324 | 216 | 55 | 0 |
| 25 bp | 324 | 248 | 39 | 0 |
| 50 bp | 324 | 285 | 18 | 0 |

The screen requires a complete return path, 100–500 positions every month, absolute realized proxy beta at most 0.10, and no cached TimesFM contamination. Cash-heavy rejections can result from failed net validation **or an unassessable validation path**; these counts do not prove time series are useless.

At 25 bp, the highest observed partial IR is compact snapshot earnings trees, N31, at **0.486**. It trades only 12 of 68 months, has one unknown held-return month, and proxy beta −0.223. The corresponding trajectory head N32 has IR 0.385 and the same path/position problem. Peer-news innovation, 22, has partial IR **0.285**, trades 48 months, and has one unknown return (the short in permno 63765 for October 2024). Its missing month prevents full-window performance claims. Cash-generation trajectory, 10, has a complete return path but only 20 active months and IR 0.041.

Across **900 assigned baseline comparisons**, zero positive incremental-net-return tests pass the approximate Holm adjustment. Only 676 comparisons have complete paired 68-month windows; the rest remain explicitly flagged as partial. Multiple-testing correction cannot repair discovery overlap, missing outcomes, or absent borrow/delisting data.

The result is a useful research screen, not a submission-ready alpha engine. Keep the reported Street-confirmation lead from §9.10 as the primary candidate to reproduce and confirm; preserve the effective history summaries, avoid paying for a larger sequence model without a stronger ablation result, and resolve implementation/data gaps before promoting partial leaderboard returns.

The combined numerical memo is [output/time_series_batch/summary.md](../../output/time_series_batch/summary.md). It links the interpretation to saved books, annual validation choices, model audits, matched prediction attribution, and independent holdings reconciliation.
