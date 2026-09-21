# 40 time-series experiments for FIAM

This is a research screen to find hypotheses worth confirming. It is not a portfolio submission or a fresh holdout: the team has already examined 2021–2026 extensively. Freeze this menu before the run. Do not keep adding variants in response to the leaderboard.

Experiments 01–36 are implemented in `src/time_series_search.py`; four additional forecasting roles, 37–40, are in `src/time_series_followups.py`. The batches use CPU, the supplied monthly panel, local I/B/E/S extracts, monthly economic graphs, and cached TimesFM forecasts. They do not call LLMs, download data, rent GPUs, fine-tune TimesFM, or run its multivariate mode. Pooled Ridge experiments test whether trajectories add predictive information before spending on a larger sequence model.

## Run

From the repository root:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_overnight --full-characteristics
```

This runs all 40 ideas at 10/25/50 bp, the supplementary controls, and final audits. For just the core 36 ideas at 25 bp, use `python -u -m src.time_series_search`.

To keep it running after closing the terminal:

```bash
nohup env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_overnight --full-characteristics > /tmp/fiam26_time_series_search.log 2>&1 &
```

Progress:

```bash
tail -f /tmp/fiam26_time_series_search.log
```

The core search tests **36 hypotheses + four baselines**, three universes, raw and factor-adjusted scores, and 1/3/6-month overlapping holding cohorts. This makes 240 experiment/universe/control combinations and 720 candidate holding configurations at one cost; each experiment's holding horizon is chosen separately each year using its two-year validation window. The wrapper adds roles 37–40, other costs, controls, and audits. It finishes when the work is done.

Rerun the identical command to resume. Completed combinations are skipped, and the expensive feature panel/model forecasts are cached. Changed code, source timestamps, costs, universes, modes, or holding menu require a new output directory. A different experiment subset can use the same directory.

A smaller first batch:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_search \
  --experiments 15,16,17,18,19,25,29,30,31,32,35,36,B1,B2,B3,B4 \
  --universes large_mega investable --output output/time_series_priority
```

To test another cost assumption, use a separate output directory:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_search \
  --cost-bp 50 --output output/time_series_cost50
```

The full three-cost batch, prediction diagnostics, and combined morning memo can also be reproduced with one command:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_overnight
```

It resumes `output/time_series_cost25/`, `time_series_cost10/`, and `time_series_cost50/`, then runs the four follow-ups, predictive diagnostics, and a snapshot/calibrated-EWMA risk attribution audit. It writes `output/time_series_batch/summary.md`. Across the three costs there are 792 experiment/universe/control/cost combinations and 2,376 logical holding configurations. The combined comparison file adjusts approximate paired p-values across costs as well as experiments/universes/modes. Follow-up trading comparisons use 37 versus 31, 38/40 versus B2, and 39 versus B1.

## The hypotheses

Every direction below is fixed in advance. A failed direction is a rejection, not an invitation to flip it after reading test results. Some ideas extend earlier failed static signals; the new test is the trajectory or conditioning, not proof that the old signal worked.

| ID | Use of time series | Exact first test | What it should add |
|---|---|---|---|
| 01 | Peer-relative momentum | Mean of the last three realized monthly returns minus contemporary graph-peer returns | Trend beyond broad peer moves; compare with momentum baseline |
| 02 | Trend acceleration | Three-month peer-relative mean minus six-month mean | Distinguish strengthening from fading trends |
| 03 | Trend consistency | Three-month relative trend weighted by the strength of six-month positive/negative-month consistency | Distinguish persistent moves from jumps |
| 04 | Recovery from drawdown | Sum of ranks of trailing drawdown depth and latest peer-relative return | Identify deep losers starting to recover; not a pure loser sort |
| 05 | Volatility contraction | Relative trend conditioned on twelve-month minus three-month return volatility | Test whether calmer trends persist |
| 06 | Asymmetric return paths | Relative trend conditioned on trailing positive versus negative squared-return variation | Prefer trends with less downside variation |
| 07 | Liquidity trajectory | Negative six-month change in Amihud illiquidity | Test improving liquidity, beyond current liquidity/size |
| 08 | Attention trajectory | Relative trend conditioned on six-month change in share turnover | Test whether increasing activity confirms a trend |
| 09 | Profitability trajectory | Six-month change in quarterly ROA (`niq_at`) | Test change beyond the current profitability level |
| 10 | Cash-generation trajectory | Six-month change in OCF/assets | Distinguish improving cash generation from accounting profits |
| 11 | Deleveraging | Negative six-month change in debt/market equity | Test improving financing condition; price-driven denominator is a confound |
| 12 | Accrual cleanup | Negative six-month change in operating accruals/assets | Test improved earnings quality |
| 13 | Own-history valuation | Current book-to-market minus its trailing twelve-month mean | Identify unusual cheapness for that company |
| 14 | Quality/valuation divergence | Rank of six-month gross-profitability improvement plus current book-to-market rank | Test improving operations while valuation stays cheap |
| 15 | Consensus acceleration | Same-quarter EPS second difference over two exact monthly lags, scaled by I/B/E/S price | Detect acceleration, beyond revision level |
| 16 | Revision persistence | Sum of two same-quarter EPS revision steps, only when their signs agree | Distinguish persistent updating from a one-off revision |
| 17 | Disagreement resolution | Revision rank weighted more heavily when same-quarter dispersion falls | Test directional agreement emerging |
| 18 | Disagreement expansion | Revision rank weighted more heavily when same-quarter dispersion rises | Competing hypothesis: information uncertainty amplifies the move |
| 19 | Coverage confirmation | Revision rank conditioned on same-quarter change in analyst count | Test whether increased coverage confirms new information; count is not broker identity |
| 20 | Earnings-error innovation | Last public surprise minus mean of last four public surprises | Separate a new beat/miss from a company's routine bias |
| 21 | Earnings-error consistency | Four-quarter mean error divided by eight-quarter error volatility plus a fixed floor | Test predictable versus erratic analyst errors |
| 22 | Peer-news innovation | Current recent-peer surprise minus its trailing three-month mean | Separate new peer information from persistent peer bias |
| 23 | Relative expectation revision | Own three-month revision minus current graph peers' three-month revisions | Identify improving expectations relative to economic peers |
| 24 | Reporting information wave | Aligned peer surprise multiplied by newly reporting peer weight | Test the arrival of peer information rather than its static level |
| 25 | Expectations/price path divergence | Revision rank minus three-month peer-relative price-return rank | Extend earlier static non-repricing tests using relative trajectories |
| 26 | Persistent TimesFM forecast | Three-month mean of cached forecasts | Test whether smoothing removes forecast churn |
| 27 | TimesFM forecast innovation | Current forecast minus its trailing three-month mean | Test whether changes in inferred return dynamics matter |
| 28 | TimesFM uncertainty score | Forecast divided by interval width, with a fixed floor | Test uncertainty-adjusted ranking; intervals are not assumed calibrated |
| 29 | Snapshot return forecast | Pooled Ridge predicts next-month excess return from current company/Street/peer state | Statistical baseline for 30 |
| 30 | Trajectory return forecast | Same model plus a fixed block of return, fundamental, estimate and peer-clock trajectories | Direct test of whether histories add return information |
| 31 | Snapshot fundamental forecast | Pooled Ridge predicts the rank of next announced analyst EPS surprise from current state | Statistical baseline for 32; trade the prediction as a candidate signal |
| 32 | Trajectory fundamental forecast | Same surprise model plus the temporal block | Test improved fundamentals prediction and whether it translates into returns |
| 33 | Time-series risk sizing | TimesFM-ranked book, weights tilted by inverse trailing EWMA volatility | Compare with B3 for risk, turnover, and return retention |
| 34 | Time-varying beta sizing | TimesFM-ranked book, bounded weight tilts using rolling 24-month stock/market beta, minimum 12 observations | Compare with B3 for neutrality; no ETF hedge or synthetic return adjustment |
| 35 | Information-based exit: reversal | Enter after an actual public beat/miss; exit an old position when new comparable-peer news opposes its direction and represents at least 20% of peer weight | Test whether an information change improves exits over B4 |
| 36 | Information-based exit: completion | Same entry; exit when at least 80% of peer weight has reported comparable quarters, or the issuer's reported quarter changes | Test whether graph reporting progress is a useful information clock |
| 37 | Forecast error versus today's consensus | Temporal Ridge predicts eventual actual EPS minus the current snapshot's consensus, scaled by current I/B/E/S price | Separate predicting the actual fundamental outcome from predicting error versus final pre-announcement consensus; seasonal AR gap is the prediction baseline |
| 38 | Forecast the next analyst update | Temporal Ridge predicts next month's same-fiscal-quarter consensus change, scaled by current price | Test whether the expectation path itself is forecastable; no-change is the baseline |
| 39 | Forecast return-path risk | Temporal Ridge predicts log(next-month excess return squared + 0.0001), then inverse-risk weights the momentum baseline | Test risk forecasting separately from signed alpha; EWMA variance is the forecasting baseline; this is a noisy monthly risk proxy, not daily volatility or VaR |
| 40 | Forecast disagreement resolution | Temporal Ridge predicts next month's same-quarter dispersion change; current revision direction receives more weight when resolution is predicted | Test whether impending agreement is more useful than the already-observed dispersion change; no-change is the forecast baseline |

Baselines: **B1** current twelve-month momentum, **B2** current three-month analyst revision, **B3** cached TimesFM forecast, **B4** last public earnings surprise. They use the same screens, portfolio rules, cost, and horizon selection. Models 29 and 31 also serve as ablation baselines for 30 and 32.

The four extra target definitions were written before their runs and are kept fixed. Each uses snapshot + trajectory features and the same four Ridge penalties. Training and validation labels must be public by their cutoffs; **forecasts are still emitted independently of whether the future label exists**. Revisions and dispersion labels join the same fiscal quarter at the next exact calendar month. All four are evaluated at the three stated costs, universes, modes, and holding horizons. Information exits 35/36 begin at subsequent monthly rebalances; each new entry is held for at least one month.

To run 37–40 after the core batch has prepared `output/time_series_cost25/panel.parquet`:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_followups
```

These add 72 experiment/universe/control/cost combinations and 216 holding configurations. The complete 40-idea effort therefore covers **792 combinations and 2,376 logical holding configurations**, plus the separate predictive diagnostics. Extra-model labels are not stock returns, so their MSE and label IC must not be presented as stock-return OOS R².

The fundamental deltas in 09–14 are changes in the observable monthly company state. They are not forecasts constructed by shifting a carried-forward accounting value three rows. Future fundamental outcomes in 31/32/37 use explicitly dated I/B/E/S announcement labels; 38/40 use exact next-month consensus/dispersion observations as labels only.

## Shared protocol

- Universe screens use information at formation month-end: common stock, primary security, main exchange, absolute price at least $5, top 80% by current dollar volume, known sector. **Future target availability does not determine eligibility.** Large+mega and small+large+mega are the main research universes; micro is an implementation diagnostic.
- Per-stock lags use exact calendar months. Same-quarter consensus, dispersion, and coverage changes join on fiscal quarter as well as stock/month; target-quarter rollovers cannot create revisions. I/B/E/S per-share values use its own split-consistent price basis.
- Only already-announced fundamentals enter features. The peer completion clock uses each stock's already-reported quarter and public peer announcements. Graph weights come from the relevant historical month; their underlying classification-vintage caveats remain.
- **TimesFM cache issue discovered while building this screen:** `src/timesfm_forecast.py:load_returns` filters out rows with missing future targets before forecasting. In the current panel, 178 otherwise eligible test-period stock-months with at least twelve observed returns have unavailable future targets, and all 178 have missing cached forecasts. This makes cached forecast availability partly future-conditioned. **26/27/28/33/34/B3 are diagnostics and cannot enter the implementation-candidate list.** Regenerate forecasts from the formation universe independently of future outcomes before treating these tests as clean alpha evidence. Core trajectory models 30/32 deliberately exclude TimesFM features. **Update 2026-09-21:** the generator is repaired and safe forecasts were regenerated (`output/preds_timesfm_all.parquet`, manifest under `output/timesfm_safe_v2/`); tests 26/27/28/33/34/B3 still cite the pre-regeneration cache and must be re-scored on the safe file before they can clear the implementation screen.
- All strategies begin at the next month's start. A new cohort has 50 longs and 50 shorts with matched sector seats. Overlapping cohorts are combined and capped at 250 names per leg; sector dollar exposures are matched, long and short dollars are each 100%, and gross is at most 200%. A book unable to support 100 positions is empty and explicitly flagged. Positions are reweighted monthly; a nominal holding horizon is a maximum cohort life, not buy-and-forget.
- `raw` means the economically directed score with sector-matched construction. `factor` first residualizes the score on eight current characteristic ranks and sector dummies. This tests incremental signal; it **does not guarantee zero portfolio factor exposure** after tail selection. Realized beta and its uncertainty are reported.
- Ridge coefficients are fitted on the expanding training window. Four fixed penalties are selected by validation MSE. Each year's own fitted model scores its validation and test years. Surprise training and validation labels must have been announced by their respective cutoffs; target ranks are calculated among those fold-admissible observations only. Models refit annually.
- Holding horizon is chosen using **net active-return IR** on the prior two validation years. If no valid candidate has positive validation IR, the strategy holds cash for that test year. There is no test-period sign reversal, universe selection, or weight fitting. Cash years appear in the 68-month evaluation and fail the 100-position screen; this is a research rejection rather than a compliant submission.
- Trading cost defaults to **25 bp one way per dollar traded**, including entry, liquidation, drift in prior holdings, and switches between annual choices. NAV collateral earns the same month's TB3MS; active return is stock excess P&L minus costs minus 4%/12. Cash is an accounting return, never a future forecasting feature. Borrow fees and short availability are not modeled.
- A held stock with an unknown realized return invalidates that month. It is **not assigned zero** or dropped from the selection universe. CAGR/drawdown are suppressed if the path has gaps. Mean/IR statistics then cover available months and must not be promoted as complete-window performance. Later turnover accounting is marked uncertain after an unknown NAV outcome.
- P&L uses the supplied `ret_exc_lead1m` for the formation row. There are 694 observations with a valid supplied forward outcome but no next-month characteristic row/return; requiring that next row would create artificial missing outcomes. Historical features still use realized current-month `ret_exc`.
- Alpha/beta use the cached S&P 500 **price-index excess-return proxy**, not a total-return market index. Newey–West/HAC errors use six monthly lags. Daily risk, delisting-return reconciliation, borrow, and certified classification vintages remain separate confirmation requirements.

## What to inspect in the morning

Everything is under `output/time_series_search/`:

| File | Purpose |
|---|---|
| `report.md` | Human-readable leaderboard and completion count |
| `leaderboard.csv` | Every completed experiment/universe/control combination |
| `implementation_candidates.csv` | Only books with no unknown-return months, 100–500 names every month, and absolute realized proxy beta ≤0.10; cached TimesFM diagnostics excluded; a filter, not a proof of alpha |
| `paired_comparisons.csv` | Incremental net return versus the assigned baseline, paired HAC t-statistics, approximate Holm-adjusted p-values across completed comparisons |
| `results/*.json` | Annual validation choices, chosen-horizon results, fixed-horizon diagnostics, cost sensitivities |
| `books/*.csv` | Monthly return, long/short P&L, costs, turnover, exposure, position-count and missing-return diagnostics |
| `books/*_holdings.parquet` | Actual signed research holdings on the selected annual path |
| `model_*_audit.json` | Each fold's penalty, features, admissible training counts, test rank IC, and zero-benchmark OOS R² for the return models |
| `manifest.json` | Frozen menu, settings, source/code fingerprint |
| `failures.json`, `run.log` | Failures and progress; failure in one experiment does not silently stop the others |

`python -m src.time_series_diagnostics` adds `diagnostics/signal_diagnostics.csv` and monthly predictive statistics independently of cash/holding-horizon selection. It ranks public signals, residualizes them using all eligible public scores before observing targets, and counts missing outcomes separately. IC and controlled-return-slope tests are jointly adjusted with approximate Holm p-values. For 33–36 these are only entry-score diagnostics; risk sizing and exit rules must be assessed through their actual portfolios.

The 10/25/50-bp sensitivities hold the selected strategy fixed; they do not retune validation at each cost. Use separate `--cost-bp` runs if that distinction matters.

Read 30 versus 29 and 32 versus 31 first: they answer whether trajectories improve prediction. Then inspect 33/34 versus B3 and 35/36 versus B4. A higher surprise IC without incremental returns is useful fundamental forecasting, not demonstrated trading alpha. Use annual return columns to check whether one year drives the result, and read the short-book capitalization and missing-return flags before ranking an idea by IR.

Holm adjustment uses approximate paired HAC p-values across all completed comparisons, including correlated universes/modes. It does not erase the team's historical discovery overlap, candidate-menu design, incomplete runs, model comparison decisions, or the retrospective selection of the final winner. Treat promising findings as hypotheses for a frozen confirmation study rather than claiming a newly untouched OOS result.

## Regenerating TimesFM safely later

The generator now preserves stocks with unknown future outcomes, uses current realized `ret_exc` through formation month-end, and takes a contiguous monthly suffix after any history gap. Its default output is a separate `output/timesfm_safe/` directory so existing forecasts are preserved. It rejects the old `output/returns_panel.parquet`, which already lost rows through future-target filtering.

On a machine with the TimesFM/PyTorch dependencies and model weights:

```bash
python -m src.timesfm_forecast --start 2016-02-01 --end 2026-08-31 --batch 128
```

Use the full raw panel by default, or export the full formation universe with `(permno, eom, ret_exc, ret_exc_lead1m)`. Safe regeneration has not been performed in this CPU research environment because PyTorch, TimesFM, and the weights are absent. This runner continues to mark the old cached TimesFM tests as diagnostics; swapping a file alone does not remove that restriction.

## Supplementary nonlinear attribution

Six boosted-tree ablations repeat the return snapshot/trajectory, surprise snapshot/trajectory, and risk snapshot/trajectory tests. They select depth 2/4/6 and early stopping entirely on the dated validation loss, use a fixed seed, and retain the same portfolio protocol. This is a retrospective check for nonlinear interactions, not an expansion of the 40 economic hypotheses or a fresh holdout.

```bash
PYTHONPATH=.deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_nonlinear
# Reproduce the entire 40-idea batch and the supplementary ablations:
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_overnight --nonlinear
```

The project-local `.deps` XGBoost works; the global XGBoost installation crashes while loading its native library. The wrapper sets the project-local dependency path for this supplementary step. These six variants add 108 portfolio comparisons and 324 holding configurations, bringing the complete run to **900 comparisons and 2,700 holding configurations**. `output/time_series_prediction_audit/` compares snapshot/trajectory forecast losses and label IC on matched observations, separately for liquid universes. The risk attribution audit additionally compares a calibrated EWMA baseline to avoid treating the log-square target's scale bias as forecast skill.

Four further controls use **all 147 supplied baseline characteristics**, plus the same compact current Street/graph controls, with and without the temporal block. They repeat the return and surprise heads and add 72 portfolio comparisons. The complete effort including these controls has **972 comparisons and 2,916 logical holding configurations**.

```bash
PYTHONPATH=.deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_full_characteristics
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -u -m src.time_series_overnight --full-characteristics
```

The wrapper runs the expensive stages sequentially to limit peak memory. Each fitted tree fold is saved separately. A session interruption can be resumed without refitting completed folds. Changed code/input fingerprints require a fresh output directory; retained fit caches after the accounting-only repair are documented in the results memo.
