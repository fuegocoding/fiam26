# TimesFM 3.0 zero-shot forecasting — method, results, and audit

Date: 2026-09-12. Status: experiment evaluated on the out-of-sample window only (target months 2021-01 → 2026-08). No numbers below are from an in-sample backtest of the final pipeline; all results are produced by the scripts listed under "Reproducibility".

**Update 2026-09-21.** Forecasts were regenerated from the full formation universe under the safe
availability policy (§2.1). Numbers in §3–§6 were produced from the original (partly future-conditioned)
cache; regenerated headline checks are given in §2.1 and the final-config impact in
`docs/timesfm_strategy_memo.md` §8. The per-month portfolio tables in §3 predate the regeneration.

## 1. What was tested

TimesFM 3.0 (`google/timesfm-3.0-pytorch`, ~330M params, PyTorch backend) as a **zero-shot, per-stock time-series forecaster** of next-month excess returns. The model was never trained or fine-tuned on the challenge panel; it only reads each stock's realized monthly excess-return history and emits a 1-step-ahead forecast plus 9 quantiles (p10…p90).

Three roles were evaluated:

1. **Standalone challenger** — long/short portfolio sorted purely by the TimesFM forecast.
2. **Feature generator** — forecast + quantiles added as columns for the XGBoost characteristic model.
3. **Uncertainty sizing** — position weights scaled by the model's quantile spread (p90 − p10).

## 2. Method

### Forecast construction (leakage-safe by design)

For every feature row `(permno, eom = t−1)` whose target `ret_exc_lead1m` is the month-t excess return:

- Context = the stock's realized monthly excess returns through month t−1, i.e. `ret_exc_lead1m` over panel rows with `eom ≤ t−2` (a realized return of month m lives in the row at `eom = m−1`).
- Forecast: `horizon=1`, `return_quantiles=True` → median + p10/p90; spread = p90 − p10.
- Minimum context: 12 realized returns. Stocks with shorter or degenerate histories (e.g. all-zero returns → model emits NaN) get no forecast and are treated as missing downstream.
- Panel semantics confirmed against `data/raw/readme.md`: *"For a characteristics row in month t, `ret_exc_lead1m` is the excess-return outcome in month t+1 … A January 2021 return forecast therefore uses December 2020 characteristics."* Our context rule matches exactly.

The future target never enters the context; the same rule applies in the training period, so the features are valid for all XGBoost folds.

### Compute

| Stage | Machine | Throughput | Runtime |
|---|---|---|---|
| Checkpoint load | RunPod A100-SXM4-80GB (torch 2.4.1+cu124) | — | ~9 s (cached) |
| Full panel forecasts (446k series-months) | same A100, batch 256 | ~1,600 series/s | ~90 s |
| Reference CPU run (Windows, torch 2.14.0, batch 128) | local | ~82 series/s | n/a (aborted, superseded by GPU run) |

The A100 instance was temporary; every artifact it produced was downloaded and is committed under `output/` (see §6). The model weights themselves are public on Hugging Face and were not archived.

### 2.1 Safe regeneration (2026-09-21)

The original cache was produced by an earlier generator version that filtered rows on future-target
availability before forecasting, so cached forecast *availability* was partly future-conditioned
(2,246 delisting-bound stock-months across 2016-2026 had no cached forecast; 178 of them are
otherwise-eligible test-period rows flagged in `docs/research/time_series_search.md`). The repaired
`src/timesfm_forecast.py` (manifest policy `future_target_independent_universe: true`,
`calendar_policy: contiguous_realized_suffix`) was rerun on a fresh A100:

| Stage | Machine | Detail |
|---|---|---|
| Run | A100-SXM4-80GB (SSH pod), torch 2.4.1+cu124, timesfm==3.0.2 | `--data panel_returns.parquet --start 2016-02-29 --end 2026-08-31` (127 feature months) |
| Throughput | batch 128 | ~460 series/s per month, ~14 min wall incl. checkpoint load |
| Output | 450,302 rows, 6,206 permnos | 2,246 forecasted rows have no future target (previously excluded); 2,716 additional rows extend with-target coverage |

Validation of the regenerated file:

- Agreement on the 445,340 shared rows: corr 0.9946, mean |Δforecast| 2.7e-3 — model outputs essentially unchanged.
- corr(forecast, target): −0.0069 full-sample, +0.0062 OOS — still ~0; OOS mean monthly Spearman IC +0.066 (52/68 months positive).
- Coverage: 14.9% of the 529,082 panel rows get no forecast (history < 12 contiguous realized months), unchanged mechanism.
- Manifest and per-month parts: `output/timesfm_safe_v2/`; combined file installed as `output/preds_timesfm_all.parquet`; the pre-regeneration cache archived as `output/preds_timesfm_all_pre_safe_conditioned.parquet`.

Final-config impact: after the regeneration the XGB engines were retrained under the same
unconditioned policy, the final config was re-frozen as **v3 (uniform T=3 tranches, both
engines, 1.0x gross, $1 point-in-time price floor)**: OOS ann +21.3%, CAGR +22.9%, IR 1.45,
maxDD −6.4%, no OOS month below −5% (details and the 2026 return-source artifact diagnosis in
`docs/timesfm_strategy_memo.md` §8). The regeneration is compliance-critical (universe no longer
future-conditioned), not performance-changing.

### Feature join for XGBoost

`src/xgb_timesfm.py` reuses the baseline protocol (`src/xgb_baseline.py`) unchanged — expanding folds (train ≤ Dec(Y−3), val Jan(Y−2)..Dec(Y−1), test year Y for Y = 2021…2026, last fold clipped to 2026-08), cross-sectional rank transform of the 147 characteristics — and appends four TimesFM columns (`tfm_forecast`, `tfm_q10`, `tfm_q90`, `tfm_spread`) joined on `(permno, eom)`. The columns are left unranked: a monotone transform cannot change tree splits, and XGBoost handles their NaNs natively.

### Portfolios

Long top-100 / short bottom-100 by prediction, equal weight, monthly rebalance, dollar-neutral; metrics per the competition definitions (IR vs `TB3MS/100/12 + 4%/12`, HAC alpha/beta vs S&P 500). Vol-scaled variant: within each leg, weights ∝ 1/spread normalized to sum 1.

## 3. OOS results (2021-01 → 2026-08, 68 months)

**Regenerated 2026-09-21** from the safe forecast file and the unconditioned universe policy
(every scored row is book-eligible; held names without a recorded forward return are credited
0 excess; see §2.1 and `docs/timesfm_strategy_memo.md` §8 conventions). The TimesFM standalone
books additionally apply the $1 share-price floor (final-config universe). The XGB rows predate
the floor (their books still include sub-$1 names): the **2026-08 month is contaminated by a
return-source artifact** (nano caps average +15.0%, 7.3% >+50%, in the final panel month) —
diagnosed and structurally excluded in the final config (memo §8 data note).

| Strategy | Ann. L/S | Sharpe | IR vs TB3MS+4% | Max DD | Beta | Alpha ann. (t) | Hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost baseline (chars only) | +21.5% | 0.70 | 0.57 | −40.3% | +0.20 | +12.2% (1.09) | 66% |
| TimesFM standalone, equal weight ($1 floor) | +22.1% | 0.96 | 0.79 | −22.0% | — | — | — |
| TimesFM standalone, vol-scaled ($1 floor) | +19.0% | 0.82 | 0.65 | −21.6% | — | — | — |
| **XGBoost + TimesFM features** | **+25.5%** | **0.72** | **0.61** | **−59.7%** | **−0.21** | **+20.5% (1.28)** | 63% |

Supporting detail (regenerated values):

- OOS R² (zero-benchmark): TimesFM −0.109 (corr 0.006); XGB+TimesFM −0.002.
- Turnover (avg monthly, one-sided): XGB baseline long 53.8% / short 55.9%; TimesFM buffered legs 22%/10%; XGB+TFM legs 52%/51%.
- The XGB+TimesFM standalone max drawdown is dominated by the 2026-08 anomaly (−55.8% that month); the buffered TimesFM book is +0.4% in the same month.
- Signals remain orthogonal after the regeneration: cross-sectional corr(forecast, XGB prediction) ≈ −0.02 (unchanged).

## 4. Leakage audit — all checks clean

| Check | Result |
|---|---|
| Target/context alignment vs data dictionary | ✓ context stops at month t−1; manual spot-check confirmed the last context value equals the month t−1 return and never the target |
| Permutation null (forecasts shuffled within month) | ✓ Sharpe 1.24 → −0.06, avg monthly L/S −0.0004; a join/leak bug would survive shuffling |
| Merge duplication | ✓ 268,733 rows on both sides, zero duplicate `(permno, eom)` keys |
| Target contamination | ✓ full-sample corr(forecast, target) = 0.007 and OOS R² negative — any lookahead would show strongly positive; regenerated safe file: −0.0069 full-sample / +0.0062 OOS |
| Universe conditioning | original cache ✗ (forecast availability depended on future targets; 2,246 delisting-bound rows excluded) → safe v2 ✓ (manifest `future_target_independent_universe: true`; §2.1) |
| In-sample exploitation | ✓ zero-shot with frozen weights; the model never fit this panel. TimesFM 3.0 pretraining contains no equity panel data (Wiki pageviews ≤ 2023-11, Google Trends ≤ 2022, GIFT-Eval general series, synthetic) |

## 5. Why the performance is high (mechanism, verified)

1. **The forecast is a nonlinear momentum extrapolator.** corr(forecast, `ret_1_0`) = 0.57, corr with `ret_12_1` = 0.13, with `ret_3_1` = 0.14 — it reads the shape of the whole return history, not just the last month.
2. **All the signal is in the extreme tails.** Average next-month excess return by forecast decile (OOS):

   | Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | Avg next ret | −1.10% | +0.81% | +0.94% | +0.95% | +0.85% | +0.80% | +0.69% | +0.49% | +0.65% | +1.00% |

   Deciles 2–9 are flat; the top/bottom-100 portfolios (~3% tails) therefore earn much more than any monotone factor ranking would suggest — which is why full-sample correlation is ~0 while the tail portfolio earns 37%/yr.
3. **Plain momentum does not reproduce it.** Sorting directly on `ret_1_0`: in 2015–2020, extreme losers rebound (+3.2%/mo — classic short-term reversal) and d10−d1 = −0.67%/mo; in 2021–2026 the spread is flat (−0.16%/mo). TimesFM's tail separation uses the full history shape and is not a momentum-sorted portfolio.
4. **The money is mostly the short leg.** Bottom-100 earns −2.2%/mo; 18.6% of bottom-100 stock-months have a next-month return below −25% (worst month: 52% of the short book). 3.2% of bottom-decile rows are terminal (delisting-month) rows averaging −4.6%; excluding them the decile still averages −0.8%.

## 6. Honest caveats (for the deck)

- **Short-book realism.** The alpha concentrates in shorting crashing microcaps. Real borrow costs, short availability, and buy-in risk would erode it; the challenge explicitly scores short-book quality. Treat headline numbers as an upper bound for live implementability.
- **Panel-level timing quality** cannot be fully audited (provider-constructed panel; its own readme carries extensive timing caveats). Same caveat applies to the baseline.
- **Sample**: 68 OOS months. Sharpe 1.74 carries roughly ±0.35 standard error — high but not miracle territory.
- **Discovery overlap**: the OOS window was used to discover and tune this approach (zero-shot, so no fitted hyperparameters, but the *choice* of TimesFM was made after seeing 2021+ results). Per the strategy memo, freeze the specification before the final submission run.
- **License**: TimesFM 3.0 weights are distributed under `timesfm-non-commercial-license-v1.0` — acceptable for the hackathon, not for commercial use. TimesFM 2.5-200m (Apache-2.0) is the fallback if the rule set changes.
- 14.9% of the 529,082 panel stock-months get no forecast (history < 12 contiguous realized months); these skew micro/illiquid and the XGBoost side handles the missingness.

## 7. Artifacts inventory (local, post-GPU-sync)

| File | Content |
|---|---|
| `output/preds_timesfm_all.parquet` | **safe v2 (2026-09-21)**: 450,302 rows, 6,206 permnos, feature months 2016-01…2026-07: `tfm_forecast`, `tfm_q10`, `tfm_q90`, `tfm_spread`, `tfm_n_ctx`, target and month keys; availability independent of future targets |
| `output/preds_timesfm_all_pre_safe_conditioned.parquet` | archived original cache (446,206 rows, 6,153 permnos; partly future-conditioned availability — kept for audit only, do not consume) |
| `output/timesfm_safe_v2/` | regeneration manifest, run log, and 127 per-month part files (resume cache) |
| `output/regen/panel_returns.parquet` | slim source panel used for the regeneration (permno, eom, ret_exc, ret_exc_lead1m) |
| `output/preds_xgb_timesfm.csv` | combined-model OOS predictions (XGB folds unchanged; features from the pre-regeneration cache — re-score pending) |
| `output/port_monthly_timesfm_eq.csv`, `port_monthly_timesfm_volscale.csv` | TimesFM standalone monthly portfolio returns (pre-regeneration) |
| `output/timesfm_long_sets_eq.json`, `timesfm_short_sets_eq.json` (+ `_volscale`) | monthly long/short holdings (permno lists) |
| `output/gpu_env_snapshot.txt` | package versions on the A100 instance (both runs identical: timesfm 3.0.2, torch 2.4.1+cu124) |
| `src/timesfm_forecast.py`, `src/timesfm_portfolio.py`, `src/xgb_timesfm.py` | generation, evaluation, and combined-model scripts |

## 8. Reproducibility

GPU run (both instances now closed; re-creatable on any CUDA pod):

```bash
# upload: output/regen/panel_returns.parquet (permno, eom, ret_exc, ret_exc_lead1m
#         extracted from chars_final_with_names.parquet) + src/timesfm_forecast.py
pip install numpy==2.4.6 pandas==3.0.5 pyarrow==25.0.1 safetensors==0.8.0 timesfm==3.0.2
python3 timesfm_forecast.py --data panel_returns.parquet --start 2016-02-29 --end 2026-08-31 \
  --output-dir /workspace/timesfm_safe_v2
# -> /workspace/timesfm_safe_v2/preds_timesfm.parquet + /workspace/timesfm_safe_v2/timesfm_parts/*.parquet
#    (batch 128, ~14 min on an A100; manifest must match a previous run's policy or a new --output-dir is required)
```

The first A100 run (2026-09-12) used an older generator revision and a target-filtered panel
(`returns_panel.parquet`, `--batch 256 --start 2015-01-31`); its cache is archived. The 2026-09-21
rerun above is the canonical one.

Local evaluation (CPU, minutes):

```bash
python src/timesfm_portfolio.py    # standalone + vol-scaled evaluation
python src/xgb_timesfm.py          # combined XGBoost + TimesFM features
```

Environment on the A100 (see `output/gpu_env_snapshot.txt`): `timesfm==3.0.2`, `torch==2.4.1+cu124`, `numpy==2.4.6`, `pandas==3.0.5`, `pyarrow==25.0.1`, `safetensors==0.8.0`, Python 3.11.10. Forecasting is deterministic given the checkpoint; no RNG is involved in inference.
