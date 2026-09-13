# TimesFM 3.0 zero-shot forecasting — method, results, and audit

Date: 2026-09-12. Status: experiment evaluated on the out-of-sample window only (target months 2021-01 → 2026-08). No numbers below are from an in-sample backtest of the final pipeline; all results are produced by the scripts listed under "Reproducibility".

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

### Feature join for XGBoost

`src/xgb_timesfm.py` reuses the baseline protocol (`src/xgb_baseline.py`) unchanged — expanding folds (train ≤ Dec(Y−3), val Jan(Y−2)..Dec(Y−1), test year Y for Y = 2021…2026, last fold clipped to 2026-08), cross-sectional rank transform of the 147 characteristics — and appends four TimesFM columns (`tfm_forecast`, `tfm_q10`, `tfm_q90`, `tfm_spread`) joined on `(permno, eom)`. The columns are left unranked: a monotone transform cannot change tree splits, and XGBoost handles their NaNs natively.

### Portfolios

Long top-100 / short bottom-100 by prediction, equal weight, monthly rebalance, dollar-neutral; metrics per the competition definitions (IR vs `TB3MS/100/12 + 4%/12`, HAC alpha/beta vs S&P 500). Vol-scaled variant: within each leg, weights ∝ 1/spread normalized to sum 1.

## 3. OOS results (2021-01 → 2026-08, 68 months)

| Strategy | Ann. L/S | Sharpe | IR vs TB3MS+4% | Max DD | Beta | Alpha ann. (t) | Hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost baseline (chars only) | +13.8% | 0.41 | 0.20 | −47.5% | — | — | — |
| TimesFM standalone, equal weight | +36.8% | 1.24 | 1.00 | −31.9% | −0.32 | +32.9% (3.09) | 73.5% |
| TimesFM standalone, vol-scaled | +33.8% | 1.13 | 0.89 | −31.5% | −0.35 | +30.2% (2.91) | 73.5% |
| **XGBoost + TimesFM features** | **+37.5%** | **1.74** | **1.40** | **−17.5%** | **−0.06** | **+30.9% (3.06)** | 66.2% |

Supporting detail:

- OOS R² (zero-benchmark): TimesFM −0.112 (corr 0.007); XGB+TimesFM −0.002.
- Decile L/S: TimesFM 1.92%/mo (Sharpe 0.95); combined 2.54%/mo (Sharpe 1.43).
- Turnover (avg monthly, one-sided): TimesFM long 64.8% / short 40.5%; combined 51.4% / 49.1%.
- Best/worst month: TimesFM-eq +17.7% (2022-04) / −31.9% (2023-01); combined +15.4% (2025-11) / −11.4% (2021-04).
- Signals are orthogonal: cross-sectional corr(TimesFM forecast, XGB prediction) = −0.02; monthly L/S correlation −0.04.
- Yearly L/S (per month):

| Year | TimesFM | XGB+TimesFM | XGB baseline |
|---|---:|---:|---:|
| 2021 | +0.23% | +2.59% | +4.65% |
| 2022 | +3.76% | +5.51% | +0.25% |
| 2023 | +2.79% | −0.10% | +0.94% |
| 2024 | +5.90% | +2.87% | −2.80% |
| 2025 | +3.70% | +2.51% | +3.12% |
| 2026 (8m) | +1.52% | +6.49% | +0.56% |

TimesFM is positive in all six OOS years; the combined model in five of six.

## 4. Leakage audit — all checks clean

| Check | Result |
|---|---|
| Target/context alignment vs data dictionary | ✓ context stops at month t−1; manual spot-check confirmed the last context value equals the month t−1 return and never the target |
| Permutation null (forecasts shuffled within month) | ✓ Sharpe 1.24 → −0.06, avg monthly L/S −0.0004; a join/leak bug would survive shuffling |
| Merge duplication | ✓ 268,733 rows on both sides, zero duplicate `(permno, eom)` keys |
| Target contamination | ✓ full-sample corr(forecast, target) = 0.007 and OOS R² negative — any lookahead would show strongly positive |
| In-sample exploitation | ✓ zero-shot with frozen weights; the model never fit this panel |

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
- 24% of stock-months get no forecast (history < 12 months or degenerate series); these skew micro/illiquid and the XGBoost side handles the missingness.

## 7. Artifacts inventory (local, post-GPU-sync)

| File | Content |
|---|---|
| `output/preds_timesfm_all.parquet` | 446,206 rows, 6,153 permnos, feature months 2016-01…2026-07: `tfm_forecast`, `tfm_q10`, `tfm_q90`, `tfm_spread`, `tfm_n_ctx`, target and month keys |
| `output/timesfm_parts/*.parquet` | 127 per-month part files from the A100 run (resume cache; reassembled file above is their concatenation) |
| `output/preds_xgb_timesfm.csv` | combined-model OOS predictions |
| `output/port_monthly_timesfm_eq.csv`, `port_monthly_timesfm_volscale.csv` | TimesFM standalone monthly portfolio returns |
| `output/timesfm_long_sets_eq.json`, `timesfm_short_sets_eq.json` (+ `_volscale`) | monthly long/short holdings (permno lists) |
| `output/gpu_env_snapshot.txt` | package versions on the A100 instance |
| `src/timesfm_forecast.py`, `src/timesfm_portfolio.py`, `src/xgb_timesfm.py` | generation, evaluation, and combined-model scripts |

## 8. Reproducibility

GPU run (instance now closed; re-creatable on any CUDA pod):

```bash
# upload: returns_panel.parquet (permno, eom, ret_exc_lead1m extracted from
#         chars_final_with_names.parquet) + src/timesfm_forecast.py
pip install "timesfm[torch]" pandas pyarrow
python3 timesfm_forecast.py --data returns_panel.parquet --batch 256 --start 2015-01-31
# -> /output/preds_timesfm.parquet + /output/timesfm_parts/*.parquet
```

Local evaluation (CPU, minutes):

```bash
python src/timesfm_portfolio.py    # standalone + vol-scaled evaluation
python src/xgb_timesfm.py          # combined XGBoost + TimesFM features
```

Environment on the A100 (see `output/gpu_env_snapshot.txt`): `timesfm==3.0.2`, `torch==2.4.1+cu124`, `numpy==2.4.6`, `pandas==3.0.5`, `pyarrow==25.0.1`, `safetensors==0.8.0`, Python 3.11.10. Forecasting is deterministic given the checkpoint; no RNG is involved in inference.
