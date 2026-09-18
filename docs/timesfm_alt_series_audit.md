# TimesFM on alternative context series — exploration audit

Date: 2026-09-17. Status: exploration experiment, evaluated out-of-sample only (target months 2021-01 → 2026-08, 68 months). Purpose: the committed TimesFM engine concentrates its alpha in nano/micro names (short leg 95% micro+nano; large/mega ICC ≈ 0), so we tested whether a different single context series lets the same zero-shot forecaster extract alpha elsewhere — specifically in larger caps — and whether anything orthogonal emerges. The committed production specification is unchanged by this experiment.

## 1. What was tested

TimesFM 3.0 (`google/timesfm-3.0-pytorch`, zero-shot, never fit on the panel) was re-run with the identical protocol of `src/timesfm_forecast.py` (one univariate series per stock, horizon 1, median + p10/p90 + spread, `MIN_CONTEXT = 12`, leakage-safe context rule) on four context series:

| Variant | Context series | Rationale |
|---|---|---|
| `exc` (control) | realized monthly excess returns via `ret_exc_lead1m` | must reproduce the committed baseline; validates the harness |
| `resid_mkt` | excess return minus the equal-weight market mean of its month | remove the common market component; direct falsification test for large caps |
| `resid_sec` | excess return minus the equal-weight GICS sector-month mean | same, one level stronger (removes sector momentum) |
| `dolvol` | monthly log1p dollar-volume growth, consecutive months only | different information class (liquidity/attention); regular monthly series |

A derived signal `dolvol_jump = tfm_forecast − mean(last 6 context values)` (expected volume jump vs the recent average) was evaluated on the consumer side.

**Context rule.** For a target month t, the context is the series' values with month ≤ t−1. For `exc` the series month is eom+1 (a realized return of month m lives in the row at eom=m−1), so the rule matches the original script exactly; for contemporaneous columns (`ret_exc`, `dolvol`) the value of month m lives in the row at eom=m. The future target never enters any context. Scripts: `src/timesfm_forecast_alt.py` (A100), `src/timesfm_alt_eval.py` (local CPU, pandas-only).

### Harness validity check — passed

The `exc` control reproduces the committed artifacts (`output/preds_timesfm_all.parquet`) to float precision on every row (max |Δforecast| = 2.4e-8, corr 1.0, zero NaN mismatches) and the standalone portfolio matches the committed evaluation exactly (Sharpe 1.24, +36.8%/yr, maxDD −31.9%, hit rate 74%). Two defects were found and fixed in the evaluation harness during this audit; the first eval pass (published nowhere) wrongly kept NaN forecasts in the book. All numbers below are from the corrected pass.

## 2. Results by variant and size group

Mean monthly Spearman ICC with HAC(3) t-stat; standalone long top-100 / short bottom-100, equal weight, monthly (side = min(100, n//2) within the group).

| Variant | Universe | ICC [t] | Ann. L/S | Sharpe | Max DD | Hit |
|---|---|---:|---:|---:|---:|---:|
| exc (baseline) | ALL | +0.0668 [5.66] | +36.8% | 1.24 | −31.9% | 74% |
| exc | nano | +0.1479 [10.20] | +32.9% | 1.29 | −33.0% | 72% |
| exc | micro | +0.0643 [3.41] | +12.4% | 0.47 | −33.7% | 54% |
| exc | large | −0.0074 [−0.52] | −1.9% | −0.14 | −31.3% | 57% |
| exc | mega | −0.0061 [−0.38] | −1.0% | −0.08 | −25.0% | 60% |
| resid_mkt | ALL | +0.0660 [5.57] | +36.4% | 1.17 | −30.2% | 69% |
| resid_mkt | nano | +0.1529 [10.83] | +29.5% | 1.21 | −34.4% | 74% |
| resid_mkt | large | −0.0003 [−0.02] | +2.0% | 0.14 | −26.8% | 65% |
| resid_sec | ALL | +0.0630 [5.97] | +35.1% | 1.30 | −30.7% | 72% |
| resid_sec | nano | +0.1418 [10.83] | +33.5% | 1.29 | −28.1% | 71% |
| resid_sec | large | −0.0034 [−0.27] | −2.1% | −0.16 | −33.8% | 54% |
| dolvol | ALL | +0.0203 [3.18] | +13.9% | 0.63 | −19.7% | 72% |
| dolvol | nano | +0.0522 [5.11] | +3.9% | 0.18 | −33.0% | 63% |
| dolvol | micro | +0.0285 [3.81] | +14.1% | 0.99 | −12.1% | 56% |
| dolvol | large | −0.0073 [−0.73] | −1.1% | −0.14 | −19.7% | 56% |
| dolvol_jump | ALL | +0.0084 [1.09] | +16.3% | 0.69 | −19.3% | 63% |

Deciles (next-month excess return by forecast decile, 1..10):

- exc/resid variants: classic TimesFM shape — negative extreme decile (−0.98%/−0.97%/−0.85%), flat-to-positive interior, no monotone gradient; the money is the tails.
- dolvol: **flat everywhere** (+0.21% … +0.97% … +0.30%) — a spread-level signal, not a tail signal.
- dolvol_jump: interior hump, top decile ≈ 0 (−0.03%).

### Orthogonality (ALL universe, 68 months)

| Pair | Mean monthly rank corr | Monthly L/S return corr |
|---|---:|---:|
| exc vs resid_mkt | +0.864 | 0.884 |
| exc vs resid_sec | +0.783 | 0.906 |
| dolvol vs exc | −0.015 | 0.251 |
| dolvol vs resid_mkt | −0.026 | 0.139 |
| dolvol vs resid_sec | −0.014 | 0.162 |
| dolvol_jump vs exc | −0.158 | −0.098 |

Yearly monthly L/S: dolvol 2021 −3.85%, 2022 +1.09%, 2023 +0.22%, 2024 +1.61%, 2025 +2.76%, 2026(8m) −0.01% — positive in three of six years; never dominant.

## 3. Findings

1. **The large-cap question is cleanly falsified.** No variant produces any large/mega signal: all |ICC| ≤ 0.007, |t| ≤ 1, and group-restricted books hover at zero. Residualizing — the correct test of "large-cap series are dominated by the market/sector common component" — changed nothing: nano ICC stays ≈ 0.14–0.15 [t ≈ 10–11], large stays ≈ 0. The residual variants are the same signal reshaped (rank corr 0.78–0.87 with the control; L/S corr 0.88–0.91). `resid_sec`'s Sharpe 1.30 vs 1.24 is within noise of a 68-month window and fails a strict pre-registered bar; the spec stays frozen.
2. **New potential alpha found: TimesFM on dollar-volume growth (`dolvol`).** The only variant with a statistically significant signal that the return engine does not have: ICC +0.0203 [t = 3.2] overall, +0.0285 [t = 3.8] micro, +0.0522 [t = 5.1] nano; cross-sectional rank correlation with the return-engine forecasts ≈ −0.01 (orthogonal by construction), portfolio L/S correlation 0.14–0.25. Mechanically it behaves like a liquidity/attention continuation signal read from the shape of the volume-growth series.
3. **But the dolvol signal is not large-cap and not tail-monetizable.** It lives again in micro/nano (large/mega ICC ≈ −0.007), its deciles are flat, so the top/bottom-100 book captures little of it: ALL Sharpe 0.63 (below the pre-registered 0.7 bar), micro 0.99, nano 0.18, small −0.19. It fails as a standalone tail book.
4. **`dolvol_jump` is rejected.** The expected-volume-jump transform weakens the signal (ALL ICC +0.008 [t = 1.1]) and is mildly contrarian to the return engine (−0.10 to −0.17 L/S corr) without enough strength to matter.
5. **Decision:** keep the committed specification unchanged (context = realized excess returns). The credible next use of the dolvol finding is consumer-side: the `dolvol` forecast columns (`tfm_dolvol` forecast/spread) as additional features for the XGBoost combined model, where a spread-level orthogonal signal can actually be exploited by trees — one experiment, not a portfolio. Large-cap alpha remains the XGBoost layer's job (existing unvalidated leads there: filing tone in $10–200bn caps, analyst-error-history features).

## 4. Honest caveats

- **Multiple testing.** Three experimental variants were run on the same 68-month OOS window already used to discover the committed approach (discovery overlap, docs/timesfm_analysis.md §6). The dolvol ICC (t = 3.2) is the standout among three candidates; its true strength is best read as "promising, needs the XGBoost-side confirmation step" rather than established.
- **Turnover.** The dolvol book turns over ~90%/month (long) and ~80%/month (short) versus 65%/40% for the return engine — volume-growth forecasts churn. Any tail or spread book built on it faces materially higher costs than the ICC suggests.
- **Same size concentration.** dolvol's signal (micro/nano) overlaps the existing engine's alpha region in exposure space (L/S corr 0.25 with exc): blending it would diversify less than the rank orthogonality suggests.
- **Nano fragility.** Within-nano, all variants carry −28% to −34% max drawdowns on the L/S book; dolvol's nano book additionally earns almost nothing (+3.9%/yr).
- ICC t-stats are HAC(3) on 68 monthly observations; no benchmark-adjusted alpha is claimed for any variant.
- 2026 (8 months) is negative for both dolvol variants (−0.01%/−0.73% monthly) — the newest period does not confirm the signal.

## 5. Artifacts inventory

| File | Content |
|---|---|
| `output/timesfm_alt/preds_exc.csv` (+ resid_mkt, resid_sec, dolvol) | zero-shot forecasts per variant: 248,674–250,380 rows, feature months 2020-12…2026-07, columns `tfm_forecast`, `tfm_q10`, `tfm_q90`, `tfm_spread`, `tfm_n_ctx`, `ctx_mean6`, target and month keys |
| `output/timesfm_alt/attrs.csv` | permno/target_month attributes (size_grp, GICS sector) for the local evaluation |
| `output/timesfm_alt/eval_summary.csv`, `eval_output.txt` | corrected evaluation summary and full console output |
| `output/timesfm_alt/port_monthly_*.csv` | monthly long/short returns per variant × size group |
| `output/timesfm_alt/run_full.log` | A100 run log |
| `src/timesfm_forecast_alt.py`, `src/timesfm_alt_eval.py` | generation (GPU) and evaluation (CPU) scripts |

## 6. Reproducibility

A100 (RunPod A100-SXM4-80GB; torch 2.4.1+cu124, `timesfm==3.0.2`, pandas 2.3.3, pyarrow 25.0.1):

```bash
# upload chars_final_with_names.parquet to data/raw/ + the two scripts
pip install "timesfm[torch]" "pandas<3" pyarrow
python3 src/timesfm_forecast_alt.py --batch 256      # 68 target months x 4 variants, ~23 min
python3 src/timesfm_alt_eval.py --dir output/timesfm_alt
```

Forecasting is deterministic given the checkpoint (no RNG in inference); the `exc` control's float-precision match against the committed run is the harness check. The evaluation is CPU-only and deliberately avoids parquet/scipy dependencies so it runs anywhere the CSVs land.
