# Portfolio Construction Audit

## Update 2026-09-21: universe conditioning fixed, credit policy stated

All XGBoost/TimesFM book builders (`src/xgb_baseline.py`, `src/xgb_timesfm.py`,
`src/xgb_timesfm_backcast.py`, `src/timesfm_portfolio.py`, `src/repro/repro_lib.py`) previously
dropped rows with missing `ret_exc_lead1m` **before** train/test splitting, so the test
candidate pool — the book universe at eom t — was defined partly by future delistings. Fixed
per the brief rule *"target availability must not define the prediction universe"*:

- Every row with characteristics is scored and eligible for the book (train/validation folds still use target rows only).
- Held names without a recorded forward return are **credited 0 excess return** (the panel carries no delisting returns); counts reported per month (`n_long_missing`/`n_short_missing`).
- A −30% haircut sensitivity on those holdings is reported in the baseline run.
- Feature ranks are now computed over the full point-in-time cross-section (previously survivor-conditioned).

Effect on the XGB baseline book (OOS 2021-26): ann +13.8% → +21.5%, IR 0.29 → 0.57, with
essentially identical long/short sets on shared predictions (100/100 overlap) — most of the
change comes from retraining on the full-cross-section rank base. Pre-change artifacts are
archived as `output/*_pre_conditioned.*`. The 2026-08 month is a disclosed data anomaly
(nano melt-up; see the strategy memo §8) and drives the large max-drawdown figures in both
pipelines.

## Current implementation

The existing TimesFM portfolio uses all eligible rows, ranks by forecast, and
selects the top 100 names for the long leg and bottom 100 for the short leg.
Each leg is equal-weighted, so the intended exposure is 100% long, 100% short,
200% gross, and 0% net. Turnover is measured as the share of each leg replaced
from one month to the next.

The selection currently has no market-cap, dollar-volume, or small-cap screen.
The fields needed for such a screen are present in the characteristics panel:
`me`, `dolvol`, and `size_grp`.

## OOS diagnostics

Regenerated 2026-09-21 (safe forecasts, unconditioned universe). Using the TimesFM equal-weight
output for January 2021 through August 2026 (68 months):

| Diagnostic | Long | Short | Combined |
|---|---:|---:|---:|
| Median market equity (`me`, USD millions) | 1,926 | 22.9 | — |
| Median dollar volume (`dolvol`) | 534m | 13.0m | — |
| Micro + nano share | 31.43% | 92.98% | — |
| Average one-sided turnover | 52% | 51% | — |

The compounded portfolio drawdown is approximately **−29.9%** (includes the disclosed 2026-08
data anomaly). The short leg remains the immediate implementation concern: it is dominated by
very small and illiquid names, where borrow costs, buy-ins, price impact, and delisting
handling can invalidate the paper return.

## Reusable screen

`src/portfolio_constraints.py` provides an opt-in contemporaneous screen:

```python
from src.portfolio_constraints import apply_investability_screen

screened = apply_investability_screen(
    predictions,
    min_me_percentile=0.25,
    min_dolvol_percentile=0.25,
    excluded_size_groups=("micro", "nano"),
)
```

The standalone TimesFM runner exposes the same screen without changing the
default portfolio:

```bash
python src/timesfm_portfolio.py --min-me-percentile 0.25 \
  --min-dolvol-percentile 0.25 --exclude-size-group micro \
  --exclude-size-group nano
```

Percentiles are calculated within each characteristic month, so the screen does
not use future returns. It is deliberately not enabled by default until the
threshold is selected on a training/validation period and then frozen before a
final OOS run.

As an exploratory OOS-only sensitivity (pre-regeneration figures), the 25th-percentile size/volume
screen with micro/nano exclusions reduced drawdown to about −23.0%, but also reduced average
monthly return from 3.07% to about 0.70%. This is evidence of a real risk/return tradeoff, not a
validated production setting; re-derive on the regenerated outputs before citing.

## Remaining research decision

Before submission, select the screen using pre-2021 validation data, rerun the
frozen specification on 2021–2026, and report turnover, market-cap/dollar-volume
coverage, drawdown, short-book quality, and performance net of estimated costs.
