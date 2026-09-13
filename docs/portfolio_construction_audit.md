# Portfolio Construction Audit

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

Using the existing TimesFM equal-weight output for January 2021 through August
2026 (68 months):

| Diagnostic | Long | Short | Combined |
|---|---:|---:|---:|
| Median market equity (`me`, USD millions) | 1,910 | 31.6 | — |
| Median dollar volume (`dolvol`) | 633m | 14.4m | — |
| Micro + nano share | 32.19% | 93.37% | 62.78% |
| Average one-sided turnover | 64.82% | 40.52% | 52.67% |

The compounded portfolio drawdown is approximately **−31.85%**. The short leg
is the immediate implementation concern: it is dominated by very small and
illiquid names, where borrow costs, buy-ins, price impact, and delisting
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

As an exploratory OOS-only sensitivity, the 25th-percentile size/volume screen
with micro/nano exclusions reduced drawdown to about −23.0%, but also reduced
average monthly return from 3.07% to about 0.70%. This is evidence of a real
risk/return tradeoff, not a validated production setting.

## Remaining research decision

Before submission, select the screen using pre-2021 validation data, rerun the
frozen specification on 2021–2026, and report turnover, market-cap/dollar-volume
coverage, drawdown, short-book quality, and performance net of estimated costs.
