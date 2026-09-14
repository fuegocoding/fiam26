"""REPRO SCRIPT 1/3 - memo sections 2 and 3: raw book and rank-buffer book.

Reproduces, from output/preds_timesfm_all.parquet (TimesFM 3.0 forecasts,
target months 2016-02 .. 2026-08):

  1. Base book      : long top-100 / short bottom-100 by forecast, equal
                      weight, monthly rebalance, no buffer          (memo SS2)
  2. Buffered book  : same book with rank buffer k=1500            (memo SS3)

Printed for each: yearly ann. L/S return and Sharpe, full-window ann. L/S,
Sharpe, max drawdown, and monthly turnover per leg.

Memo reference values:
  Base   : ann +23.8%, Sharpe 0.70, maxDD -66.6%, turnover 59%/mo/leg
  Buffer : ann +30.1%, Sharpe 1.02, maxDD -64.5%, turnover 16%/mo/leg
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_lib import (book_returns, load_preds, monthly_sets, perf_stats,
                       turnover_per_leg, yearly_table)

EXPECT = {
    "base": {"ann": 0.238, "sharpe": 0.70, "maxdd": -0.666, "turnover": 0.59},
    "buffer": {"ann": 0.301, "sharpe": 1.02, "maxdd": -0.645, "turnover": 0.16},
}
BASE_YEARLY = {  # memo SS2 yearly (ann, sharpe)
    2016: (0.180, 0.68), 2017: (0.134, 0.40), 2018: (0.647, 2.62),
    2019: (0.002, 0.01), 2020: (-0.520, -0.90), 2021: (0.028, 0.11),
    2022: (0.451, 1.51), 2023: (0.334, 0.74), 2024: (0.708, 3.54),
    2025: (0.444, 2.24), 2026: (0.183, 0.52),
}
BUFFER_YEARLY = {  # memo SS3 yearly (ann, sharpe)
    2016: (0.032, 0.12), 2017: (0.326, 1.06), 2018: (0.817, 3.85),
    2019: (0.132, 0.46), 2020: (-0.505, -1.24), 2021: (0.046, 0.12),
    2022: (0.611, 2.17), 2023: (0.436, 1.33), 2024: (0.480, 2.35),
    2025: (0.442, 2.19), 2026: (0.555, 2.40),
}


def run(label, buffer_k, yearly_ref):
    preds = load_preds()
    long_sets, short_sets = monthly_sets(preds, buffer_k=buffer_k)
    port = book_returns(preds, long_sets, short_sets)
    idx = pd.PeriodIndex(port["target_month"], freq="M")
    stats = perf_stats(port["ret_ls"])
    to_long, to_short = turnover_per_leg(long_sets, short_sets)
    to_avg = (to_long + to_short) / 2.0  # memo reports the leg average

    expect = EXPECT["base" if buffer_k is None else "buffer"]

    print(f"\n=== {label} ({stats['months']} months) ===")
    print(f"ann L/S  : {stats['ann_ret']:+.3f}   (memo {expect['ann']:+.3f})")
    print(f"Sharpe   : {stats['sharpe']:.2f}      (memo {expect['sharpe']:.2f})")
    print(f"maxDD    : {stats['maxdd']:.1%}   (memo {expect['maxdd']:.1%})")
    print(f"turnover : {to_avg:.0%}/mo per leg (long {to_long:.0%}, short {to_short:.0%}) "
          f"(memo {expect['turnover']:.0%}/mo per leg)")

    yt = yearly_table(port["ret_ls"], idx)
    print("\nyear |  ann L/S | Sharpe | memo (ann, Sharpe)")
    for yr, row in yt.iterrows():
        ref = yearly_ref.get(int(yr), (np.nan, np.nan))
        print(f"{yr} | {row['ann_ret']:+7.1%} | {row['sharpe']:5.2f} | ({ref[0]:+.3f}, {ref[1]:.2f})")
    return port


if __name__ == "__main__":
    run("Base k=100 (memo SS2)", buffer_k=None, yearly_ref=BASE_YEARLY)
    run("Buffer k=1500 (memo SS3)", buffer_k=1500, yearly_ref=BUFFER_YEARLY)
