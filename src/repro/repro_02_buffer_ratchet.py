"""REPRO SCRIPT 2/3 - memo section 5.4: rank-buffer book + ratchet sizing.

Reproduces the drawdown-responsive sizing layer on top of the buffered book
(k=1500): exposure starts at 0.5x, steps up +0.1 on each new equity high
(cap 1.0x), and while the strategy is in >10% drawdown scales down
proportional to drawdown depth with a 0.3x floor (exact mapping implemented
in repro_lib.ratchet_scales; the memo text does not pin the formula).

Prints: the memo SS5.4 comparison table (full window / 2016-20 / 2021-26)
and the yearly detail table ("Yearly performance with buffer + ratchet").

Memo reference values (buffer k=1500 + ratchet):
  full: ann +24.5%, Sharpe 1.13, maxDD -40.9%
  16-20: Sharpe 0.64, DD -33.8% | 21-26: Sharpe 1.72, DD -12.8%
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_lib import (book_returns, load_preds, monthly_sets, max_drawdown,
                       perf_stats, ratchet_scales, yearly_table)

MEMO = {
    "base": {"ann": 0.238, "sharpe": 0.70, "maxdd": -0.666},
    "buffer": {"ann": 0.301, "sharpe": 1.02, "maxdd": -0.645},
    "buffer_ratchet": {"ann": 0.245, "sharpe": 1.13, "maxdd": -0.409},
    "sub_16_20": {"sharpe": 0.64, "maxdd": -0.338},
    "sub_21_26": {"sharpe": 1.72, "maxdd": -0.128},
}
YEARLY_REF = {
    2016: (0.037, 0.23), 2017: (0.228, 0.87), 2018: (0.811, 3.88),
    2019: (0.042, 0.18), 2020: (-0.338, -1.35), 2021: (0.008, 0.07),
    2022: (0.264, 2.21), 2023: (0.290, 1.24), 2024: (0.480, 2.35),
    2025: (0.408, 2.07), 2026: (0.555, 2.40),
}


def subwindow_stats(ret, months):
    df = pd.DataFrame({"ret": np.asarray(ret)}, index=months)
    out = {}
    for label, mask in (("2016-20", df.index.year <= 2020),
                        ("2021-26", df.index.year >= 2021)):
        g = df.loc[mask, "ret"]
        out[label] = {
            "sharpe": g.mean() / g.std(ddof=1) * np.sqrt(12),
            "maxdd": max_drawdown(g),
        }
    return out


def main():
    preds = load_preds()
    long_sets, short_sets = monthly_sets(preds, buffer_k=1500)
    port = book_returns(preds, long_sets, short_sets)
    idx = pd.PeriodIndex(port["target_month"], freq="M")
    ret_ls = pd.Series(port["ret_ls"].values, index=idx)

    scales = ratchet_scales(ret_ls)
    ret_ratchet = scales * ret_ls

    for label, ret, memo_key in (
        ("Base k=100", None, "base"),
        ("Buffer k=1500", None, "buffer"),
        ("Buffer k=1500 + ratchet", ret_ratchet, "buffer_ratchet"),
    ):
        if ret is None:
            continue
        stats = perf_stats(ret)
        print(f"\n=== {label} ===")
        print(f"ann L/S : {stats['ann_ret']:+.1%}   (memo {MEMO[memo_key]['ann']:+.1%})")
        print(f"Sharpe  : {stats['sharpe']:.2f}    (memo {MEMO[memo_key]['sharpe']:.2f})")
        print(f"maxDD   : {stats['maxdd']:.1%}   (memo {MEMO[memo_key]['maxdd']:.1%})")

    subs = subwindow_stats(ret_ratchet, idx)
    print("\nsub-window | Sharpe | maxDD   | memo (Sharpe, maxDD)")
    for label, memo_key in (("2016-20", "sub_16_20"), ("2021-26", "sub_21_26")):
        s = subs[label]
        m = MEMO[memo_key]
        print(f"{label}   | {s['sharpe']:.2f}  | {s['maxdd']:.1%} | ({m['sharpe']:.2f}, {m['maxdd']:.1%})")

    yt = yearly_table(ret_ratchet, idx)
    print("\nyear | ann L/S | Sharpe | memo (ann, Sharpe)")
    for yr, row in yt.iterrows():
        ref = YEARLY_REF.get(int(yr), (np.nan, np.nan))
        print(f"{yr} | {row['ann_ret']:+6.1%} | {row['sharpe']:5.2f} | ({ref[0]:+.3f}, {ref[1]:.2f})")

    avg_scale = scales.mean()
    print(f"\naverage ratchet scale: {avg_scale:.2f} (start 0.5x, cap 1.0x, floor 0.3x)")


if __name__ == "__main__":
    main()
