"""REPRO SCRIPT 5 - parameter-disclosure sensitivity tables (memo section 8).

Reproduces every table cited in the section 8 multiple-testing disclosure,
evaluated at the current final conventions unless the item itself varies
them (TimesFM buffered k=1500 monthly; XGBoost raw in T=3 staggered
tranches; blend 0.8x gross; SPX sleeve 0.25x price return; no ratchet).

  T1  TimesFM buffer-k grid (standalone book, 2016-2026) - k=1500 is the argmax
  T2  XGBoost buffer-k grid (standalone book, 2018-2026) - rejected
  T3  XGB tranche cadence T in the blended final - the turnover control
  T4  TimesFM cadence swap: buffer vs staggered tranches, standalone + blended
  T5  ratchet ablation at the blended final (incl. ratchet off)
  T6  blend-weight sensitivity at the blended final
  T7  book-gross monotone table at the blended final

Runtime note: T1/T2 re-run the buffered set construction per k (the slow part).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_lib import (book_returns, load_preds, max_drawdown, monthly_market,
                       monthly_sets, ratchet_scales, staggered_sets, turnover_per_leg)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"


def to_series(sets):
    ms = sorted(sets)
    vals = {}
    for prev, cur in zip(ms, ms[1:]):
        if len(sets[cur]):
            vals[cur] = len(sets[cur] - sets[prev]) / len(sets[cur])
    return pd.Series(vals)


def leg_series(preds, long_sets, short_sets):
    port = book_returns(preds, long_sets, short_sets)
    idx = pd.PeriodIndex(port["target_month"], freq="M")
    return (pd.Series(port["ret_long"].values, index=idx),
            pd.Series(port["ret_short"].values, index=idx))


def turnover_rng(long_sets, short_sets):
    tl, ts = to_series(long_sets), to_series(short_sets)
    return tl, ts


def blend_stats(ls_t, ls_x, mkt, common, g=0.8, w_spx=0.25, ratchet=None):
    ls_b = 0.5 * (ls_t.reindex(common) + ls_x.reindex(common))
    comb = g / 2 * ls_b + w_spx * mkt["spx_ret"].reindex(common)
    r = ratchet_scales(comb, **ratchet) * comb if ratchet else comb
    bench = mkt["benchmark"].reindex(common)
    act = r - bench
    oos = common.year >= 2021
    yrs = len(common) / 12
    return {
        "cagr": ((1 + r).prod()) ** (1 / yrs) - 1,
        "cagr_oos": ((1 + r[oos]).prod()) ** (12 / 68) - 1,
        "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(12),
        "ir": np.sqrt(12) * act.mean() / act.std(ddof=1),
        "ir_oos": np.sqrt(12) * act[oos].mean() / act[oos].std(ddof=1),
        "mdd": max_drawdown(r),
        "mdd_oos": max_drawdown(r[oos]),
        "worst": f"{r.idxmin()} {r.min():+.1%}",
    }


def fmt_blend(label, m):
    print(f"  {label:26s}: CAGR {m['cagr']:+6.1%} IR {m['ir']:.2f} Sh {m['sharpe']:.2f} "
          f"mdd {m['mdd']:6.1%} (OOS {m['mdd_oos']:.1%}) | OOS CAGR {m['cagr_oos']:+.1%} "
          f"IR {m['ir_oos']:.2f} | worst {m['worst']}")


def main():
    tfm = load_preds()
    xgbp = pd.read_csv(OUT / "preds_xgb_timesfm_backcast.csv")
    xgbo = pd.read_csv(OUT / "preds_xgb_timesfm.csv")
    for d in (xgbp, xgbo):
        d["target_month"] = pd.to_datetime(d["target_month"])
    xgb = pd.concat([xgbp, xgbo], ignore_index=True).rename(
        columns={"pred_xgb_tfm": "tfm_forecast"})
    mkt = monthly_market()

    # ---- final-config legs ----
    lt_f, st_f = monthly_sets(tfm, buffer_k=1500)
    rl_t, rs_t = leg_series(tfm, lt_f, st_f)
    ls_t = rl_t - rs_t
    lx_f, sx_f = staggered_sets(xgb, T=3)
    rl_x, rs_x = leg_series(xgb, lx_f, sx_f)
    ls_x = rl_x - rs_x
    common = (ls_t.index.intersection(ls_x.index).intersection(mkt.index))
    common = common[common >= pd.Period("2018-01")]

    # T1 - TimesFM buffer-k grid (standalone, 2016-2026)
    print("T1  TimesFM buffer-k grid (standalone L/S, 2016-2026):")
    print(f"    {'k':>6} {'ann':>7} {'Sharpe':>6} {'maxDD':>7} | {'TO long':>14} {'TO short':>14}")
    for k in (None, 500, 1000, 1500, 2000, 2500):
        l, s = monthly_sets(tfm, buffer_k=k)
        rl, rs = leg_series(tfm, l, s)
        r = rl - rs
        tl, ts = turnover_rng(l, s)
        lab = "raw" if k is None else k
        print(f"    {lab!s:>6} {r.mean()*12:+7.1%} {r.mean()/r.std(ddof=1)*np.sqrt(12):6.2f} "
              f"{max_drawdown(r):7.1%} | {tl.mean():.0%}/mx {tl.max():.0%} {ts.mean():.0%}/mx {ts.max():.0%}")

    # T2 - XGBoost buffer grid (standalone, 2018-2026) - rejected
    print("\nT2  XGBoost buffer-k grid (standalone L/S, 2018-2026):")
    print(f"    {'k':>6} {'ann':>7} {'Sharpe':>6} {'maxDD':>7} | {'TO long':>14} {'TO short':>14}")
    for k in (None, 150, 300, 600, 1000, 1500):
        l, s = monthly_sets(xgb, buffer_k=k)
        rl, rs = leg_series(xgb, l, s)
        r = rl - rs
        tl, ts = turnover_rng(l, s)
        lab = "raw" if k is None else k
        print(f"    {lab!s:>6} {r.mean()*12:+7.1%} {r.mean()/r.std(ddof=1)*np.sqrt(12):6.2f} "
              f"{max_drawdown(r):7.1%} | {tl.mean():.0%}/mx {tl.max():.0%} {ts.mean():.0%}/mx {ts.max():.0%}")

    # T3 - XGB tranche cadence T in the blended final
    print("\nT3  XGB tranche cadence T in the blended final (0.8x + 0.25x SPX):")
    print(f"    {'T':>3} {'NAV TO avg/max':>15} | blend stats")
    tl_t, ts_t = turnover_rng(lt_f, st_f)
    for T in (1, 2, 3):
        lx, sx = staggered_sets(xgb, T=T)
        rl, rs = leg_series(xgb, lx, sx)
        m = blend_stats(ls_t, rl - rs, mkt, common)
        tl, ts = turnover_rng(lx, sx)
        idx = tl.index.intersection(tl_t.index)
        nav = 0.2*tl_t[idx] + 0.2*tl[idx] + 0.2*ts_t[idx] + 0.2*ts[idx]
        fmt_blend(f"T={T}", m)
        print(f"      NAV 1-sided turnover: avg {nav.mean():.0%}/mo, max {nav.max():.0%}")

    # T4 - TimesFM cadence swap: buffer vs tranches
    print("\nT4  TimesFM cadence swap (standalone, 2016-2026):")
    print(f"    {'mechanism':>24} {'ann':>7} {'Sharpe':>6} {'maxDD':>7} | {'TO long':>14} {'TO short':>14}")
    tfm_variants = {}
    for label, mk in (("buffer k=1500", lambda: monthly_sets(tfm, buffer_k=1500)),
                      ("T=3 tranches", lambda: staggered_sets(tfm, T=3)),
                      ("T=2 tranches", lambda: staggered_sets(tfm, T=2))):
        l, s = mk()
        rl, rs = leg_series(tfm, l, s)
        r = rl - rs
        tl, ts = turnover_rng(l, s)
        print(f"    {label:>26} {r.mean()*12:+7.1%} {r.mean()/r.std(ddof=1)*np.sqrt(12):6.2f} "
              f"{max_drawdown(r):7.1%} | {tl.mean():.0%}/mx {tl.max():.0%} {ts.mean():.0%}/mx {ts.max():.0%}")
        tfm_variants[label] = (rl, rs)
    print("    blended final with each TFM mechanism:")
    for label, (rl, rs) in tfm_variants.items():
        fmt_blend(f"TFM {label}", blend_stats(rl - rs, ls_x, mkt, common))

    # T5 - ratchet ablation at the blended final
    print("\nT5  ratchet ablation at the blended final (0.8x + 0.25x SPX, XGB T=3):")
    fmt_blend("no ratchet (final)", blend_stats(ls_t, ls_x, mkt, common))
    base = dict(start=0.5, step=0.1, floor=0.3, dd_gate=0.10, depth_mult=2.0)
    for label, kw in (("ratchet as-is", {}), ("depth_mult 1.5", dict(depth_mult=1.5)),
                      ("depth_mult 2.5", dict(depth_mult=2.5)),
                      ("gate 0.05", dict(dd_gate=0.05)), ("floor 0.5", dict(floor=0.5)),
                      ("start 1.0 (off)", dict(start=1.0, step=0.0, floor=1.0, dd_gate=2.0))):
        kw_full = {**base, **kw}
        fmt_blend(label, blend_stats(ls_t, ls_x, mkt, common, ratchet=kw_full))

    # T6 - blend-weight sensitivity at the blended final
    print("\nT6  blend weight (0.8x book, XGB T=3):")
    for w in (0.3, 0.5, 0.7):
        m = blend_stats(ls_t, ls_x, mkt, common)
        ls_b_w = w * ls_t.reindex(common) + (1 - w) * ls_x.reindex(common)
        r = 0.4 * ls_b_w + 0.25 * mkt["spx_ret"].reindex(common)
        bench = mkt["benchmark"].reindex(common)
        act = r - bench
        oos = common.year >= 2021
        print(f"    w_tfm={w}: CAGR {((1+r).prod())**(1/(len(common)/12))-1:+6.1%} "
              f"IR {np.sqrt(12)*act.mean()/act.std(ddof=1):.2f} Sh {r.mean()/r.std(ddof=1)*np.sqrt(12):.2f} "
              f"mdd {max_drawdown(r):6.1%} | OOS IR {np.sqrt(12)*act[oos].mean()/act[oos].std(ddof=1):.2f}")

    # T7 - book gross monotone table at the blended final
    print("\nT7  book gross (XGB T=3, blend 50/50, SPX 0.25):")
    for g in (0.6, 0.8, 1.0, 1.2):
        fmt_blend(f"gross {g:.1f}x", blend_stats(ls_t, ls_x, mkt, common, g=g))


if __name__ == "__main__":
    main()
