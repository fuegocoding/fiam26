"""REPRO SCRIPT 3/3 - memo section 8: final competition configuration.

FINAL CONFIG (v3, blend, uniform tranches; adopted 2026-09-21):
  - Alpha book 0.8x gross, BOTH engines in 3 staggered tranches (uniform
    turnover control, no rank-buffer parameter): TimesFM buffered book replaced
    by the T=3 tranche book (each tranche reconstituted every 3rd month from
    raw ranks; every name held ~3 months; ~15%/mo NAV turnover) at 0.4x gross
    + XGBoost+TimesFM raw book at 0.4x gross, same T=3 cadence
  - S&P 500 sleeve at 0.25x (structural beta anchor)
  - NO ratchet (ablation shows it is inert for the blend; static exposure)
  - Window 2018-01 .. 2026-08 (104 months). The XGBoost sleeve uses the
    competition expanding-window folds for 2021-26 (output/preds_xgb_timesfm.csv)
    plus the BACKCAST folds 2018-20 (src/xgb_timesfm_backcast.py ->
    output/preds_xgb_timesfm_backcast.csv: 2018 thin-train, 2019-20 quasi-OOS).
    2016-17 cannot be backcast for XGBoost (panel data starts 2015); the
    2016-10..2017-12 segment is covered by the TimesFM-only narrative in
    memo sections 2-5.

The TFM rank buffer (k=1500) was the pre-registered v2 mechanism; the uniform
T=3 treatment was adopted on the a-priori simplicity criterion (one cadence
rule, no tuned buffer parameter) and is performance-equivalent in the fixed
pipeline (memo section 8 sensitivity T4). The v2 buffer variant remains
disclosed as the rejected alternative.

SUPERSEDED CONFIG (v1, single-engine): TimesFM buffered book 1x gross + 0.5x
SPX (price return) + ratchet, 2016-10..2026-08. Printed for continuity.

Artifacts: output/repro/final_config_monthly.csv, timesfm_final_equity_curve_repro.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_lib import (book_returns, load_preds, max_drawdown, monthly_market,
                       monthly_sets, ratchet_scales, staggered_sets,
                       turnover_per_leg, yearly_table)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
REPRO_OUT = OUT / "repro"
REPRO_OUT.mkdir(parents=True, exist_ok=True)

MEMO_V2 = {"ann": 0.144, "cagr": 0.148, "sharpe": 1.49, "ir": 0.80, "maxdd": -0.124,
           "beta": 0.18, "beta_oos": 0.27}


def load_xgb_full():
    back = pd.read_csv(OUT / "preds_xgb_timesfm_backcast.csv")
    oos = pd.read_csv(OUT / "preds_xgb_timesfm.csv")
    xgb = pd.concat([back, oos], ignore_index=True)
    xgb["target_month"] = pd.to_datetime(xgb["target_month"])
    return xgb.rename(columns={"pred_xgb_tfm": "tfm_forecast"})


def engine_legs(preds, buffer_k):
    long_sets, short_sets = monthly_sets(preds, buffer_k=buffer_k)
    port = book_returns(preds, long_sets, short_sets)
    idx = pd.PeriodIndex(port["target_month"], freq="M")
    legs = (pd.Series(port["ret_long"].values, index=idx),
            pd.Series(port["ret_short"].values, index=idx))
    return legs, long_sets, short_sets


def beta_with_se(port_ret, mkt):
    df = pd.DataFrame({"port": port_ret, "mkt": mkt}).dropna()
    x, y = df["mkt"].values, df["port"].values
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    resid = y - beta * x
    n = len(x)
    se = float(np.sqrt(np.sum(resid ** 2) / (n - 2) / np.sum((x - x.mean()) ** 2)))
    return beta, se


def metrics(ret, bench, mkt_rf):
    active = ret - bench
    yrs = len(ret) / 12
    oos = ret.index.year >= 2021
    return {
        "ann": ret.mean() * 12,
        "cagr": ((1 + ret).prod()) ** (1 / yrs) - 1,
        "sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(12),
        "ir": np.sqrt(12) * active.mean() / active.std(ddof=1),
        "maxdd": max_drawdown(ret),
        "maxdd_oos": max_drawdown(ret[oos]),
        "beta": beta_with_se(ret, mkt_rf)[0],
        "beta_se": beta_with_se(ret, mkt_rf)[1],
        "beta_oos": beta_with_se(ret[oos], mkt_rf[oos])[0],
        "hit": (ret > 0).mean(),
        "deep": int((ret < -0.05).sum()),
        "worst": f"{ret.idxmin()} {ret.min():+.1%}",
        "cum": float((1 + ret).prod()),
    }


def fmt(label, m):
    print(f"{label:26s} ann {m['ann']:+6.1%} CAGR {m['cagr']:+6.1%} Sh {m['sharpe']:.2f} "
          f"IR {m['ir']:5.2f} | mdd {m['maxdd']:6.1%} (OOS {m['maxdd_oos']:5.1%}) | "
          f"beta {m['beta']:+.2f} (OOS {m['beta_oos']:+.2f}) | hit {m['hit']:.0%} "
          f"<-5% {m['deep']:2d} | worst {m['worst']} | cum {m['cum']:.1f}x")


def main():
    tfm = load_preds()
    xgb = load_xgb_full()

    lt, st = staggered_sets(tfm, T=3)                          # TimesFM tranches (uniform cadence)
    port_t = book_returns(tfm, lt, st)
    idx_t = pd.PeriodIndex(port_t["target_month"], freq="M")
    rl_t = pd.Series(port_t["ret_long"].values, index=idx_t)
    rs_t = pd.Series(port_t["ret_short"].values, index=idx_t)
    lx, sx = staggered_sets(xgb, T=3)                          # XGB raw, same 3 staggered tranches
    port_x = book_returns(xgb, lx, sx)
    idx_x = pd.PeriodIndex(port_x["target_month"], freq="M")
    rl_x = pd.Series(port_x["ret_long"].values, index=idx_x)
    rs_x = pd.Series(port_x["ret_short"].values, index=idx_x)
    ls_t = rl_t - rs_t
    ls_x = rl_x - rs_x

    mkt = monthly_market()
    ls_t_full = rl_t - rs_t   # native window 2016-02..2026-08 (for v1 continuity)
    common = ls_t.index.intersection(ls_x.index).intersection(mkt.index)
    common = common[common >= pd.Period("2018-01")]
    ls_t, ls_x = ls_t.reindex(common), ls_x.reindex(common)
    spx = mkt["spx_ret"].reindex(common)
    bench = mkt["benchmark"].reindex(common)
    mkt_rf = mkt["mkt_rf"].reindex(common)
    ls_blend = 0.5 * (ls_t + ls_x)
    print(f"window {common[0]}..{common[-1]}: {len(common)} months\n")

    # --- FINAL v3: blend 0.8x gross (0.4x per engine, both engines in 3 staggered tranches) + 0.25x SPX, no ratchet ---
    ret_v2 = 0.4 * ls_blend + 0.25 * spx
    m2 = metrics(ret_v2, bench, mkt_rf)
    print("--- FINAL v3: blend 0.8x gross, BOTH engines in T=3 staggered tranches + 0.25x SPX, no ratchet ---")
    fmt("full window 2018-2026", m2)
    print(f"    memo v2 targets  : ann {MEMO_V2['ann']:+.1%} CAGR {MEMO_V2['cagr']:+.1%} "
          f"Sh {MEMO_V2['sharpe']:.2f} IR {MEMO_V2['ir']:.2f} mdd {MEMO_V2['maxdd']:.1%} "
          f"beta {MEMO_V2['beta']:+.2f}/{MEMO_V2['beta_oos']:+.2f}")

    # --- SUPERSEDED v1: TimesFM-only 1x gross + 0.5x SPX + ratchet ---
    tfm_native = ls_t_full[ls_t_full.index >= pd.Period("2016-10")]
    mkt_nat = monthly_market().reindex(tfm_native.index)
    comb_v1 = 0.5 * tfm_native + 0.5 * mkt_nat["spx_ret"]
    sc_v1 = ratchet_scales(comb_v1)
    ret_v1 = sc_v1 * comb_v1
    m1 = metrics(ret_v1, mkt_nat["benchmark"], mkt_nat["mkt_rf"])
    print("\n--- SUPERSEDED v1: TFM-only 1x gross + 0.5x SPX + ratchet (native window) ---")
    fmt(f"{tfm_native.index[0]}..{tfm_native.index[-1]}", m1)

    # OOS sub-window rows (the graded window)
    for label, ret in (("v2", ret_v2), ("v1", ret_v1.reindex(common).dropna())):
        oos = ret[ret.index.year >= 2021]
        mo = metrics(oos, bench.reindex(oos.index), mkt_rf.reindex(oos.index))
        print(f"    {label} OOS 2021-26: ann {mo['ann']:+.1%} CAGR {mo['cagr']:+.1%} "
              f"IR {mo['ir']:.2f} mdd {mo['maxdd']:.1%} beta {mo['beta']:+.2f}")

    # same-window comparison for the memo
    m1_same = metrics(ret_v1.reindex(common).dropna(), bench, mkt_rf)
    print("\n--- like-for-like, both configs on 2018-2026 ---")
    fmt("v3 blend 0.8x + 0.25x SPX", m2)
    fmt("v1 TFM-only 1.0x + 0.5x SPX", m1_same)

    yt = yearly_table(ret_v2, ret_v2.index)
    print("\nyear | ann | Sharpe")
    for yr, row in yt.iterrows():
        print(f"{yr} | {row['ann_ret']:+6.1%} | {row['sharpe']:5.2f}")

    to_tl, to_ts = turnover_per_leg(lt, st)
    to_xl, to_xs = turnover_per_leg(lx, sx)
    total_names = [len(lt[m] | lx[m]) + len(st[m] | sx[m]) for m in sorted(set(lt) & set(lx))]
    overlap = [len(lt[m] & lx[m]) + len(st[m] & sx[m]) for m in sorted(set(lt) & set(lx))]
    print(f"\ncompliance: positions avg {np.mean(total_names):.0f} (min {min(total_names)}, max {max(total_names)}), "
          f"engine overlap {np.mean(overlap):.0f}")
    print(f"turnover/leg: TimesFM tranches {to_tl:.0%}/{to_ts:.0%}, XGB tranches {to_xl:.0%}/{to_xs:.0%}")
    print("exposures: gross 105% constant (0.8x book + 0.25x SPX, static), net +25%, monthly rebalance")

    out = pd.DataFrame({
        "ret_final_v2": ret_v2, "ls_tfm_buffered": ls_t, "ls_xgb_raw": ls_x,
        "spx_ret": spx, "benchmark": bench, "mkt_rf": mkt_rf,
    })
    out.to_csv(REPRO_OUT / "final_config_monthly.csv")
    print(f"\nsaved: {REPRO_OUT / 'final_config_monthly.csv'}")

    make_chart(ret_v1, ret_v2, ls_blend, mkt, common)


def make_chart(ret_v1, ret_v2, ls_blend, mkt, common):
    spx = mkt["spx_ret"].reindex(common)
    bench = mkt["benchmark"].reindex(common)
    raw = (1 + ls_blend).cumprod()
    v1 = (1 + ret_v1.reindex(common).dropna()).cumprod()
    v2 = (1 + ret_v2).cumprod()
    spxc = (1 + spx).cumprod()
    benc = (1 + bench).cumprod()

    x = [p.to_timestamp() for p in v2.index]

    def dd(cum):
        return (cum / cum.cummax() - 1) * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(x, v2, color="#1f77b4", lw=2.8,
             label=f"FINAL v3: blend 0.8x gross, uniform T=3 tranches + 0.25x SPX (Sh {metrics(ret_v2, mkt['benchmark'].reindex(common), mkt['mkt_rf'].reindex(common))['sharpe']:.2f}, maxDD {metrics(ret_v2, mkt['benchmark'].reindex(common), mkt['mkt_rf'].reindex(common))['maxdd']:.0%})")
    ax1.plot(x, v1, color="#7f7f7f", lw=1.4,
             label="Superseded v1: TFM-only 1x gross + 0.5x SPX + ratchet")
    ax1.plot(x, raw, color="#7f7f7f", lw=0.9, ls="--", label="Blend book 1x gross (no overlay)")
    ax1.plot(x, spxc, color="#ff7f0e", lw=1.4, label="S&P 500 (context)")
    ax1.plot(x, benc, color="#2ca02c", lw=1.2, ls=":", label="Benchmark: T-bill + 4%")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("TimesFM + XGBoost blend - final configuration, 2018-2026")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.fill_between(x, dd(v2), 0, color="#d62728", alpha=0.45)
    ax2.set_ylabel("Drawdown (%)")
    ax2.annotate(f"maxDD {metrics(ret_v2, mkt['benchmark'].reindex(common), mkt['mkt_rf'].reindex(common))['maxdd']:.0%}",
                 xy=(x[int(np.argmin(dd(v2).values))], dd(v2).min()),
                 xytext=(10, -10), textcoords="offset points",
                 color="#d62728", fontsize=9)
    ax2.grid(alpha=0.3)

    years = [p.year for p in v2.index]
    tick_idx = [i for i in range(len(years)) if i == 0 or years[i] != years[i - 1]]
    ax2.set_xticks([x[i] for i in tick_idx])
    ax2.set_xticklabels([years[i] for i in tick_idx])

    fig.tight_layout()
    path = REPRO_OUT / "timesfm_final_equity_curve_repro.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
