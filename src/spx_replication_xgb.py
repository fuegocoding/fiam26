"""XGBoost-tilted S&P 500 replica sleeve under the 500-position cap.

Extends src/spx_replication.py: the final v2 config's 0.25x SPX sleeve
becomes a real large/mega-cap portfolio, but the combined submission must
respect the 100-500 position rule. The blend book holds ~319 names/month,
so the sleeve is rebuilt on a ~150-name buffered universe and tilted with
the existing XGBoost forecasts (preds_xgb_timesfm[_backcast].csv) - no new
model training, same deployed signals as the committed config.

Design:
  - Selection with the TimesFM rank-buffer (hysteresis) applied to CAP rank:
    a name enters when its cap rank reaches the top `top_n`; it is retained
    until its rank falls beyond `buffer_mult * top_n`; vacancies are filled
    with the best-ranked non-holders. Low turnover like the buffered book.
  - Weights: cap weight x exp(lambda * z), z = within-sleeve cross-sectional
    z-score of the XGB forecast for the holding month (missing -> 0). This
    keeps the book an index replica with a modest, interpretable tilt.
  - No leakage: forecasts used at formation t carry target_month = t+1 and
    come from the challenge's own expanding-window fold schedule.

Variants: lambda in {0, 0.25, 0.5, 1.0} on the same buffered sleeve, plus
the union position count vs the 500 cap in the swap test.

Artifacts: output/spx_replication_xgb/{sleeve_monthly.csv, sleeve_metrics.csv,
spx_replication_xgb.png}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spx_replication import (ROOT, load_panel, month_returns, one_way_turnover,
                             track_metrics)

OUTDIR = ROOT / "output" / "spx_replication_xgb"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOP_N = 140
BUFFER_MULT = 2.0
LAMBDAS = (0.0, 0.25, 0.5, 1.0)
Z_CAP = 2.0


def load_xgb_map():
    back = pd.read_csv(ROOT / "output" / "preds_xgb_timesfm_backcast.csv")
    oos = pd.read_csv(ROOT / "output" / "preds_xgb_timesfm.csv")
    x = pd.concat([back, oos], ignore_index=True)
    x["target_month"] = pd.to_datetime(x["target_month"])
    x = x.rename(columns={"pred_xgb_tfm": "pred"})
    return {pd.Period(m, "M"): s.set_index("permno")["pred"]
            for m, s in x.groupby("target_month")}


def buffered_universe(panel, top_n=TOP_N, buffer_mult=BUFFER_MULT):
    """Hysteresis selection on cap rank; returns dict eom -> ordered permnos."""
    held = []
    out = {}
    for eom, g in panel.groupby("eom"):
        g = g.drop_duplicates("permno")
        ranks = pd.Series(np.arange(1, len(g) + 1),
                          index=g["permno"].to_numpy())
        limit = int(buffer_mult * top_n)
        survivors = [p for p in held if ranks.get(p, limit + 1) <= limit]
        non_holders = [p for p in g["permno"] if p not in set(survivors)]
        held = survivors + non_holders[: max(0, top_n - len(survivors))]
        out[eom] = held
    return out


def tilted_weights(panel, sleeve, lam, xgb_map):
    """Cap weight x exp(lam * z(forecast)) within the held sleeve, normalized."""
    me_map = panel.set_index(["permno", panel["eom"].dt.to_period("M")])["me"]
    out = {}
    for t, permnos in sleeve.items():
        m = (t + pd.offsets.MonthEnd(1)).to_period("M")
        me = me_map.reindex(pd.MultiIndex.from_arrays(
            [permnos, [t.to_period("M")] * len(permnos)]))
        preds = xgb_map.get(m)
        if preds is not None:
            f = preds.reindex(permnos).to_numpy()
        else:
            f = np.full(len(permnos), np.nan)
        if np.isfinite(f).any():
            z = np.where(np.isfinite(f), f - np.nanmean(f), 0.0)
            sd = float(np.nanstd(f))
            z = np.clip(np.where(sd > 0, z / sd, 0.0), -Z_CAP, Z_CAP)
        else:
            z = np.zeros(len(f))
        w = me.to_numpy() * np.exp(lam * z)
        w = w / w.sum()
        out[t] = (np.asarray(permnos), w)
    return out


def main():
    panel = load_panel()
    xgb_map = load_xgb_map()
    sleeve = buffered_universe(panel)
    print(f"buffered sleeve: top_n {TOP_N}, buffer {BUFFER_MULT}x -> "
          f"{np.mean([len(v) for v in sleeve.values()]):.0f} names/mo")

    mkt = None
    sys.path.insert(0, str(Path(__file__).resolve().parent / "repro"))
    from repro_lib import monthly_market
    mkt = monthly_market()
    spx = mkt["spx_ret"]

    sleeves, metrics_rows = {}, []
    for lam in LAMBDAS:
        w = tilted_weights(panel, sleeve, lam, xgb_map)
        rep = month_returns(w, ret_map=panel.set_index(
            ["permno", panel["eom"].dt.to_period("M")])["ret"])
        name = f"top{TOP_N}_buf_lam{lam}"
        sleeves[name] = rep
        to = one_way_turnover(w, panel.set_index(
            ["permno", panel["eom"].dt.to_period("M")])["ret"])
        for win, lo in (("full", "2018-01"), ("oos", "2021-01")):
            idx = rep.index >= pd.Period(lo)
            m = track_metrics(rep["ret"][idx], spx.reindex(rep.index[idx]), name)
            m.update(window=win, turnover=to, lam=lam, names=np.mean([len(v) for v in sleeve.values()]))
            metrics_rows.append(m)
        mf = next(r for r in metrics_rows if r["variant"] == name and r["window"] == "full")
        sr = rep["ret"].mean() / rep["ret"].std(ddof=1) * np.sqrt(12)
        print(f"{name:18s} corr {mf['corr']:.3f} beta {mf['beta']:+.3f} R2 {mf['r2']:.3f} "
              f"TE {mf['te_ann']:4.1%} bias {mf['bias_ann']:+5.2%} ann {mf['rep_ann']:+5.2%} "
              f"Sh {sr:4.2f} to {to:4.1%}/mo")

    pd.DataFrame(metrics_rows)[["variant", "window", "months", "corr", "beta", "r2",
                                "te_m", "te_ann", "bias_m", "bias_ann",
                                "max_abs_diff", "rep_ann", "spx_ann",
                                "turnover", "lam", "names"]].to_csv(
        OUTDIR / "sleeve_metrics.csv", index=False)
    pd.DataFrame({k: v["ret"] for k, v in sleeves.items()} |
                 {"spx_ret": spx.reindex(next(iter(sleeves.values())).index)}).to_csv(
        OUTDIR / "sleeve_monthly.csv")
    print()

    # ------------------------------------------------------------------ swap
    from spx_replication import build_swap
    from repro_03_final_config import fmt, metrics

    ls_t, ls_x, lt, st, lx, sx = build_swap()
    common = ls_t.index.intersection(ls_x.index).intersection(mkt.index)
    common = common[common >= pd.Period("2018-01")]
    ls_blend = 0.5 * (ls_t.reindex(common) + ls_x.reindex(common))
    spx_c = mkt["spx_ret"].reindex(common)
    bench = mkt["benchmark"].reindex(common)
    mkt_rf = mkt["mkt_rf"].reindex(common)

    print("--- final v2 swap: 0.25x SPX -> 0.25x tilted replica sleeve (2018-2026) ---")
    swaps = {}
    for name, rep in sleeves.items():
        ret = 0.4 * ls_blend + 0.25 * rep["ret"].reindex(common)
        swaps[name] = ret
        fmt(name, metrics(ret, bench, mkt_rf))
        oos = ret[ret.index.year >= 2021]
        mo = metrics(oos, bench.reindex(oos.index), mkt_rf.reindex(oos.index))
        print(f"    OOS 2021-26: ann {mo['ann']:+.1%} CAGR {mo['cagr']:+.1%} "
              f"IR {mo['ir']:.2f} mdd {mo['maxdd']:.1%} beta {mo['beta']:+.2f}")
    fmt("v2 0.25x SPX (committed)",
        metrics(0.4 * ls_blend + 0.25 * spx_c, bench, mkt_rf))
    print()

    # union positions: blend book + sleeve names, vs 500 cap
    sleeve_names = {(t + pd.offsets.MonthEnd(1)).to_period("M"): set(v)
                    for t, v in sleeve.items()}
    counts = []
    for m in common:
        blend = (lt.get(m, set()) | lx.get(m, set()) | st.get(m, set()) | sx.get(m, set()))
        s = sleeve_names.get(m, set())
        counts.append((len(blend), len(s), len(blend | s)))
    cnt = pd.DataFrame(counts, index=common, columns=["blend", "sleeve", "combined"])
    print(f"union positions/mo: blend {cnt['blend'].mean():.0f} + sleeve "
          f"{cnt['sleeve'].mean():.0f} -> {cnt['combined'].mean():.0f} "
          f"(min {cnt['combined'].min()}, max {cnt['combined'].max()}; limit 500)")

    pd.DataFrame({"ret_v2_spx": 0.4 * ls_blend + 0.25 * spx_c,
                  **{name: rep["ret"].reindex(common) for name, rep in sleeves.items()},
                  "ls_blend": ls_blend, "spx_ret": spx_c,
                  "benchmark": bench, "mkt_rf": mkt_rf}).to_csv(
        OUTDIR / "swap_monthly.csv")
    make_chart(sleeves, spx_c, common)


def make_chart(sleeves, spx, common):
    x = [p.to_timestamp() for p in common]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    base = sleeves[f"top{TOP_N}_buf_lam0.0"]["ret"].reindex(common)
    for name, color in ((f"top{TOP_N}_buf_lam0.0", "#1f77b4"),
                        (f"top{TOP_N}_buf_lam0.5", "#9467bd"),
                        (f"top{TOP_N}_buf_lam1.0", "#8c564b")):
        cum = (1 + sleeves[name]["ret"].reindex(common)).cumprod()
        ax1.plot(x, cum, color=color, lw=1.6, label=f"{name} ({cum.iloc[-1]:.2f}x)")
    ax1.plot(x, (1 + spx).cumprod(), color="#ff7f0e", lw=2.0, label="FRED SPX (price)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title(f"XGBoost-tilted SPX replica sleeve (top {TOP_N}, buffered), 2018-2026")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    d = sleeves[f"top{TOP_N}_buf_lam0.5"]["ret"].reindex(common) - base
    ax2.bar(x, d * 100, color=np.where(d >= 0, "#2ca02c", "#d62728"), width=20)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("tilt lam0.5 - pure replica (pp/mo)")
    ax2.grid(alpha=0.3)
    years = list(common.year)
    ticks = [i for i in range(len(years)) if i == 0 or years[i] != years[i - 1]]
    ax2.set_xticks([x[i] for i in ticks])
    ax2.set_xticklabels([str(years[i]) for i in ticks])
    fig.tight_layout()
    path = OUTDIR / "spx_replication_xgb.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
