"""S&P 500 replication from the panel's large/mega-cap universe.

Motivation: the final v2 competition config holds a 0.25x SPX sleeve
(repro_03_final_config.py). FRED SPX is a price index we cannot literally
hold; this experiment replaces it with a real portfolio of panel
large/mega-cap stocks so the submission can say "we hold this book" and the
sleeve becomes auditable (names, weights, turnover, position count).

Replication variants. Formation row eom=t (attributes as of end of month t,
tradeable at start of t+1) earns row (permno, eom=t+1) `ret`:

  top500_cw   top-500 by lagged me, cap-weighted   (primary SPX replica)
  top500_eqw  top-500 equal-weighted               (control: caps matter)
  top200_cw   top-200 by lagged me, cap-weighted   (position-count headroom)
  top100_cw   top-100 by lagged me, cap-weighted   (concentration check)
  full_cw     all large+mega cap-weighted          (Russell-1000-ish)

Tracking metrics vs FRED SPX monthly. Known systematic gaps, expected and
not bugs: SPX is PRICE return while panel `ret` is TOTAL return (expected
+dividend-yield drift ~+1.5%/yr for the replica), and all 2026 panel rows
switch to Compustat return sourcing (source_crsp=0). No leakage: weights
come from month-t rows, returns from month t+1.

Swap test: the repro_03 final v2 with the 0.25x SPX sleeve replaced by
0.25x top500_cw, same legs and benchmark, like-for-like metrics, plus the
combined position count against the 100-500 rule.

Artifacts: output/spx_replication/{replica_monthly.csv, tracking_metrics.csv,
swap_monthly.csv, spx_replication.png}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "repro"))
from repro_lib import book_returns, load_preds, monthly_market, staggered_sets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUTDIR = OUT / "spx_replication"
OUTDIR.mkdir(parents=True, exist_ok=True)

SPECS = {
    "top500_cw": dict(top_n=500, equal_weight=False),
    "top500_eqw": dict(top_n=500, equal_weight=True),
    "top200_cw": dict(top_n=200, equal_weight=False),
    "top100_cw": dict(top_n=100, equal_weight=False),
    "full_cw": dict(top_n=None, equal_weight=False),
}


def load_panel():
    c = pd.read_parquet(
        ROOT / "data" / "raw" / "chars_final_with_names.parquet",
        columns=["permno", "eom", "ret", "me", "size_grp"],
    )
    c["eom"] = pd.to_datetime(c["eom"])
    c = c[c["size_grp"].isin(["large", "mega"]) & c["me"].notna() & c["ret"].notna()]
    return c.sort_values(["eom", "me"], ascending=[True, False]).reset_index(drop=True)


def build_weights(panel, top_n=None, equal_weight=False):
    """Formation weights per eom for one variant; dict eom -> (permnos, weights)."""
    out = {}
    for eom, g in panel.groupby("eom"):
        g = g.drop_duplicates("permno")
        if top_n is not None:
            g = g.head(top_n)
        w = np.full(len(g), 1.0 / len(g)) if equal_weight else g["me"].to_numpy()
        w = w / w.sum()
        out[eom] = (g["permno"].to_numpy(), w.astype(np.float64))
    return out


def month_returns(weights, ret_map):
    """Monthly replica return: weights at eom=t applied to `ret` at eom=t+1."""
    rows = []
    for t, (permnos, w) in sorted(weights.items()):
        m = (t + pd.offsets.MonthEnd(1)).to_period("M")
        r = ret_map.reindex(pd.MultiIndex.from_arrays([permnos, [m] * len(permnos)]))
        missing = int(r.isna().sum())
        r = r.fillna(0.0)  # month-t+1 row absent: 0 contribution (rare; reported)
        rows.append({"holding_month": m, "ret": float(np.dot(w, r.to_numpy())),
                     "n_names": len(permnos), "missing": missing})
    return pd.DataFrame(rows).set_index("holding_month")


def one_way_turnover(weights, ret_map):
    """Standard one-way turnover: mean over months of sum(max(0, w_new - w_drift))."""
    vals = []
    months = sorted(weights)
    for prev, cur in zip(months, months[1:]):
        p_perm, p_w = weights[prev]
        c_perm, c_w = weights[cur]
        m = (prev + pd.offsets.MonthEnd(1)).to_period("M")
        r = ret_map.reindex(pd.MultiIndex.from_arrays([p_perm, [m] * len(p_perm)]))
        r = r.fillna(0.0).to_numpy()
        port = float(np.dot(p_w, r))
        w_drift = p_w * (1 + r) / (1 + port)
        s_prev = pd.Series(w_drift, index=p_perm)
        s_cur = pd.Series(c_w, index=c_perm)
        delta = s_cur.sub(s_prev, fill_value=0.0)
        vals.append(float(delta.clip(lower=0.0).sum()))
    return float(np.mean(vals)) if vals else np.nan


def track_metrics(rep, spx, label):
    df = pd.DataFrame({"rep": rep, "spx": spx}).dropna()
    d = df["rep"] - df["spx"]
    beta = float(np.cov(df["rep"], df["spx"], ddof=1)[0, 1] / np.var(df["spx"], ddof=1))
    corr = float(np.corrcoef(df["rep"], df["spx"])[0, 1])
    return {
        "variant": label,
        "months": len(df),
        "corr": corr,
        "beta": beta,
        "r2": corr ** 2,
        "te_m": float(d.std(ddof=1)),
        "te_ann": float(d.std(ddof=1) * np.sqrt(12)),
        "bias_m": float(d.mean()),
        "bias_ann": float(d.mean() * 12),
        "max_abs_diff": float(d.abs().max()),
        "rep_ann": float(df["rep"].mean() * 12),
        "spx_ann": float(df["spx"].mean() * 12),
    }


def build_swap():
    """Rebuild the repro_03 v2 engine legs (TimesFM buffered + XGB T=3 blend)."""
    from repro_03_final_config import engine_legs, load_xgb_full

    tfm = load_preds()
    xgb = load_xgb_full()
    (rl_t, rs_t), lt, st = engine_legs(tfm, buffer_k=1500)
    lx, sx = staggered_sets(xgb, T=3)
    port_x = book_returns(xgb, lx, sx)
    idx_x = pd.PeriodIndex(port_x["target_month"], freq="M")
    rl_x = pd.Series(port_x["ret_long"].values, index=idx_x)
    rs_x = pd.Series(port_x["ret_short"].values, index=idx_x)
    def to_period_keys(sets):
        return {pd.Period(k, "M"): v for k, v in sets.items()}
    return ((rl_t - rs_t), (rl_x - rs_x),
            to_period_keys(lt), to_period_keys(st),
            to_period_keys(lx), to_period_keys(sx))


def main():
    panel = load_panel()
    ret_map = panel.set_index(["permno", panel["eom"].dt.to_period("M")])["ret"]
    print(f"panel large+mega rows: {len(panel):,} "
          f"({panel['eom'].min().date()}..{panel['eom'].max().date()})\n")

    mkt = monthly_market()
    spx = mkt["spx_ret"]
    monthly, tracking = {}, []
    print(f"{'variant':10s} {'n/mo':>6s} {'to/mo':>6s} {'top10w':>7s} | "
          f"corr   beta    R2   TEann  biasAnn  repAnn  spxAnn")
    for name, spec in SPECS.items():
        w = build_weights(panel, **spec)
        rep = month_returns(w, ret_map)
        monthly[name] = rep
        to = one_way_turnover(w, ret_map)
        top10 = np.mean([wm[:10].sum() if not spec["equal_weight"] else 10 / len(wm)
                         for _, wm in w.values()])
        for win, lo in (("full", "2016-10"), ("oos", "2021-01")):
            idx = rep.index >= pd.Period(lo)
            m = track_metrics(rep["ret"][idx], spx.reindex(rep.index[idx]), name)
            m["window"] = win
            tracking.append(m)
        mf = next(r for r in tracking if r["variant"] == name and r["window"] == "full")
        print(f"{name:10s} {rep['n_names'].mean():6.0f} {to:6.1%} {top10:7.1%} | "
              f"{mf['corr']:.3f} {mf['beta']:+.3f} {mf['r2']:.3f}  "
              f"{mf['te_ann']:5.1%}  {mf['bias_ann']:+5.2%}  {mf['rep_ann']:+5.2%}  {mf['spx_ann']:+5.2%}")
        miss = rep["missing"].sum()
        if miss:
            print(f"{'':10s} note: {miss} holding months with a missing return row")

    pd.DataFrame(tracking)[["variant", "window", "months", "corr", "beta", "r2",
                            "te_m", "te_ann", "bias_m", "bias_ann", "max_abs_diff",
                            "rep_ann", "spx_ann"]].to_csv(
        OUTDIR / "tracking_metrics.csv", index=False)
    series = {k: v["ret"] for k, v in monthly.items()}
    series["spx_ret"] = spx.reindex(monthly["top500_cw"].index)
    pd.DataFrame(series).to_csv(OUTDIR / "replica_monthly.csv")
    print()

    # ------------------------------------------------------------------ swap
    ls_t, ls_x, lt, st, lx, sx = build_swap()
    mkt_all = mkt
    common = ls_t.index.intersection(ls_x.index).intersection(mkt_all.index)
    common = common[common >= pd.Period("2018-01")]
    ls_blend = 0.5 * (ls_t.reindex(common) + ls_x.reindex(common))
    spx_c = mkt_all["spx_ret"].reindex(common)
    rep_c = monthly["top500_cw"]["ret"].reindex(common)
    bench = mkt_all["benchmark"].reindex(common)
    mkt_rf = mkt_all["mkt_rf"].reindex(common)

    from repro_03_final_config import fmt, metrics

    ret_spx = 0.4 * ls_blend + 0.25 * spx_c
    ret_rep = 0.4 * ls_blend + 0.25 * rep_c
    print("--- final v2 swap: 0.25x SPX sleeve -> 0.25x top500_cw replica (2018-2026) ---")
    fmt("v2 0.25x SPX (committed)", metrics(ret_spx, bench, mkt_rf))
    fmt("v2 0.25x replica        ", metrics(ret_rep, bench, mkt_rf))
    for label, ret in (("SPX", ret_spx), ("replica", ret_rep)):
        oos = ret[ret.index.year >= 2021]
        mo = metrics(oos, bench.reindex(oos.index), mkt_rf.reindex(oos.index))
        print(f"    {label:7s} OOS 2021-26: ann {mo['ann']:+.1%} CAGR {mo['cagr']:+.1%} "
              f"IR {mo['ir']:.2f} mdd {mo['maxdd']:.1%} beta {mo['beta']:+.2f}")
    print()

    # combined position count: blend book union + replica sleeve names vs 500 cap
    sleeve_names = {(t + pd.offsets.MonthEnd(1)).to_period("M"): set(p)
                    for t, (p, _) in build_weights(panel, top_n=500).items()}
    counts = []
    for m in common:
        blend = (lt.get(m, set()) | lx.get(m, set()) | st.get(m, set()) | sx.get(m, set()))
        sleeve = sleeve_names.get(m, set())
        counts.append((len(blend), len(sleeve), len(blend | sleeve)))
    cnt = pd.DataFrame(counts, index=common, columns=["blend", "sleeve", "combined"])
    print(f"combined positions/mo: blend {cnt['blend'].mean():.0f} + sleeve "
          f"{cnt['sleeve'].mean():.0f} -> union {cnt['combined'].mean():.0f} "
          f"(min {cnt['combined'].min()}, max {cnt['combined'].max()}; limit 500)")

    pd.DataFrame({"ret_v2_spx": ret_spx, "ret_v2_replica": ret_rep,
                  "ls_blend": ls_blend, "spx_ret": spx_c, "replica_ret": rep_c,
                  "benchmark": bench, "mkt_rf": mkt_rf}).to_csv(
        OUTDIR / "swap_monthly.csv")
    make_chart(series)


def make_chart(series):
    x = [p.to_timestamp() for p in series["top500_cw"].index]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, color in (("top500_cw", "#1f77b4"), ("top200_cw", "#9467bd"),
                        ("full_cw", "#8c564b")):
        cum = (1 + series[name]).cumprod()
        ax1.plot(x, cum, color=color, lw=1.6, label=f"{name} replica ({cum.iloc[-1]:.2f}x)")
    spx = (1 + series["spx_ret"]).cumprod()
    ax1.plot(x, spx, color="#ff7f0e", lw=2.0,
             label=f"FRED SPX price index ({spx.iloc[-1]:.2f}x)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("S&P 500 replication from panel large/mega caps vs FRED SPX, "
                  "2016-10..2026-08")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    d = series["top500_cw"] - series["spx_ret"]
    ax2.bar(x, d * 100, color=np.where(d >= 0, "#2ca02c", "#d62728"), width=20)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("top500_cw - SPX (pp/month)")
    ax2.grid(alpha=0.3)
    years = list(series["top500_cw"].index.year)
    ticks = [i for i in range(len(years)) if i == 0 or years[i] != years[i - 1]]
    ax2.set_xticks([x[i] for i in ticks])
    ax2.set_xticklabels([str(years[i]) for i in ticks])
    fig.tight_layout()
    path = OUTDIR / "spx_replication.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
