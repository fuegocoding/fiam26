"""Size-cap breakdown of the final v3 configuration (memo section 8).

Rebuilds the v3 book (both engines, T=3 staggered tranches, $1 price floor,
1.0x book gross + 0.25x SPX sleeve) and decomposes it by CRSP size group:

- share of book gross notional per group (long / short / combined)
- monthly return contribution per group (annualized pp) and the group
  basket's own annualized return
- one-way turnover share per group

Artifacts: output/repro/v3_size_breakdown.csv (monthly),
           output/repro/v3_returns_by_cap.png (equity curve + stacked
           cumulative contribution by size group).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "repro"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_lib import load_preds, staggered_sets, monthly_market
from xgb_baseline import apply_book_universe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUTDIR = OUT / "repro"
OUTDIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["mega", "large", "small", "micro", "nano"]
ENGINE_W = 0.25  # capital weight per engine leg (0.5x gross per engine)
LEG_N = 100


def form_eom(m):
    return (pd.Period(m, "M") - 1).to_timestamp(how="end").normalize()


def main():
    panel = pd.read_parquet(
        ROOT / "data" / "raw" / "chars_final_with_names.parquet",
        columns=["permno", "eom", "prc", "size_grp", "ret_exc_lead1m"],
    )
    panel["eom"] = pd.to_datetime(panel["eom"])
    ret_map = panel.set_index(["permno", "eom"])["ret_exc_lead1m"]

    tfm = load_preds().merge(panel[["permno", "eom", "prc"]],
                             on=["permno", "eom"], how="left", validate="many_to_one")
    xgb = pd.concat([
        pd.read_csv(OUT / "preds_xgb_timesfm_backcast.csv"),
        pd.read_csv(OUT / "preds_xgb_timesfm.csv"),
    ], ignore_index=True)
    xgb["target_month"] = pd.to_datetime(xgb["target_month"])
    xgb["eom"] = pd.to_datetime(xgb["eom"])
    xgb = xgb.rename(columns={"pred_xgb_tfm": "tfm_forecast"})
    xgb = xgb.merge(panel[["permno", "eom", "prc"]],
                    on=["permno", "eom"], how="left", validate="many_to_one")
    tfm = apply_book_universe(tfm)
    xgb = apply_book_universe(xgb)

    lt, st = staggered_sets(tfm, T=3)
    lx, sx = staggered_sets(xgb, T=3)
    # ret credit maps from the FLOORED frames (a held name without a current
    # floored row is zero-credited, matching book_returns exactly)
    ret_tfm = tfm.set_index(["permno", "eom"])["ret_exc_lead1m"]
    ret_xgb = xgb.set_index(["permno", "eom"])["ret_exc_lead1m"]
    # size attribution: last known panel row per permno (calendar-ffill)
    g2 = panel.assign(mm=panel.eom.dt.to_period("M")).pivot_table(
        index="permno", columns="mm", values="size_grp", aggfunc="last")
    g2 = g2.ffill(axis=1).bfill(axis=1)

    months = sorted(set(lt) & set(lx))

    rows = []
    prev_w = None
    for m in months:
        form = form_eom(m)
        fperiod = pd.Period(m, "M") - 1
        weights = {}
        for side, leg_sets in (("L", (lt, lx)), ("S", (st, sx))):
            for leg in leg_sets:
                names = leg.get(m, set())
                if not names:
                    continue
                sign = 1.0 if side == "L" else -1.0
                w = ENGINE_W / len(names)  # equal-weight within the tranche leg
                for p in names:
                    weights[p] = weights.get(p, 0.0) + sign * w
        contrib = {g: 0.0 for g in GROUPS}
        contrib["unknown"] = 0.0
        gross_w = {g: 0.0 for g in GROUPS + ["unknown"]}
        side_split = {g: [0.0, 0.0] for g in GROUPS + ["unknown"]}  # [long, short] gross weights

        def size_of(p):
            srow = g2.loc[p] if p in g2.index else None
            sg = srow.get(fperiod) if srow is not None else None
            return sg if (sg is not None and not pd.isna(sg)) else "unknown"

        # contributions computed per (engine, leg) with that engine's own ret
        # map - exactly mirrors book_returns' leg means (zero-credit missing)
        for side, leg, rmap in (("L", lt, ret_tfm), ("L", lx, ret_xgb),
                                ("S", st, ret_tfm), ("S", sx, ret_xgb)):
            names = leg.get(m, set())
            if not names:
                continue
            sign = 1.0 if side == "L" else -1.0
            w = ENGINE_W / len(names)
            for p in names:
                sg = size_of(p)
                r = rmap.get((p, form))
                r = float(r) if (r is not None and not pd.isna(r)) else 0.0
                contrib[sg] += sign * w * r
        for p, w in weights.items():
            sg = size_of(p)
            gross_w[sg] += abs(w)
            side_split[sg][0 if w > 0 else 1] += abs(w)
        row = {"target_month": str(m)}
        row.update({f"w_{g}": gross_w[g] for g in GROUPS + ["unknown"]})
        row.update({f"w_long_{g}": side_split[g][0] for g in GROUPS + ["unknown"]})
        row.update({f"w_short_{g}": side_split[g][1] for g in GROUPS + ["unknown"]})
        row.update({f"contrib_{g}": contrib[g] for g in GROUPS + ["unknown"]})
        if prev_w is not None:
            delta = {g: 0.0 for g in GROUPS + ["unknown"]}
            for p in set(weights) | set(prev_w):
                srow = g2.loc[p] if p in g2.index else None
                sg = srow.get(fperiod) if srow is not None else None
                sg = sg if (sg is not None and not pd.isna(sg)) else "unknown"
                delta[sg] += abs(weights.get(p, 0.0) - prev_w.get(p, 0.0))
            row.update({f"turn_{g}": delta[g] / 2.0 for g in GROUPS + ["unknown"]})
        else:
            row.update({f"turn_{g}": 0.0 for g in GROUPS + ["unknown"]})
        prev_w = weights
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "v3_size_breakdown.csv", index=False)

    # ---------------------------------------------------------------- summary
    sel = df[df.target_month >= "2018-01"]
    total_gross = sel[[f"w_{g}" for g in GROUPS + ["unknown"]]].sum(axis=1).mean()
    total_turn = sel[[f"turn_{g}" for g in GROUPS + ["unknown"]]].sum(axis=1).mean()
    print(f"months: {len(sel)} (2018-01..{sel.target_month.iloc[-1]}), "
          f"avg net book gross {total_gross:.1%}x")
    print(f"\n{'group':8s} {'gross%':>7s} {'long%':>6s} {'short%':>6s} "
          f"{'contrib/yr':>10s} {'basket/yr':>10s} {'turnover%':>9s}")
    for g in GROUPS + ["unknown"]:
        share = sel[f"w_{g}"].mean() / total_gross * 100
        lo = sel[f"w_long_{g}"].mean() / total_gross * 100
        sh = sel[f"w_short_{g}"].mean() / total_gross * 100
        c = sel[f"contrib_{g}"].mean() * 12 * 100
        w_g = sel[f"w_{g}"].mean()
        basket = sel[f"contrib_{g}"].mean() * 12 / w_g * 100 if w_g > 0 else np.nan
        to_share = sel[f"turn_{g}"].mean() / total_turn * 100
        print(f"{g:8s} {share:6.1f}% {lo:6.1f}% {sh:6.1f}% "
              f"{c:+10.2f}pp {basket:+9.2f}% {to_share:9.1f}%")

    # ------------------------------------------------- cap-group risk metrics
    def group_basket_stats(g):
        w = sel[f"w_{g}"].values
        c = sel[f"contrib_{g}"].values
        ok = w > 1e-9
        if ok.sum() < 24:
            return np.nan, np.nan, np.nan
        b = c[ok] / w[ok]
        sharpe = b.mean() / b.std(ddof=1) * np.sqrt(12) if b.std(ddof=1) > 0 else np.nan
        cum = np.cumprod(1.0 + b)
        mdd = float((cum / np.maximum.accumulate(cum) - 1.0).min())
        return sharpe, mdd, float(np.mean(1 - ok))

    print("\nper-cap risk metrics (book baskets, full window 2018-26):")
    print(f"{'group':8s} {'basket Sharpe':>13s} {'basket maxDD':>12s} "
          f"{'ICC tfm':>8s} {'ICC xgb':>8s}")
    for g in GROUPS:
        sharpe, mdd, _ = group_basket_stats(g)
        print(f"{g:8s} {sharpe:+13.2f} {mdd:+12.1%}")

    # ICC by cap group: monthly cross-sectional spearman(forecast, target)
    # within each group, on the floored pools
    panel["eom"] = panel["eom"]
    def group_icc(preds, col):
        d = preds.merge(panel[["permno", "eom", "size_grp"]],
                        on=["permno", "eom"], how="left", validate="many_to_one")
        d = d.dropna(subset=["size_grp", "ret_exc_lead1m"])
        d["mm"] = d.eom.dt.to_period("M")
        ics = d.groupby(["mm", "size_grp"]).apply(
            lambda g: g[col].corr(g.ret_exc_lead1m, method="spearman")
            if len(g) >= 30 else np.nan, include_groups=False)
        return ics.groupby(level=1).mean()

    icc_t = group_icc(tfm, "tfm_forecast")
    icc_x = group_icc(xgb, "pred_xgb" if "pred_xgb" in xgb.columns else "tfm_forecast")
    print(f"\nICC by cap group (monthly spearman, mean):")
    for g in GROUPS:
        print(f"{g:8s} tfm {icc_t.get(g, np.nan):+.3f}  xgb {icc_x.get(g, np.nan):+.3f}")

    # persist summary
    rows = []
    for g in GROUPS:
        sharpe, mdd, _ = group_basket_stats(g)
        rows.append({
            "group": g,
            "gross_share": sel[f"w_{g}"].mean() / total_gross,
            "long_share": sel[f"w_long_{g}"].mean() / total_gross,
            "short_share": sel[f"w_short_{g}"].mean() / total_gross,
            "contrib_ann_pp": sel[f"contrib_{g}"].mean() * 12 * 100,
            "basket_ann": sel[f"contrib_{g}"].mean() * 12 / sel[f"w_{g}"].mean()
            if sel[f"w_{g}"].mean() > 0 else np.nan,
            "basket_sharpe": sharpe,
            "basket_maxdd": mdd,
            "turnover_share": sel[f"turn_{g}"].mean() / total_turn,
            "icc_tfm": icc_t.get(g, np.nan),
            "icc_xgb": icc_x.get(g, np.nan),
        })
    pd.DataFrame(rows).to_csv(OUTDIR / "v3_size_cap_metrics.csv", index=False)
    print(f"saved: {OUTDIR / 'v3_size_cap_metrics.csv'}")

    # ---------------------------------------------------------------- chart
    mkt = monthly_market()
    idx = pd.PeriodIndex(sel.target_month, freq="M")
    ls_blend = sel[[f"contrib_{g}" for g in GROUPS + ["unknown"]]].sum(axis=1).values
    sleeve = (0.25 * mkt.spx_ret.reindex(idx)).values
    bench = mkt.benchmark.reindex(idx).values
    spx = mkt.spx_ret.reindex(idx).values
    xs = [p.to_timestamp() for p in idx]
    total = np.cumprod(1.0 + ls_blend + sleeve)
    cum_bench = np.cumprod(1.0 + bench)
    cum_spx = np.cumprod(1.0 + spx)
    cum_contrib = np.cumsum(np.column_stack(
        [sel[f"contrib_{g}"].values for g in GROUPS]), axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})
    ax1.plot(xs, total, lw=2.6, color="#1f77b4",
             label=f"FINAL v3 (cum {total[-1]:.2f}x)")
    ax1.plot(xs, cum_bench, lw=1.3, color="#2ca02c", ls=":",
             label=f"Benchmark TB3MS+4% (cum {cum_bench[-1]:.2f}x)")
    ax1.plot(xs, cum_spx, lw=1.3, color="#ff7f0e",
             label=f"S&P 500 (cum {cum_spx[-1]:.2f}x)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title("FINAL v3 (uniform T=3 tranches, 1.0x gross, $1 floor, +0.25x SPX) - 2018-2026")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.stackplot(xs, *cum_contrib.T, labels=GROUPS,
                  colors=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"],
                  alpha=0.85)
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_ylabel("Cumulative book contribution (not compounded)")
    ax2.legend(loc="upper left", fontsize=9, ncol=5)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = OUTDIR / "v3_returns_by_cap.png"
    fig.savefig(path, dpi=150)
    print(f"\nsaved: {OUTDIR / 'v3_size_breakdown.csv'}\nsaved: {path}")


if __name__ == "__main__":
    main()
