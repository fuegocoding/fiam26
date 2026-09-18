"""Sector x size breakdown of the current final v2 model.

Rebuilds the exact final competition configuration (repro_03_final_config.py)
for every GICS sector and every cap-size bucket (size_grp) within each sector,
plus the totals:

  - Engine A: TimesFM 3.0 buffered book (k=1500), top/bottom-100, monthly
  - Engine B: XGBoost+TimesFM raw book, 3 staggered tranches
  - Group book: 0.4x engine A L/S + 0.4x engine B L/S (alpha book, 0.8x gross;
    the 0.25x SPX beta sleeve is not attributable to sectors/sizes)

For each group the engines are rebuilt from scratch on the group-restricted
cross-section (self-contained attribution), then reported:

  CAGR, max drawdown, per-leg turnover (mean of long/short replacement rate),
  annualized OLS alpha vs market excess (mkt_rf), Sharpe, and ICC (mean monthly
  Spearman rank IC of predictions vs next-month excess returns, averaged over
  the two engines, computed within the group).

Window: 2018-01 .. 2026-08 (104 months, same as the final config).

Artifact: output/repro/sector_size_breakdown.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "repro"))
from repro_lib import book_returns, load_preds, monthly_market, turnover_per_leg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
REPRO_OUT = OUT / "repro"

GICS_NAMES = {
    "10": "Energy", "15": "Materials", "20": "Industrials",
    "25": "Consumer Discretionary", "30": "Consumer Staples",
    "35": "Health Care", "40": "Financials", "45": "Info Tech",
    "50": "Comm Services", "55": "Utilities", "60": "Real Estate",
}
SIZES = ["mega", "large", "small", "micro", "nano"]


def sets_scaled(preds, side_max=100, buffer_mult=15.0, T=1):
    """Monthly long/short sets mirroring the model rules, scaled to the group.

    Same mechanics as repro_lib.monthly_sets (rank buffer, hysteresis) and
    repro_lib.staggered_sets (T tranches), but the per-leg side is
    min(side_max, n // 2) each month so small cells keep a non-overlapping
    book; the rank buffer keeps the model's 15x-side ratio (1500/100).
    """
    months = sorted(preds["target_month"].unique())
    state = {c: (set(), set()) for c in range(T)}
    long_sets, short_sets = {}, {}
    prev_long, prev_short = [], []
    for i, month in enumerate(months):
        g = preds[preds["target_month"] == month].sort_values("tfm_forecast",
                                                              ascending=False)
        names = list(g["permno"])
        n = len(names)
        side = min(side_max, n // 2)
        if side < 3:
            continue
        pos = {p: j for j, p in enumerate(names)}
        c = i % T
        k_leg = side // T + (1 if c < side - (side // T) * T else 0)
        k_buf = int(buffer_mult * k_leg)
        if T == 1:
            keep_l = [p for p in prev_long if p in pos and pos[p] < k_buf]
            keep_s = [p for p in prev_short if p in pos and n - 1 - pos[p] < k_buf]
            keep_l = sorted(keep_l, key=lambda p: pos[p])[:k_leg]
            keep_s = sorted(keep_s, key=lambda p: pos[p])[-k_leg:]
            fills_l = [p for p in names if p not in keep_l]
            fills_s = [p for p in reversed(names) if p not in keep_s]
            state[0] = (set(keep_l + fills_l[:k_leg - len(keep_l)]),
                        set(keep_s + fills_s[:k_leg - len(keep_s)]))
        else:
            state[c] = (set(names[:k_leg]), set(names[n - k_leg:]))
        long_sets[month] = set().union(*[state[cc][0] for cc in range(T)])
        short_sets[month] = set().union(*[state[cc][1] for cc in range(T)])
        if T == 1:
            prev_long = sorted(long_sets[month], key=lambda p: pos[p])
            prev_short = sorted(short_sets[month], key=lambda p: pos[p])
    return long_sets, short_sets


def load_xgb_full():
    back = pd.read_csv(OUT / "preds_xgb_timesfm_backcast.csv")
    oos = pd.read_csv(OUT / "preds_xgb_timesfm.csv")
    xgb = pd.concat([back, oos], ignore_index=True)
    xgb["target_month"] = pd.to_datetime(xgb["target_month"])
    return xgb.rename(columns={"pred_xgb_tfm": "tfm_forecast"})


def load_attributes():
    cols = ["permno", "eom", "gics", "size_grp"]
    df = pd.read_parquet(ROOT / "data/raw/chars_final_with_names.parquet", columns=cols)
    df["eom"] = pd.to_datetime(df["eom"])
    df["target_month"] = df["eom"] + pd.offsets.MonthEnd(1)
    df["sector"] = df["gics"].astype(str).str[:2].map(GICS_NAMES)
    df.loc[df["gics"].isna(), "sector"] = "Unknown"
    return df[["permno", "target_month", "sector", "size_grp"]]


def leg_book(preds, long_sets, short_sets):
    """Equal-weight leg L/S returns per month (skips months with no names)."""
    ret_by_month = {m: g.set_index("permno")["ret_exc_lead1m"]
                    for m, g in preds.groupby("target_month")}
    rows = []
    for month in sorted(long_sets):
        r = ret_by_month[month]
        long_leg = r.reindex(sorted(long_sets[month])).dropna()
        short_leg = r.reindex(sorted(short_sets[month])).dropna()
        if len(long_leg) == 0 or len(short_leg) == 0:
            continue
        rows.append({"target_month": month,
                     "ret_ls": float(long_leg.mean() - short_leg.mean())})
    if not rows:
        return None
    port = pd.DataFrame(rows)
    idx = pd.PeriodIndex(port["target_month"], freq="M")
    return pd.Series(port["ret_ls"].values, index=idx)


def group_book(preds_t, preds_x):
    """0.8x-gross alpha book = 0.4x TFM-buffered + 0.4x XGB-staggered L/S.

    preds_t / preds_x are the group-restricted prediction frames of the two
    engines (TimesFM forecasts and XGBoost+TimesFM OOS predictions resp.),
    each with columns permno, target_month, tfm_forecast, ret_exc_lead1m.
    """
    if preds_t["permno"].nunique() < 10:
        return None, None
    lt, st = sets_scaled(preds_t, T=1)
    ls_t = leg_book(preds_t, lt, st)
    lx, sx = sets_scaled(preds_x, T=3)
    ls_x = leg_book(preds_x, lx, sx)
    if ls_t is None or ls_x is None:
        return None, None
    ls = 0.4 * (0.5 * ls_t + 0.5 * ls_x)
    to = np.nanmean(turnover_per_leg(lt, st) + turnover_per_leg(lx, sx))
    return ls, to


def rank_ic(preds, pred_col):
    """Mean monthly Spearman rank IC within the group, per engine."""
    ics = []
    for _, g in preds.groupby("target_month"):
        g = g.dropna(subset=[pred_col, "ret_exc_lead1m"])
        if len(g) >= 10:
            ics.append(g[pred_col].corr(g["ret_exc_lead1m"], method="spearman"))
    return float(np.mean(ics)) if ics else np.nan


def max_dd(ret):
    cum = (1.0 + ret).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def evaluate(ret, mkt_rf):
    if ret is None or ret.dropna().shape[0] < 12:
        return None
    ret = ret.dropna()
    df = pd.DataFrame({"y": ret, "x": mkt_rf.reindex(ret.index)}).dropna()
    xm = df["x"] - df["x"].mean()
    beta = float(np.dot(xm, df["y"] - df["y"].mean()) / np.dot(xm, xm))
    alpha = float(df["y"].mean() - beta * df["x"].mean()) * 12
    yrs = len(ret) / 12
    return {
        "cagr": float((1 + ret).prod() ** (1 / yrs) - 1),
        "dd": max_dd(ret),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(12)),
        "alpha": alpha,
    }


def main():
    tfm = load_preds()
    xgb = load_xgb_full()
    attrs = load_attributes()

    tfm = tfm.merge(attrs, on=["permno", "target_month"], how="left")
    xgb = xgb.merge(attrs, on=["permno", "target_month"], how="left")

    mkt = monthly_market()
    window = pd.PeriodIndex(pd.period_range("2018-01", "2026-08", freq="M"))
    mkt_rf = mkt["mkt_rf"].reindex(window)
    rows = []

    def run_group(label, sector, size):
        if sector is None:
            m = pd.Series(True, index=tfm.index)
        else:
            m = tfm["sector"] == sector
        if size is not None:
            m &= tfm["size_grp"] == size
        g_tfm = tfm[m]
        g_xgb = xgb[xgb["target_month"] <= pd.Timestamp("2026-08-31")]
        if sector is not None:
            g_xgb = g_xgb[g_xgb["sector"] == sector]
        if size is not None:
            g_xgb = g_xgb[g_xgb["size_grp"] == size]

        cols = ["permno", "target_month", "tfm_forecast", "ret_exc_lead1m"]
        ls, to = group_book(g_tfm[cols], g_xgb[cols])
        ls = ls.reindex(window) if ls is not None else None
        mets = evaluate(ls, mkt_rf)
        icc = np.nanmean([rank_ic(g_tfm, "tfm_forecast"),
                          rank_ic(g_xgb, "tfm_forecast")])
        n_names = g_tfm["permno"].nunique()
        rows.append({
            "sector": sector if sector else "ALL",
            "size": size if size else "all",
            "n_names": n_names,
            "cagr": mets["cagr"] if mets else np.nan,
            "dd": mets["dd"] if mets else np.nan,
            "turnover": to,
            "alpha": mets["alpha"] if mets else np.nan,
            "sharpe": mets["sharpe"] if mets else np.nan,
            "icc": icc,
        })

    # Complete model: all sectors, all sizes
    run_group(None, None, None)
    # Each size, complete universe
    for s in SIZES:
        run_group(None, None, s)
    # Each sector, all sizes + each size within sector
    for sec in list(GICS_NAMES.values()) + ["Unknown"]:
        run_group(None, sec, None)
        for s in SIZES:
            run_group(None, sec, s)

    res = pd.DataFrame(rows)
    res.to_csv(REPRO_OUT / "sector_size_breakdown.csv", index=False)

    pd.set_option("display.width", 200)
    view = res.copy()
    for c in ["cagr", "dd", "turnover", "alpha"]:
        view[c] = (view[c] * 100).round(1).astype(str) + "%"
    view["sharpe"] = view["sharpe"].round(2)
    view["icc"] = view["icc"].round(3)
    print(view.to_string(index=False))
    print(f"\nsaved: {REPRO_OUT / 'sector_size_breakdown.csv'}")


if __name__ == "__main__":
    main()
