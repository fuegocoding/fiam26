"""Local evaluation of TimesFM zero-shot forecasts on alternative series.

CPU-only, pandas/numpy (no parquet reader needed): reads
output/timesfm_alt/preds_<variant>.csv + attrs.csv written by
src/timesfm_forecast_alt.py. For every variant (overall and per size group)
it reports the mean monthly Spearman ICC with HAC(3) t-stat, the standalone
long top-100 / short bottom-100 portfolio (equal weight, monthly, same
mechanics as src/timesfm_portfolio.py), and a decile table. It then reports
cross-variant orthogonality: mean monthly forecast rank correlations and the
correlation matrix of monthly long/short returns. If the 'dolvol' variant is
present, a derived signal 'dolvol_jump' = tfm_forecast - ctx_mean6 (expected
volume jump vs the recent average) is evaluated as an extra variant.

Artifacts: output/timesfm_alt/eval_summary.csv, port_monthly_<variant>.csv.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "output" / "timesfm_alt"
SIDE = 100
SIZES = ["ALL", "mega", "large", "small", "micro", "nano"]


def hac_tstat(values, maxlags=3):
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return np.nan
    d = x - x.mean()
    gamma0 = float(np.dot(d, d)) / n
    var = gamma0
    for lag in range(1, maxlags + 1):
        if n > lag:
            var += 2.0 * (1.0 - lag / (maxlags + 1.0)) * float(np.dot(d[lag:], d[:-lag])) / n
    se = np.sqrt(max(var, 0.0) / n)
    return float(x.mean() / se) if se > 0 else np.nan


def max_drawdown(series):
    cum = (1.0 + series).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def turnover_rate(month_sets):
    rates = []
    months = sorted(month_sets)
    for prev, cur in zip(months, months[1:]):
        if len(month_sets[cur]) == 0:
            continue
        rates.append(len(month_sets[cur] - month_sets[prev]) / len(month_sets[cur]))
    return float(np.mean(rates)) if rates else np.nan


def build_portfolio(preds, pred_col="tfm_forecast"):
    preds = preds.dropna(subset=[pred_col])
    rows = []
    long_sets = {}
    short_sets = {}
    for month, g in preds.groupby("target_month"):
        g = g.sort_values(pred_col, ascending=False)
        n = len(g)
        n_long = min(SIDE, n // 2)
        n_short = min(SIDE, n - n_long)
        if n_long == 0 or n_short == 0:
            continue
        top = g.head(n_long)
        bottom = g.tail(n_short)
        long_sets[month] = set(top["permno"])
        short_sets[month] = set(bottom["permno"])
        rows.append({
            "target_month": month,
            "n_long": n_long,
            "n_short": n_short,
            "ret_long": float(top["ret_exc_lead1m"].mean()),
            "ret_short": float(bottom["ret_exc_lead1m"].mean()),
        })
    port = pd.DataFrame(rows)
    if len(port):
        port["ret_ls"] = port["ret_long"] - port["ret_short"]
    return port, long_sets, short_sets


def rank_corr(a, b):
    ra = a.rank().to_numpy(dtype=float)
    rb = b.rank().to_numpy(dtype=float)
    ok = ~(np.isnan(ra) | np.isnan(rb))
    if ok.sum() < 2 or np.std(ra[ok]) == 0 or np.std(rb[ok]) == 0:
        return np.nan
    return float(np.corrcoef(ra[ok], rb[ok])[0, 1])


def icc_series(preds, pred_col="tfm_forecast"):
    ics = []
    for month, g in preds.groupby("target_month"):
        g = g.dropna(subset=[pred_col, "ret_exc_lead1m"])
        if len(g) >= 10:
            ics.append(rank_corr(g[pred_col], g["ret_exc_lead1m"]))
    return pd.Series(ics)


def summarize(preds, pred_col="tfm_forecast"):
    iccs = icc_series(preds, pred_col)
    port, long_sets, short_sets = build_portfolio(preds, pred_col)
    y = preds["ret_exc_lead1m"].to_numpy()
    yhat = preds[pred_col].to_numpy()
    ok = ~(np.isnan(y) | np.isnan(yhat))
    r2 = np.nan
    corr = np.nan
    if ok.sum() > 2:
        yv, hv = y[ok], yhat[ok]
        denom = np.sum(yv ** 2)
        if denom > 0:
            r2 = 1.0 - np.sum((yv - hv) ** 2) / denom
        if np.std(yv) > 0 and np.std(hv) > 0:
            corr = float(np.corrcoef(yv, hv)[0, 1])
    if len(port) == 0:
        return {
            "n_names": preds["permno"].nunique(),
            "n_months": len(iccs),
            "icc": np.nan, "icc_t": hac_tstat(iccs),
            "ann_ls": np.nan, "sharpe_ls": np.nan, "maxdd": np.nan,
            "hit_rate": np.nan, "to_long": np.nan, "to_short": np.nan,
            "r2": r2, "corr_target": corr, "port": port,
            "long_sets": long_sets, "short_sets": short_sets,
        }
    ls = port["ret_ls"]
    return {
        "n_names": preds["permno"].nunique(),
        "n_months": len(port),
        "icc": float(iccs.mean()) if len(iccs) else np.nan,
        "icc_t": hac_tstat(iccs),
        "ann_ls": float(ls.mean() * 12),
        "sharpe_ls": float(ls.mean() / ls.std(ddof=1) * np.sqrt(12)) if ls.std(ddof=1) > 0 else np.nan,
        "maxdd": max_drawdown(ls),
        "hit_rate": float((ls > 0).mean()),
        "to_long": turnover_rate(long_sets),
        "to_short": turnover_rate(short_sets),
        "r2": r2,
        "corr_target": corr,
        "port": port,
        "long_sets": long_sets,
        "short_sets": short_sets,
    }


def decile_table(preds, pred_col="tfm_forecast"):
    preds = preds.dropna(subset=[pred_col, "ret_exc_lead1m"])
    if len(preds) < 100:
        return None
    ranks = preds.groupby("target_month")[pred_col].rank(method="first")
    preds = preds.assign(decile=pd.qcut(ranks, 10, labels=False))
    return preds.groupby("decile")["ret_exc_lead1m"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()
    run_dir = Path(args.dir)

    attrs = pd.read_csv(run_dir / "attrs.csv")
    attrs["target_month"] = pd.to_datetime(attrs["target_month"])

    variants = {}
    for path in sorted(run_dir.glob("preds_*.csv")):
        name = path.stem.removeprefix("preds_")
        df = pd.read_csv(path)
        df["eom"] = pd.to_datetime(df["eom"])
        df["target_month"] = pd.to_datetime(df["target_month"])
        variants[name] = df
    if not variants:
        raise SystemExit(f"no preds_*.csv found in {run_dir}")

    if "dolvol" in variants and "ctx_mean6" in variants["dolvol"].columns:
        dv = variants["dolvol"].copy()
        dv["tfm_forecast"] = dv["tfm_forecast"] - dv["ctx_mean6"]
        variants["dolvol_jump"] = dv

    all_rows = []
    ports = {}
    for name, df in variants.items():
        df = df.merge(attrs, on=["permno", "target_month"], how="left")
        print(f"\n{'=' * 72}\nvariant: {name}\n{'=' * 72}")
        dec = decile_table(df)
        if dec is not None:
            print("next-month excess return by forecast decile (1..10): "
                  + " ".join(f"{v:+.2%}" for v in dec.values))
        for size in SIZES:
            sub = df if size == "ALL" else df[df["size_grp"] == size]
            m = summarize(sub)
            m["variant"] = name
            m["size"] = size
            all_rows.append(m)
            ports[(name, size)] = m["port"]
            print(
                f"{size:>5}: n={m['n_names']:>5,} months={m['n_months']:>3} "
                f"ICC={m['icc']:+.4f} [t={m['icc_t']:+.2f}] "
                f"annLS={m['ann_ls']:+.1%} Sharpe={m['sharpe_ls']:+.2f} "
                f"maxDD={m['maxdd']:+.1%} hit={m['hit_rate']:.0%} "
                f"R2={m['r2']:+.4f}"
            )
        if m["port"] is not None and len(m["port"]):
            yearly = m["port"].groupby(m["port"]["target_month"].dt.year)["ret_ls"].agg(["mean", "count"])
            yearly["mean"] = (yearly["mean"] * 100).round(2)
            print("yearly monthly L/S %: " + "  ".join(
                f"{y}:{v}" for y, v in yearly["mean"].items()))

    summary = pd.DataFrame([
        {k: row.get(k) for k in ("variant", "size", "n_names", "n_months", "icc",
                                 "icc_t", "ann_ls", "sharpe_ls", "maxdd", "hit_rate",
                                 "to_long", "to_short", "r2", "corr_target")}
        for row in all_rows
    ])
    summary.to_csv(run_dir / "eval_summary.csv", index=False)
    for row in all_rows:
        tag = row["variant"] if row["size"] == "ALL" else f"{row['variant']}_{row['size']}"
        if row["port"] is not None and len(row["port"]):
            row["port"].to_csv(run_dir / f"port_monthly_{tag}.csv", index=False)

    print(f"\n{'=' * 72}\northogonality across variants (ALL universe)\n{'=' * 72}")
    pairs = []
    for a, b in combinations(list(variants), 2):
        m = variants[a].merge(
            variants[b][["permno", "target_month", "tfm_forecast"]],
            on=["permno", "target_month"], suffixes=("", "_b")
        ).dropna(subset=["tfm_forecast", "tfm_forecast_b"])
        ics = []
        for month, g in m.groupby("target_month"):
            if len(g) >= 10:
                ics.append(rank_corr(g["tfm_forecast"], g["tfm_forecast_b"]))
        if ics:
            pairs.append({"a": a, "b": b, "mean_rank_corr": float(np.mean(ics)),
                          "n_months": len(ics)})
    pair_df = pd.DataFrame(pairs)
    if len(pair_df):
        print(pair_df.round(4).to_string(index=False))

    ls_frames = {}
    for row in all_rows:
        if row["size"] == "ALL" and row["port"] is not None and len(row["port"]):
            ls_frames[row["variant"]] = row["port"].set_index("target_month")["ret_ls"]
    if len(ls_frames) >= 2:
        ls_df = pd.DataFrame(ls_frames).dropna()
        print("\nmonthly L/S return correlation matrix:")
        print(ls_df.corr().round(3).to_string())

    print(f"\nsaved: {run_dir / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
