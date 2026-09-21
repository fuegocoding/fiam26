"""Evaluate TimesFM 3.0 zero-shot forecasts as a standalone stock-level strategy.

OOS window only: target months 2021-01..2026-08. Two sizing variants:
1. equal weight, long top-100 / short bottom-100 (same rules as xgb_baseline);
2. vol-scaled: within each leg, weights proportional to 1 / TimesFM p90-p10 spread.
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from xgb_baseline import evaluate, turnover_rate, max_drawdown
from portfolio_constraints import apply_investability_screen

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
N_SIDE = 100


def build_portfolio(
    preds,
    spread_scaled=False,
    pred_col="tfm_forecast",
    screen_kwargs=None,
):
    if screen_kwargs:
        preds = apply_investability_screen(preds, **screen_kwargs)
    preds = preds.sort_values(["target_month", "permno"])
    rows = []
    long_sets = {}
    short_sets = {}
    for month, g in preds.groupby("target_month"):
        g = g.sort_values(pred_col, ascending=False)
        n_long = min(N_SIDE, len(g) // 2)
        n_short = min(N_SIDE, len(g) - n_long)
        top = g.head(n_long)
        bottom = g.tail(n_short)
        long_sets[month] = set(top["permno"])
        short_sets[month] = set(bottom["permno"])
        if spread_scaled:
            wl = 1.0 / top["tfm_spread"].clip(lower=1e-4)
            ws = 1.0 / bottom["tfm_spread"].clip(lower=1e-4)
            ret_long = float(np.dot(wl / wl.sum(), top["ret_exc_lead1m"].fillna(0.0)))
            ret_short = float(np.dot(ws / ws.sum(), bottom["ret_exc_lead1m"].fillna(0.0)))
        else:
            ret_long = float(top["ret_exc_lead1m"].fillna(0.0).mean())
            ret_short = float(bottom["ret_exc_lead1m"].fillna(0.0).mean())
        rows.append({
            "target_month": month,
            "n_long": len(top),
            "n_short": len(bottom),
            "n_long_missing": int(top["ret_exc_lead1m"].isna().sum()),
            "n_short_missing": int(bottom["ret_exc_lead1m"].isna().sum()),
            "ret_long": ret_long,
            "ret_short": ret_short,
        })
        g2 = g.copy()
        try:
            g2["decile"] = pd.qcut(g2[pred_col].rank(method="first"), 10, labels=False)
            dec10 = g2[g2["decile"] == 9]["ret_exc_lead1m"].mean()
            dec1 = g2[g2["decile"] == 0]["ret_exc_lead1m"].mean()
        except ValueError:
            dec10 = dec1 = np.nan
        rows[-1]["ret_decile10"] = dec10
        rows[-1]["ret_decile1"] = dec1
    port = pd.DataFrame(rows)
    port["ret_ls"] = port["ret_long"] - port["ret_short"]
    port["ret_decile_ls"] = port["ret_decile10"] - port["ret_decile1"]
    return port, long_sets, short_sets


def main():
    parser = argparse.ArgumentParser(description="Evaluate TimesFM portfolios.")
    parser.add_argument(
        "--min-me-percentile",
        type=float,
        help="optional within-month market-cap percentile floor",
    )
    parser.add_argument(
        "--min-dolvol-percentile",
        type=float,
        help="optional within-month dollar-volume percentile floor",
    )
    parser.add_argument(
        "--exclude-size-group",
        action="append",
        default=[],
        help="optional size group to exclude; repeat for micro and nano",
    )
    args = parser.parse_args()
    preds = pd.read_parquet(OUT / "preds_timesfm_all.parquet")
    preds["eom"] = pd.to_datetime(preds["eom"])
    preds["target_month"] = pd.to_datetime(preds["target_month"])
    preds = preds.dropna(subset=["tfm_forecast"])
    oos = preds[preds["target_month"] >= pd.Timestamp("2021-01-31")]
    screen_kwargs = {
        "min_me_percentile": args.min_me_percentile,
        "min_dolvol_percentile": args.min_dolvol_percentile,
        "excluded_size_groups": tuple(args.exclude_size_group),
        "date_column": "eom",
    }
    if any(
        value is not None and value != ()
        for value in (
            args.min_me_percentile,
            args.min_dolvol_percentile,
            screen_kwargs["excluded_size_groups"],
        )
    ):
        chars = pd.read_parquet(
            ROOT / "data" / "raw" / "chars_final_with_names.parquet",
            columns=["permno", "eom", "me", "dolvol", "size_grp"],
        )
        chars["eom"] = pd.to_datetime(chars["eom"])
        oos = oos.merge(chars, on=["permno", "eom"], how="left", validate="many_to_one")
        before = len(oos)
        oos = apply_investability_screen(oos, **screen_kwargs)
        print(f"investability screen: {len(oos):,}/{before:,} rows retained")
    print(f"OOS rows: {len(oos):,} ({oos['target_month'].min().date()}..{oos['target_month'].max().date()})")

    y = oos["ret_exc_lead1m"].values
    yhat = oos["tfm_forecast"].values
    has_y = ~np.isnan(y)
    y, yhat = y[has_y], yhat[has_y]
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)
    corr = np.corrcoef(y, yhat)[0, 1]
    print(f"OOS R2 (TimesFM zero-shot, zero benchmark): {r2:.5f}   corr: {corr:.5f}")

    for scaled in (False, True):
        port, long_sets, short_sets = build_portfolio(oos, spread_scaled=scaled)
        metrics = evaluate(port, long_sets, short_sets)
        label = "vol-scaled" if scaled else "equal-weight"
        print(f"\n--- TimesFM {label}: long top-{N_SIDE} / short bottom-{N_SIDE} ---")
        for k, v in metrics.items():
            if k in ("best_month", "worst_month"):
                print(f"{k}: {v['target_month'].date()} ret={v['ret_ls']:.4f}")
            elif isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        tag = "volscale" if scaled else "eq"
        suffix = "_screened" if any(
            value is not None and value != ()
            for value in (
                args.min_me_percentile,
                args.min_dolvol_percentile,
                screen_kwargs["excluded_size_groups"],
            )
        ) else ""
        port.to_csv(OUT / f"port_monthly_timesfm_{tag}{suffix}.csv", index=False)
        longs = pd.Series({str(k.date()): sorted(v) for k, v in long_sets.items()}, name="long")
        shorts = pd.Series({str(k.date()): sorted(v) for k, v in short_sets.items()}, name="short")
        longs.to_json(OUT / f"timesfm_long_sets_{tag}.json")
        shorts.to_json(OUT / f"timesfm_short_sets_{tag}.json")


if __name__ == "__main__":
    main()
