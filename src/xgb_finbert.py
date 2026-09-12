"""XGBoost baseline + FinBERT 8-K sentiment features.

Same spec as xgb_baseline.py (expanding window by TARGET month, folds 2021..2026),
with the 147 characteristics augmented by FinBERT 8-K features aggregated to
permno-month (filing in month t -> feature at eom t, predicting month t+1).

FinBERT features (rank-transformed within month like the chars, NaN = no filing):
  fb_sentiment_mean, fb_sentiment_min, fb_sentiment_max,
  fb_prob_negative_mean, fb_log_n_filings

Run: python -m src.xgb_finbert
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.xgb_baseline import (
    OUT,
    build_portfolio,
    evaluate,
    expanding_folds,
    fit_fold,
    load_model_table,
    rank_transform,
)

FINBERT = OUT / "finbert_scores.parquet"

FB_FEATURES = [
    "fb_sentiment_mean",
    "fb_sentiment_min",
    "fb_sentiment_max",
    "fb_prob_negative_mean",
    "fb_log_n_filings",
]


def finbert_monthly():
    s = pd.read_parquet(FINBERT)
    s["eom"] = pd.to_datetime(s["filing_date"]) + pd.offsets.MonthEnd(0)
    agg = s.groupby(["permno", "eom"]).agg(
        fb_sentiment_mean=("sentiment", "mean"),
        fb_sentiment_min=("sentiment_min_chunk", "min"),
        fb_sentiment_max=("sentiment_max_chunk", "max"),
        fb_prob_negative_mean=("prob_negative", "mean"),
        fb_n_filings=("document_id", "count"),
    )
    agg["fb_log_n_filings"] = np.log1p(agg["fb_n_filings"])
    return agg.reset_index()


def main():
    table, features = load_model_table()
    print(f"model table: {len(table):,} rows, {len(features)} chars, "
          f"{table['permno'].nunique():,} permnos")

    fb = finbert_monthly()
    print(f"finbert monthly rows: {len(fb):,} "
          f"({fb['permno'].nunique():,} permnos, {fb['eom'].nunique()} months)")
    table = table.merge(fb, on=["permno", "eom"], how="left")
    all_features = features + FB_FEATURES

    print("rank-transforming features within each month ...")
    transformed = rank_transform(table, all_features)
    for f in all_features:
        table[f] = transformed[f].values
    del transformed
    cov = table[FB_FEATURES].notna().mean().mul(100).round(1)
    print("finbert feature coverage % of stock-months:\n", cov.to_string())

    folds = expanding_folds(table)
    all_preds = []
    for i, (train_hi, val_lo, val_hi, test_lo, test_hi) in enumerate(folds):
        train = table[table["target_month"] <= train_hi]
        val = table[(table["target_month"] >= val_lo) & (table["target_month"] <= val_hi)]
        test = table[(table["target_month"] >= test_lo) & (table["target_month"] <= test_hi)]
        print(f"fold {i}: train<={train_hi.date()} ({len(train):,}) "
              f"val {val_lo.date()}..{val_hi.date()} ({len(val):,}) "
              f"test {test_lo.date()}..{test_hi.date()} ({len(test):,})")
        pred, model = fit_fold(train, val, test, all_features)
        out = test[["permno", "eom", "target_month", "ret_exc_lead1m"]].copy()
        out["pred_xgb"] = pred
        all_preds.append(out)
    preds = pd.concat(all_preds, ignore_index=True)

    y = preds["ret_exc_lead1m"].values
    yhat = preds["pred_xgb"].values
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)
    print(f"\nOOS R2 (chars + finbert, XGBoost defaults): {r2:.5f}")
    preds["year"] = preds["target_month"].dt.year
    for yr, g in preds.groupby("year"):
        gy, gyh = g["ret_exc_lead1m"].values, g["pred_xgb"].values
        r2y = 1.0 - np.sum((gy - gyh) ** 2) / np.sum(gy ** 2)
        print(f"  {yr}: R2 {r2y:.5f}  (n={len(g):,})")

    port, long_sets, short_sets = build_portfolio(preds)
    print("\n--- portfolio metrics WITH finbert (long top-100 / short bottom-100) ---")
    metrics_fb = evaluate(port, long_sets, short_sets)
    for k, v in metrics_fb.items():
        if k in ("best_month", "worst_month"):
            print(f"{k}: {v['target_month'].date()} ret={v['ret_ls']:.4f}")
        else:
            print(f"{k}: {v}")

    # baseline comparison from saved predictions (no retrain)
    base = pd.read_csv(OUT / "preds_xgb_default.csv", parse_dates=["target_month"])
    port_b, ls_b, ss_b = build_portfolio(base.rename(columns={"pred_xgb": "pred_xgb"}))
    print("\n--- portfolio metrics BASELINE (chars only) ---")
    metrics_b = evaluate(port_b, ls_b, ss_b)
    for k in ("avg_monthly_ls", "ann_ls", "sharpe_ls", "ir_benchmark",
              "alpha_ann", "alpha_tstat", "hit_rate_ls", "max_drawdown"):
        print(f"{k}: {metrics_b.get(k)}")

    print("\n--- delta (finbert - baseline) ---")
    for k in ("ann_ls", "sharpe_ls", "ir_benchmark", "alpha_ann", "hit_rate_ls"):
        b, f = metrics_b.get(k), metrics_fb.get(k)
        if b is not None and f is not None:
            print(f"{k}: {b:.4f} -> {f:.4f}  ({f - b:+.4f})")

    preds_out = preds[["permno", "eom", "target_month", "ret_exc_lead1m", "pred_xgb"]]
    preds_out.to_csv(OUT / "preds_xgb_finbert.csv", index=False)
    port_out = port.reset_index(drop=True)
    port_out.to_csv(OUT / "port_monthly_xgb_finbert.csv", index=False)
    print(f"\nsaved: {OUT/'preds_xgb_finbert.csv'}, {OUT/'port_monthly_xgb_finbert.csv'}")


if __name__ == "__main__":
    main()
