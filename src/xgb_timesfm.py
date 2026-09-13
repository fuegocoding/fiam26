"""XGBoost on the 147 characteristics plus TimesFM 3.0 forecast features.

Same expanding-window folds, rank transform, portfolio rules and metrics as
xgb_baseline.py. Added columns (joined on permno/eom from the zero-shot run):
tfm_forecast, tfm_q10, tfm_q90, tfm_spread. They are left unranked because a
monotone transform does not change tree splits; XGBoost handles their NaNs
(rows before 2016 or with degenerate histories) natively.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from xgb_baseline import (
    load_model_table,
    rank_transform,
    expanding_folds,
    fit_fold,
    evaluate,
)
from timesfm_portfolio import build_portfolio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

TFM_FEATURES = ["tfm_forecast", "tfm_q10", "tfm_q90", "tfm_spread"]


def main():
    table, features = load_model_table()
    print(f"model table: {len(table):,} rows, {len(features)} features")

    tfm = pd.read_parquet(OUT / "preds_timesfm_all.parquet")
    tfm["eom"] = pd.to_datetime(tfm["eom"])
    table = table.merge(tfm[["permno", "eom"] + TFM_FEATURES], on=["permno", "eom"], how="left")
    covered = table["tfm_forecast"].notna().mean()
    print(f"tfm features joined: {covered:.1%} of rows covered")

    print("rank-transforming characteristics within each month ...")
    transformed = rank_transform(table, features)
    for f in features:
        table[f] = transformed[f].values
    del transformed

    all_features = features + TFM_FEATURES
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
        out["pred_xgb_tfm"] = pred
        all_preds.append(out)
    preds = pd.concat(all_preds, ignore_index=True)

    y = preds["ret_exc_lead1m"].values
    yhat = preds["pred_xgb_tfm"].values
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)
    print(f"\nOOS R2 (XGBoost + TimesFM features): {r2:.5f}")

    port, long_sets, short_sets = build_portfolio(
        preds.rename(columns={"pred_xgb_tfm": "tfm_forecast"}),
        pred_col="tfm_forecast",
    )
    metrics = evaluate(port, long_sets, short_sets)
    print("\n--- XGBoost + TimesFM features: long top-100 / short bottom-100 ---")
    for k, v in metrics.items():
        if k in ("best_month", "worst_month"):
            print(f"{k}: {v['target_month'].date()} ret={v['ret_ls']:.4f}")
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    preds.to_csv(OUT / "preds_xgb_timesfm.csv", index=False)
    print(f"\nsaved: {OUT / 'preds_xgb_timesfm.csv'}")


if __name__ == "__main__":
    main()
