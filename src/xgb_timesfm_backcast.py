"""XGBoost+TimesFM BACKCAST folds for 2018-2020 (full-window blend support).

The competition spec's expanding window starts test folds at 2021 (train
2015-2018, val 2019-20). This script applies the SAME fold structure shifted
back to produce predictions for target years 2018-2020:

  test 2018: train targets 2015-02..2015-12 (thin - data starts Feb 2015),
             val 2016-01..2017-12
  test 2019: train ..2016-12, val 2017-2018
  test 2020: train ..2017-12, val 2018-2019

Each target month is predicted by a model that never trained on it. Labels:
2018 is thin-train; 2019-2020 are quasi-OOS (these periods later serve as
validation for the 2021+ folds' models, but the backcast models here are
refit independently). 2016-2017 cannot be backcast for XGBoost under any
windowed scheme (no earlier training data exists).

Purpose: extend the blended TimesFM+XGB book to 2018-2020 so the overlay/DD
analysis covers the 2020 small-cap squeeze cascade. NOT a claim of OOS
purity on par with the 2021-26 test window.

Output: output/preds_xgb_timesfm_backcast.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xgb_baseline import load_model_table, rank_transform, fit_fold, expanding_folds
from timesfm_portfolio import build_portfolio
from xgb_baseline import evaluate, turnover_rate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
TFM_FEATURES = ["tfm_forecast", "tfm_q10", "tfm_q90", "tfm_spread"]


def backcast_folds(table):
    folds = []
    for y in (2018, 2019, 2020):
        train_hi = pd.Timestamp(f"{y - 3}-12-31")
        val_lo = pd.Timestamp(f"{y - 2}-01-31")
        val_hi = pd.Timestamp(f"{y - 1}-12-31")
        test_lo = pd.Timestamp(f"{y}-01-31")
        test_hi = pd.Timestamp(f"{y}-12-31")
        folds.append((train_hi, val_lo, val_hi, test_lo, test_hi))
    return folds


def main():
    table, features = load_model_table()
    print(f"model table: {len(table):,} rows, {len(features)} features")

    tfm = pd.read_parquet(OUT / "preds_timesfm_all.parquet")
    tfm["eom"] = pd.to_datetime(tfm["eom"])
    table = table.merge(tfm[["permno", "eom"] + TFM_FEATURES], on=["permno", "eom"], how="left")

    print("rank-transforming characteristics within each month ...")
    transformed = rank_transform(table, features)
    for f in features:
        table[f] = transformed[f].values
    del transformed

    all_features = features + TFM_FEATURES
    all_preds = []
    for i, (train_hi, val_lo, val_hi, test_lo, test_hi) in enumerate(backcast_folds(table)):
        train = table[table["target_month"] <= train_hi].dropna(subset=["ret_exc_lead1m"])
        val = table[(table["target_month"] >= val_lo) & (table["target_month"] <= val_hi)].dropna(
            subset=["ret_exc_lead1m"]
        )
        test = table[(table["target_month"] >= test_lo) & (table["target_month"] <= test_hi)]
        print(f"backcast fold {i}: train<={train_hi.date()} ({len(train):,}) "
              f"val {val_lo.date()}..{val_hi.date()} ({len(val):,}) "
              f"test {test_lo.date()}..{test_hi.date()} ({len(test):,})")
        pred, model = fit_fold(train, val, test, all_features)
        out = test[["permno", "eom", "target_month", "ret_exc_lead1m"]].copy()
        out["pred_xgb_tfm"] = pred
        all_preds.append(out)
    preds = pd.concat(all_preds, ignore_index=True)

    evaluated = preds.dropna(subset=["ret_exc_lead1m"])
    y = evaluated["ret_exc_lead1m"].values
    yhat = evaluated["pred_xgb_tfm"].values
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)
    print(f"\nbackcast R2 (2018-2020, thin-train caveat): {r2:.5f}")

    port, long_sets, short_sets = build_portfolio(
        preds.rename(columns={"pred_xgb_tfm": "tfm_forecast"}),
        pred_col="tfm_forecast",
    )
    metrics = evaluate(port, long_sets, short_sets)
    print("\n--- XGB+TFM backcast book 2018-2020: long top-100 / short bottom-100 ---")
    for k, v in metrics.items():
        if k in ("best_month", "worst_month"):
            print(f"{k}: {v['target_month'].date()} ret={v['ret_ls']:.4f}")
        elif isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"turnover: long {turnover_rate(long_sets):.1%}, short {turnover_rate(short_sets):.1%}")

    preds.to_csv(OUT / "preds_xgb_timesfm_backcast.csv", index=False)
    print(f"\nsaved: {OUT / 'preds_xgb_timesfm_backcast.csv'}")


if __name__ == "__main__":
    main()
