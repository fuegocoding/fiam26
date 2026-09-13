"""Opt-in XGBoost evaluation with redacted council stock-month features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from agentic.features import merge_council_features
from xgb_baseline import expanding_folds, evaluate, fit_fold, load_model_table, rank_transform

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
COUNCIL = OUT / "council_features.parquet"


def main() -> None:
    if not COUNCIL.exists():
        raise FileNotFoundError(
            f"missing {COUNCIL}; run python -m src.agentic.run_council first"
        )
    table, base_features = load_model_table()
    council = pd.read_parquet(COUNCIL)
    table, council_features = merge_council_features(table, council)
    print(
        f"model table: {len(table):,} rows, {len(base_features)} baseline features, "
        f"{len(council_features)} council features"
    )

    transformed = rank_transform(table, base_features)
    for feature in base_features:
        table[feature] = transformed[feature].values
    del transformed

    all_features = base_features + council_features
    folds = expanding_folds(table)
    all_preds = []
    for i, (train_hi, val_lo, val_hi, test_lo, test_hi) in enumerate(folds):
        train = table[table["target_month"] <= train_hi]
        val = table[(table["target_month"] >= val_lo) & (table["target_month"] <= val_hi)]
        test = table[(table["target_month"] >= test_lo) & (table["target_month"] <= test_hi)]
        print(
            f"fold {i}: train<={train_hi.date()} ({len(train):,}) "
            f"val {val_lo.date()}..{val_hi.date()} ({len(val):,}) "
            f"test {test_lo.date()}..{test_hi.date()} ({len(test):,})"
        )
        pred, _ = fit_fold(train, val, test, all_features)
        output = test[["permno", "eom", "target_month", "ret_exc_lead1m"]].copy()
        output["pred_xgb_agentic"] = pred
        all_preds.append(output)
    preds = pd.concat(all_preds, ignore_index=True)
    y = preds["ret_exc_lead1m"].to_numpy()
    yhat = preds["pred_xgb_agentic"].to_numpy()
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum(y**2)
    print(f"OOS R2 (XGBoost + agentic council features): {r2:.5f}")

    # Reuse the baseline portfolio mechanics without changing constraints.
    portfolio_input = preds.rename(columns={"pred_xgb_agentic": "pred_xgb"})
    from xgb_baseline import build_portfolio

    port, long_sets, short_sets = build_portfolio(portfolio_input)
    metrics = evaluate(port, long_sets, short_sets)
    print("--- XGBoost + council features: long top-100 / short bottom-100 ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    preds.to_csv(OUT / "preds_xgb_agentic.csv", index=False)


if __name__ == "__main__":
    main()
