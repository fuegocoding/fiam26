"""Walk-forward XGBoost ranking next-month returns from characteristics."""

import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

import config
from sleeves import base

SEEDS = (0, 1, 2, 3)
FIRST_TRAIN_TARGET = pd.Timestamp("2015-02-01")
# Heavily regularized: monthly returns are mostly noise, and the unregularized
# default (learning rate 0.3) stopped at a single tree.
PARAMS = {
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 500,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "reg_lambda": 10.0,
    "n_estimators": 3000,
    "early_stopping_rounds": 200,
    "tree_method": "hist",
    "eval_metric": "rmse",
}


def path(work_dir: pathlib.Path) -> pathlib.Path:
    """Location of the annual XGBoost forecasts."""
    return work_dir / "xgb_rank" / "forecasts.parquet"


def build(context: base.Context) -> pathlib.Path:
    """Fits a seed-averaged model per test year Y on targets through Y-3.

    The target is each month's percentile rank of the next-month excess
    return, centred at zero: a long/short book only needs the cross-sectional
    order, and ranks remove the outliers that dominate a raw-return fit.
    Early stopping uses Y-2..Y-1. Characteristics are ranked to [-1, 1]
    within each month; no other sleeve's forecast is an input.
    """
    output = path(context.work_dir)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    features = list(pd.read_csv(context.data_dir / config.FACTOR_LIST_FILE)["variable"])
    columns = ["permno", "eom", "ret_exc_lead1m", *features]
    table = pq.read_table(context.panel_path, columns=columns).to_pandas()
    table["eom"] = pd.to_datetime(table.eom)
    table["target_month"] = table.eom + pd.offsets.MonthEnd(1)
    table[features] = _rank_transform(table, features).astype("float32")
    table["target"] = table.groupby("eom").ret_exc_lead1m.rank(pct=True) - 0.5
    forecasts = []
    for year in config.TEST_YEARS:
        train, validation, test = _split(table, year)
        predictions = []
        for seed in SEEDS:
            model = xgb.XGBRegressor(random_state=seed, device=context.device, **PARAMS)
            model.fit(
                train[features].values,
                train.target.values,
                eval_set=[(validation[features].values, validation.target.values)],
                verbose=False,
            )
            model.save_model(output.parent / f"model_{year}_seed{seed}.json")
            predictions.append(model.predict(test[features].values))
            print(
                f"XGB rank {year} seed {seed}: best iteration {model.best_iteration}",
                flush=True,
            )
        part = test[["permno", "eom", "target_month"]].copy()
        part["pred_xgb_rank"] = np.mean(predictions, axis=0)
        forecasts.append(part)
    pd.concat(forecasts, ignore_index=True).to_parquet(output, index=False)
    return output


def _split(
    table: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Training to Y-3, validation Y-2..Y-1 (labelled rows), test year Y."""
    months = table.target_month
    train_end = pd.Timestamp(f"{year - 3}-12-31")
    train = table.loc[months.between(FIRST_TRAIN_TARGET, train_end)]
    validation = table.loc[
        months.between(
            pd.Timestamp(f"{year - 2}-01-01"), pd.Timestamp(f"{year - 1}-12-31")
        )
    ]
    test = table.loc[
        months.between(
            pd.Timestamp(f"{year}-01-01"),
            min(pd.Timestamp(f"{year}-12-31"), config.LAST_TARGET),
        )
    ]
    return (
        train.dropna(subset=["target"]),
        validation.dropna(subset=["target"]),
        test,
    )


def _rank_transform(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Dense monthly ranks mapped onto [-1, 1]; missing values stay missing."""
    ranks = table.groupby("eom")[features].rank(method="dense")
    span = ranks.groupby(table.eom).transform("max") - 1.0
    return (ranks - 1.0) / span.replace(0.0, np.nan) * 2.0 - 1.0
