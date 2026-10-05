"""Trains annual XGBoost models to rank stocks using filing events."""

import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

import config
from sleeves import base
from sleeves.filing_events import edgar

SEEDS = (0, 1, 2, 3)
FIRST_TRAIN_TARGET = pd.Timestamp("2015-02-01")
# The recipe of the characteristics model: heavily regularized, because
# monthly returns are mostly noise.
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
    """Location of the annual forecasts."""
    return work_dir / "filing_events" / "forecasts.parquet"


def build(context: base.Context) -> pathlib.Path:
    """Fits a seed-averaged model per test year Y on targets through Y-3.

    The target is each month's percentile rank of the next-month excess
    return among stocks of the same size group and sector, centred at zero.
    The model can therefore earn nothing from size or sector tilts, only from
    what the filings say about a stock relative to its peers.  Event counts
    are zero for a stock with no filings, so a missing company link looks
    like a quiet stock and carries no information.  Early stopping uses
    Y-2..Y-1; no other sleeve's forecast is an input.
    """
    output = path(context.work_dir)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(
        edgar.build(
            context.panel_path,
            context.data_dir / config.FILINGS_FILE,
            output.parent,
        )
    )
    events = events.rename(columns={"return_month": "target_month"})
    features = [c for c in events.columns if c not in ("permno", "target_month")]
    table = pq.read_table(
        context.panel_path,
        columns=["permno", "eom", "ret_exc_lead1m", "size_grp", "gics"],
    ).to_pandas()
    table["eom"] = pd.to_datetime(table.eom)
    table["target_month"] = table.eom + pd.offsets.MonthEnd(1)
    table = table.merge(events, on=["permno", "target_month"], how="left")
    table[features] = table[features].fillna(0.0)
    table[features] = _rank_transform(table, features).astype("float32")
    peers = [
        table.eom,
        table.size_grp.astype(str),
        table.gics.astype("string").str[:2].fillna("na"),
    ]
    table["target"] = table.groupby(peers).ret_exc_lead1m.rank(pct=True) - 0.5
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
            predictions.append(model.predict(test[features].values))
            print(
                f"Filing events {year} seed {seed}: best iteration"
                f" {model.best_iteration}",
                flush=True,
            )
        part = test[["permno", "eom", "target_month"]].copy()
        part["pred_filing_events"] = np.mean(predictions, axis=0)
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
    """Dense monthly ranks mapped onto [-1, 1]; constant columns become NaN."""
    ranks = table.groupby("eom")[features].rank(method="dense")
    span = ranks.groupby(table.eom).transform("max") - 1.0
    return (ranks - 1.0) / span.replace(0.0, np.nan) * 2.0 - 1.0
