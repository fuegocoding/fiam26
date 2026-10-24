"""Walk-forward trained sleeves: xgb_rank and filing_events
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 13.3, 14; ports of first-round recipes).

Both sleeves fit per test year Y on targets through Y-3, early-stop on
Y-2..Y-1, predict Y — the same folds as the first round's timing masks.
Four seeds, heavily regularized XGBoost; the target is a cross-sectional
rank of the next-month excess return. Filing-event features use the
8-K corpus's dated cik->permno links (evidence-based) rather than name
normalization, and each month's features use only filings known before
that month (acceptance time).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

XGB_PARAMS = {
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
SEEDS = (0, 1, 2, 3)
FIRST_TRAIN_TARGET = pd.Timestamp("2015-02-01")

FILING_FAMILIES = {
    "act13d": ["SC 13D", "SCHEDULE 13D"],
    "act13d_a": ["SC 13D/A", "SCHEDULE 13D/A"],
    "g13": ["SC 13G", "SCHEDULE 13G"],
    "g13_a": ["SC 13G/A", "SCHEDULE 13G/A"],
    "nt": ["NT 10-K", "NT 10-Q"],
    "comment": ["UPLOAD"],
    "corresp": ["CORRESP"],
    "offer": ["424B5", "424B4", "424B3"],
    "shelf": ["S-3", "S-3ASR", "S-1", "S-1/A"],
    "proxy": ["DEFC14A", "PREC14A", "DFAN14A", "PX14A6G"],
    "tender_i": ["SC TO-I"],
    "tender_t": ["SC TO-T", "SC 14D9"],
    "mergerp": ["DEFM14A", "PREM14A"],
    "dereg": ["15-12G", "15-12B", "15-15D", "25-NSE"],
    "amend": ["10-K/A", "10-Q/A"],
    "f144": ["144"],
    "formd": ["D", "D/A"],
    "s4": ["S-4"],
    "spin": ["10-12B", "10-12G"],
    "succ": ["8-K12B", "8-K12G3"],
    "sc13e3": ["SC 13E3"],
    "form25": ["25"],
    "effect": ["EFFECT"],
    "defa14a": ["DEFA14A"],
    "posam": ["POS AM"],
}
WINDOWS = {"1m": 1, "3m": 3, "12m": 12}


@dataclass
class FoldAudit:
    year: int
    train_end: str
    validation_window: str
    test_window: str
    train_rows: int
    validation_rows: int
    test_rows: int
    best_iterations: list


def _rank_transform(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    ranks = table.groupby("eom")[features].rank(method="dense")
    span = ranks.groupby(table["eom"]).transform("max") - 1.0
    return (ranks - 1.0) / span.replace(0.0, np.nan) * 2.0 - 1.0


def _split(table: pd.DataFrame, year: int, last_target: pd.Timestamp):
    months = table["target_month"]
    train_end = pd.Timestamp(f"{year - 3}-12-31")
    train = table.loc[months.between(FIRST_TRAIN_TARGET, train_end)]
    validation = table.loc[
        months.between(pd.Timestamp(f"{year - 2}-01-01"), pd.Timestamp(f"{year - 1}-12-31"))
    ]
    test = table.loc[
        months.between(
            pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year}-12-31"), last_target)
        )
    ]
    assert train["target_month"].max() <= train_end, "training labels leak past the cutoff"
    return (
        train.dropna(subset=["target"]),
        validation.dropna(subset=["target"]),
        test,
    )


def train_forecasts(
    table: pd.DataFrame,
    features: list[str],
    *,
    test_years: list[int],
    last_target: pd.Timestamp,
    target_column: str,
    output_column: str,
    device: str = "cpu",
    on_fold=None,
) -> tuple[pd.DataFrame, list[FoldAudit]]:
    import xgboost as xgb

    audits: list[FoldAudit] = []
    parts = []
    for year in test_years:
        train, validation, test = _split(table, year, last_target)
        predictions = []
        best_iterations = []
        for seed in SEEDS:
            model = xgb.XGBRegressor(random_state=seed, device=device, **XGB_PARAMS)
            model.fit(
                train[features].values,
                train[target_column].values,
                eval_set=[(validation[features].values, validation[target_column].values)],
                verbose=False,
            )
            best_iterations.append(int(model.best_iteration))
            predictions.append(model.predict(test[features].values))
        audits.append(
            FoldAudit(
                year=year,
                train_end=str(train["target_month"].max().date()) if len(train) else "empty",
                validation_window=f"{year - 2}..{year - 1}",
                test_window=str(year),
                train_rows=len(train),
                validation_rows=len(validation),
                test_rows=len(test),
                best_iterations=best_iterations,
            )
        )
        if on_fold:
            on_fold(audits[-1])
        part = test[["permno", "eom", "target_month"]].copy()
        part[output_column] = np.mean(predictions, axis=0)
        parts.append(part)
    return pd.concat(parts, ignore_index=True), audits


def build_xgb_rank(
    panel: pd.DataFrame,
    feature_columns: list[str],
    *,
    test_years: list[int],
    last_target: pd.Timestamp,
    device: str = "cpu",
    on_fold=None,
) -> tuple[pd.DataFrame, list[FoldAudit]]:
    columns = ["permno", "eom", "ret_exc_lead1m", *feature_columns]
    table = panel[columns].copy()
    table["eom"] = pd.to_datetime(table["eom"])
    table["target_month"] = table["eom"] + pd.offsets.MonthEnd(1)
    table[feature_columns] = _rank_transform(table, feature_columns).astype("float32")
    table["target"] = table.groupby("eom")["ret_exc_lead1m"].rank(pct=True) - 0.5
    return train_forecasts(
        table,
        feature_columns,
        test_years=test_years,
        last_target=last_target,
        target_column="target",
        output_column="pred_xgb_rank",
        device=device,
        on_fold=on_fold,
    )


def build_filing_event_features(
    filings_index: pd.DataFrame,
    links: pd.DataFrame,
    months: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Monthly filing-event counts per permno, using only filings known
    before each month (date_filed <= previous month end), keyed through
    dated cik->permno links."""
    index = filings_index.copy()
    index["date_filed"] = pd.to_datetime(index["date_filed"])
    index = index.dropna(subset=["cik"])
    index["cik"] = index["cik"].astype(str).str.lstrip("0")
    form_map = {}
    for family, forms in FILING_FAMILIES.items():
        for form in forms:
            form_map[form] = family
    index["family"] = index["form_type"].map(form_map)
    index = index.dropna(subset=["family"])
    index = index.drop(columns=["permno"], errors="ignore")

    links = links.copy()
    links["cik"] = links["cik"].astype(str).str.lstrip("0")
    links["permno"] = links["permno"].astype(int)

    rows = []
    for month_end in months:
        cutoff = month_end - pd.offsets.MonthEnd(1)
        known = index.loc[index["date_filed"] <= cutoff]
        merged = known.merge(links, on="cik", how="inner")
        merged = merged[merged["link_date"] <= cutoff]
        counts_frames = {}
        for window, months_back in WINDOWS.items():
            start = cutoff - pd.offsets.DateOffset(months=months_back) + pd.offsets.Day(1)
            windowed = merged[merged["date_filed"] >= start]
            wcounts = (
                windowed.groupby(["permno", "family"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=list(FILING_FAMILIES), fill_value=0)
            )
            wcounts.columns = [f"{family}_{window}" for family in wcounts.columns]
            counts_frames[window] = wcounts
        counts = counts_frames["1m"].join(counts_frames["3m"], how="outer").join(
            counts_frames["12m"], how="outer"
        ).fillna(0)
        counts["permno"] = counts.index
        counts["target_month"] = month_end
        rows.append(counts.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def build_filing_events(
    panel: pd.DataFrame,
    event_features: pd.DataFrame,
    *,
    test_years: list[int],
    last_target: pd.Timestamp,
    device: str = "cpu",
    on_fold=None,
) -> tuple[pd.DataFrame, list[FoldAudit]]:
    columns = ["permno", "eom", "ret_exc_lead1m", "size_grp", "gics"]
    table = panel[columns].copy()
    table["eom"] = pd.to_datetime(table["eom"])
    table["target_month"] = table["eom"] + pd.offsets.MonthEnd(1)
    features = [c for c in event_features.columns if c not in ("permno", "target_month")]
    table = table.merge(event_features, on=["permno", "target_month"], how="left")
    table[features] = table[features].fillna(0.0)
    table[features] = _rank_transform(table, features).astype("float32")
    peers = [
        table["eom"],
        table["size_grp"].astype(str),
        table["gics"].astype("string").str[:2].fillna("na"),
    ]
    table["target"] = table.groupby(peers)["ret_exc_lead1m"].rank(pct=True) - 0.5
    return train_forecasts(
        table,
        features,
        test_years=test_years,
        last_target=last_target,
        target_column="target",
        output_column="pred_filing_events",
        device=device,
        on_fold=on_fold,
    )
