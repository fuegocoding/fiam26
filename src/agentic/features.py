"""Safe panel joins for council-derived features."""
from __future__ import annotations

import pandas as pd


def merge_council_features(
    table: pd.DataFrame, council: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Left-join council columns while preserving the baseline panel exactly."""
    keys = ["permno", "eom"]
    for key in keys:
        if key not in table.columns or key not in council.columns:
            raise ValueError(f"both tables require join key {key!r}")
    council = council.copy()
    table = table.copy()
    table["eom"] = pd.to_datetime(table["eom"])
    council["eom"] = pd.to_datetime(council["eom"])
    if council.duplicated(keys).any():
        raise ValueError("council features must be unique on (permno, eom)")
    feature_columns = sorted(
        column
        for column in council.columns
        if column.startswith("council_") and column not in {"council_target"}
    )
    merged = table.merge(council[keys + feature_columns], on=keys, how="left", validate="m:1")
    return merged, feature_columns
