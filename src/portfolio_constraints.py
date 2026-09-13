"""Optional investability screens for the long-short portfolio layer."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def apply_investability_screen(
    predictions: pd.DataFrame,
    *,
    min_me_percentile: float | None = None,
    min_dolvol_percentile: float | None = None,
    excluded_size_groups: Iterable[str] = (),
    date_column: str = "eom",
) -> pd.DataFrame:
    """Filter predictions using only contemporaneous size/liquidity fields.

    Percentiles are calculated within the characteristic month (`eom`) so the
    screen is comparable through time and cannot use future returns. The screen
    is opt-in; passing no thresholds and no exclusions returns a copy unchanged.
    """
    thresholds = {
        "me": min_me_percentile,
        "dolvol": min_dolvol_percentile,
    }
    for column, threshold in thresholds.items():
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{column} percentile must be in [0, 1]")

    excluded = {str(value).lower() for value in excluded_size_groups}
    required = [date_column]
    if min_me_percentile is not None:
        required.append("me")
    if min_dolvol_percentile is not None:
        required.append("dolvol")
    if excluded:
        required.append("size_grp")
    missing = sorted(set(required) - set(predictions.columns))
    if missing:
        raise ValueError(f"missing investability columns: {missing}")

    out = predictions.copy()
    mask = pd.Series(True, index=out.index)
    if min_me_percentile is not None:
        me_pct = out.groupby(date_column)["me"].rank(pct=True, method="first")
        mask &= me_pct >= min_me_percentile
    if min_dolvol_percentile is not None:
        dolvol_pct = out.groupby(date_column)["dolvol"].rank(pct=True, method="first")
        mask &= dolvol_pct >= min_dolvol_percentile
    if excluded:
        mask &= ~out["size_grp"].astype(str).str.lower().isin(excluded)
    return out.loc[mask].copy()
