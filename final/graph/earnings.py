"""Dated EPS-surprise events per security and the next report after a month."""

import numpy as np
import pandas as pd


def dated_eps_events(surprises: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Maps each announcement to the CRSP link valid on its report date."""
    events = surprises.loc[
        np.isfinite(surprises.suescore),
        ["TICKER", "fpe", "anndats", "suescore"],
    ].copy()
    mapped = events.merge(links, on="TICKER", how="inner")
    mapped = mapped[(mapped.sdate <= mapped.anndats) & (mapped.anndats <= mapped.edate)]
    mapped = mapped.sort_values(
        ["TICKER", "fpe", "SCORE", "sdate"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["TICKER", "fpe"])
    mapped["next_suescore"] = mapped.suescore.clip(-20, 20)
    return (
        mapped.groupby(["permno", "anndats"], as_index=False)
        .next_suescore.mean()
        .sort_values("anndats")
    )


def attach_next_eps(
    panel: pd.DataFrame, events: pd.DataFrame, max_days: int = 92
) -> pd.DataFrame:
    """Attaches the first report after each month-end, within a quarter."""
    left = panel.copy()
    right = events.copy()
    left["eom"] = pd.to_datetime(left.eom).astype("datetime64[ns]")
    right["anndats"] = pd.to_datetime(right.anndats).astype("datetime64[ns]")
    result = pd.merge_asof(
        left.sort_values("eom"),
        right.sort_values("anndats"),
        by="permno",
        left_on="eom",
        right_on="anndats",
        direction="forward",
        allow_exact_matches=False,
    )
    result["days_to_eps"] = (result.anndats - result.eom).dt.days
    return result[
        result.days_to_eps.between(1, max_days) & np.isfinite(result.next_suescore)
    ].copy()
