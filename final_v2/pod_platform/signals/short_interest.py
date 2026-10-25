"""FINRA short-interest screen (faithful port of final/borrow.py).

The extract lacks publication dates. Ten business days excluding federal
holidays and Good Friday is a conservative delay beyond FINRA's schedule.
Latest published settlement per month-end; short_pct = short_shares /
(shares * 1e6); invalid values (<=0 or >1.5) become NaN. Keyed to the NEXT
return month. Missing short interest never blocks a short (NaN compares
false against the cap).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar


class _PublicationCalendar(USFederalHolidayCalendar):
    rules = [*USFederalHolidayCalendar.rules, GoodFriday]


PUBLICATION_DELAY = pd.offsets.CustomBusinessDay(
    10, calendar=_PublicationCalendar()
)


def public_dates(raw: pd.DataFrame) -> pd.Series:
    settlement = pd.to_datetime(raw.settlement)
    release = {
        date: date + PUBLICATION_DELAY for date in settlement.dropna().unique()
    }
    dates = settlement.map(release)
    if "publication_date" in raw:
        supplied = pd.to_datetime(raw.publication_date)
        if (supplied < settlement).any():
            raise ValueError("Short-interest publication precedes settlement")
        dates = supplied.fillna(dates)
    return dates


def _alphanumeric(symbols: pd.Series) -> pd.Series:
    return symbols.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def build_short_interest(external_path: str, panel_path: str) -> pd.DataFrame:
    """Returns permno, month (month-end), short_pct, days_to_cover."""
    ext = pd.read_parquet(external_path)
    raw = ext[ext["table"] == "finra"][
        ["symbol", "short_shares", "days_to_cover", "settlement"]
    ].copy()
    panel = pd.read_parquet(
        panel_path,
        columns=[
            "permno",
            "eom",
            "ticker",
            "ticker_name_reference_date",
            "shares",
        ],
    )
    panel["eom"] = pd.to_datetime(panel.eom)
    panel = panel.sort_values(["permno", "eom"])
    reference = pd.to_datetime(panel.ticker_name_reference_date)
    panel.loc[reference.isna() | reference.gt(panel.eom), "ticker"] = None
    panel["ticker"] = panel.groupby("permno").ticker.ffill()
    panel["symbol"] = _alphanumeric(panel.ticker.astype("string"))
    shared = panel.groupby(["eom", "symbol"]).permno.transform("nunique") > 1
    panel = panel.loc[
        ~shared & panel.symbol.notna(), ["permno", "eom", "symbol", "shares"]
    ]
    raw["symbol"] = _alphanumeric(raw.symbol)
    raw["public"] = public_dates(raw)
    months = []
    for month_end in sorted(panel.eom.unique()):
        known = raw.loc[raw.public <= month_end]
        if known.empty:
            continue
        latest = known.loc[known.settlement == known.settlement.max()]
        rows = panel.loc[panel.eom == month_end].merge(
            latest[["symbol", "short_shares", "days_to_cover"]],
            on="symbol",
            how="inner",
        )
        rows["short_pct"] = rows.short_shares / (rows.shares * 1e6)
        bad = (rows.short_pct <= 0) | (rows.short_pct > 1.5)
        rows.loc[bad, "short_pct"] = np.nan
        rows["return_month"] = pd.Timestamp(month_end).to_period("M") + 1
        months.append(
            rows[["permno", "return_month", "short_pct", "days_to_cover"]]
        )
    table = pd.concat(months, ignore_index=True)
    table["return_month"] = table.return_month.dt.to_timestamp("M")
    return table
