"""FINRA short interest as a borrow-availability screen for the short leg.

FINRA publishes every U.S. equity's short interest twice a month
(cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv).  Short shares as a
share of shares outstanding is the standard proxy for how scarce a borrow is:
a stock with 15% or more of its shares already short cannot be shorted at the
size or fee the cost model assumes.  `build` turns the files into one
point-in-time table and `optimizer.py` keeps such stocks out of the shorts.

Point-in-time rule: use publication dates when supplied, otherwise delay ten
business days excluding federal holidays and Good Friday. A month's table
row uses the newest settlement public by formation month-end. Tickers come from
the panel, carried forward per stock so the 2026 gap in the supplied ticker
column does not drop stocks.
"""

import pathlib

import numpy as np
import pandas as pd
from pandas.tseries.holiday import GoodFriday, USFederalHolidayCalendar

import external


class _PublicationCalendar(USFederalHolidayCalendar):
    rules = [*USFederalHolidayCalendar.rules, GoodFriday]


# The extract lacks publication dates. Ten business days excluding federal
# holidays and Good Friday is a conservative delay beyond FINRA's schedule.
PUBLICATION_DELAY = pd.offsets.CustomBusinessDay(10, calendar=_PublicationCalendar())


def public_dates(raw: pd.DataFrame) -> pd.Series:
    """Explicit publication dates when supplied, otherwise a conservative lag."""
    settlement = pd.to_datetime(raw.settlement)
    release = {date: date + PUBLICATION_DELAY for date in settlement.dropna().unique()}
    dates = settlement.map(release)
    if "publication_date" in raw:
        supplied = pd.to_datetime(raw.publication_date)
        if (supplied < settlement).any():
            raise ValueError("Short-interest publication precedes settlement")
        dates = supplied.fillna(dates)
    return dates


def build(work_dir: pathlib.Path, panel_path: pathlib.Path) -> pathlib.Path:
    """Writes short_interest.parquet: permno, return_month, short_pct, days_to_cover."""
    output = work_dir / "short_interest" / "short_interest.parquet"
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = external.read("finra")
    panel = pd.read_parquet(
        panel_path,
        columns=["permno", "eom", "ticker", "ticker_name_reference_date", "shares"],
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
        months.append(rows[["permno", "return_month", "short_pct", "days_to_cover"]])
    table = pd.concat(months, ignore_index=True)
    table["return_month"] = table.return_month.dt.to_timestamp("M")
    table.to_parquet(output, index=False)
    return output


def _alphanumeric(symbols: pd.Series) -> pd.Series:
    return symbols.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
