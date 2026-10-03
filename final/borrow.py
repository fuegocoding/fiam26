"""FINRA short interest as a borrow-availability screen for the short leg.

FINRA publishes every U.S. equity's short interest twice a month
(cdn.finra.org/equity/otcmarket/biweekly/shrtYYYYMMDD.csv).  Short shares as a
share of shares outstanding is the standard proxy for how scarce a borrow is:
a stock with 15% or more of its shares already short cannot be shorted at the
size or fee the cost model assumes.  `build` turns the files into one
point-in-time table and `optimizer.py` keeps such stocks out of the shorts.

Point-in-time rule: a settlement date is public about eight business days
later, so it is used from 12 calendar days after settlement.  A month's table
row uses the newest settlement public by that month's end.  Tickers come from
the panel, carried forward per stock so the 2026 gap in the supplied ticker
column does not drop stocks.
"""

import pathlib

import numpy as np
import pandas as pd

import external

PUBLICATION_LAG_DAYS = 12


def build(work_dir: pathlib.Path, panel_path: pathlib.Path) -> pathlib.Path:
    """Writes short_interest.parquet: permno, return_month, short_pct, days_to_cover."""
    output = work_dir / "short_interest" / "short_interest.parquet"
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = external.read("finra")
    panel = pd.read_parquet(panel_path, columns=["permno", "eom", "ticker", "shares"])
    panel["eom"] = pd.to_datetime(panel.eom)
    panel = panel.sort_values(["permno", "eom"])
    panel["ticker"] = panel.groupby("permno").ticker.ffill()
    panel["symbol"] = _alphanumeric(panel.ticker.astype("string"))
    shared = panel.groupby(["eom", "symbol"]).permno.transform("nunique") > 1
    panel = panel.loc[
        ~shared & panel.symbol.notna(), ["permno", "eom", "symbol", "shares"]
    ]
    raw["symbol"] = _alphanumeric(raw.symbol)
    raw["public"] = raw.settlement + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
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
