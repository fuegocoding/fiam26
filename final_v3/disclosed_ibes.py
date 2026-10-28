"""Disclosed I/B/E/S exception exclusively for conf_lm_gated.

Only consensus, snapshot prices and dated ticker links are accessible here.
"""
import json
import pandas as pd
import pyarrow.parquet as pq
import config

FILE = config.FINAL.parent / "external_data" / "external.parquet"
ALLOWED_TABLES = frozenset(("ibes_consensus", "ibes_price", "ibes_link"))
_MAX_LINK_SCORE = 2


def read(table: str, columns: list[str] | None = None, *, months=None) -> pd.DataFrame:
    """Read only the disclosed tables; optional whole-month snapshot pruning."""
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Undisclosed external table prohibited: {table}")
    spec = json.loads(pq.read_schema(FILE).metadata[b"tables"])[table]
    wanted = [c for c in spec["columns"] if columns is None or c in columns]
    filters = [("table", "==", table)]
    if months is not None and table != "ibes_link":
        # Stored STATPERS is ISO text; include complete months before last-snapshot selection.
        filters = [[("table", "==", table),
                    ("STATPERS", ">=", str(month.start_time.date())),
                    ("STATPERS", "<", str((month + 1).start_time.date()))]
                   for month in sorted(set(months))]
    frame = pd.read_parquet(FILE, columns=wanted, filters=filters)
    return frame.astype({c: spec["dtypes"][c] for c in wanted
                         if str(frame[c].dtype) != spec["dtypes"][c]})


def _unique_records(frame: pd.DataFrame, keys: list[str], source: str) -> pd.DataFrame:
    """Identical duplicates are harmless; conflicting undated revisions fail."""
    frame = frame.drop_duplicates()
    if frame.duplicated(keys).any():
        raise ValueError(f"Conflicting {source} records without revision timestamps")
    return frame


def month_end(values: pd.Series) -> pd.Series:
    """Rolls dates forward to their calendar month-end."""
    return pd.to_datetime(values) + pd.offsets.MonthEnd(0)


def load_link() -> pd.DataFrame:
    """Dated ticker-permno links, restricted to their supplied validity dates."""
    link = read("ibes_link")
    link = link.dropna(subset=["PERMNO"])
    link = link[link.SCORE <= _MAX_LINK_SCORE].copy()
    link["permno"] = link.PERMNO.astype(int)
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    return link[["TICKER", "permno", "sdate", "edate", "SCORE"]]


def load_consensus(months=None) -> pd.DataFrame:
    """US-dollar consensus rows from each ticker's last snapshot per month."""
    frame = read("ibes_consensus", months=months)
    frame = frame[(frame.USFIRM == 1) & (frame.CURCODE == "USD")].drop(
        columns=["USFIRM", "CURCODE"]
    )
    for column in ("STATPERS", "FPEDATS", "ANNDATS_ACT"):
        frame[column] = pd.to_datetime(frame[column])
    frame = _unique_records(frame, ["TICKER", "STATPERS", "FPEDATS"], "consensus")
    frame["snap_month"] = frame.STATPERS.dt.to_period("M")
    last = frame.groupby(["TICKER", "snap_month"]).STATPERS.transform("max")
    return frame[frame.STATPERS == last].reset_index(drop=True)


def load_prices(months=None) -> pd.DataFrame:
    """I/B/E/S share price of each snapshot, on the per-share EPS basis."""
    frame = read("ibes_price", months=months)
    frame = frame[(frame.USFIRM == 1) & (frame.CURR_PRICE == "USD") & (frame.PRICE > 0)]
    frame["STATPERS"] = pd.to_datetime(frame.STATPERS)
    frame = frame[["TICKER", "STATPERS", "PRICE"]]
    return _unique_records(frame, ["TICKER", "STATPERS"], "price")


def attach_ticker(grid: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Best dated I/B/E/S ticker for every (permno, eom)."""
    merged = grid[["permno", "eom"]].merge(link, on="permno", how="inner")
    merged = merged[(merged.sdate <= merged.eom) & (merged.eom <= merged.edate)]
    merged = merged.sort_values(
        ["permno", "eom", "SCORE", "sdate"], ascending=[True, True, True, False]
    )
    return merged.drop_duplicates(["permno", "eom"])[["permno", "eom", "TICKER"]]


