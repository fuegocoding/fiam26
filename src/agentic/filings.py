"""Shared filing helpers for the active (Jev) event pipeline.

Availability convention (competition scope): a filing dated in calendar month t
is aggregated at month-end t and predicts the next-month target t+1. No
intramonth execution claim is made.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW_FILINGS = ROOT / "data" / "raw" / "8k_20150101_20260831_identified.parquet"

#: Columns needed by redaction + Jev state. Mirrors the archived council reader.
REDACTION_COLUMNS = [
    "document_id",
    "permno",
    "filing_date",
    "form_type",
    "items",
    "text",
    "ticker",
    "company_name",
    "cik",
    "cusip",
    "cusip8",
    "isin",
    "figi",
    "composite_figi",
    "share_class_figi",
    "gvkey",
    "iid",
    "gvkey_iid",
    "sector",
    "industry",
    "report_url",
    "is_amendment",
    "content_report_date",
]


def filing_month_eom(filing_date: str) -> str:
    """Month-end date (YYYY-MM-DD) of the filing's calendar month."""
    return str(pd.Timestamp(filing_date) + pd.offsets.MonthEnd(0))[:10]


def iter_filings(
    months: pd.PeriodIndex | None = None,
    limit: int | None = None,
    batch_size: int = 2000,
) -> Iterator[dict]:
    """Stream filing records in file order (full competition scope by default).

    Pass ``months`` to restrict to calendar months, ``limit`` for bounded
    pilots. Yields plain dict records with a string ``filing_date``.
    """
    yielded = 0
    for batch in pq.ParquetFile(RAW_FILINGS).iter_batches(
        batch_size=batch_size, columns=REDACTION_COLUMNS
    ):
        frame = batch.to_pandas()
        frame["filing_date"] = pd.to_datetime(frame["filing_date"])
        if months is not None:
            frame = frame[frame["filing_date"].dt.to_period("M").isin(months)]
        for record in frame.to_dict("records"):
            yield record
            yielded += 1
            if limit is not None and yielded >= limit:
                return
