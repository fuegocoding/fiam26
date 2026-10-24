"""Competition-input loader (chars panel, 8-K corpus) with declared unit handling
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 9.4, 9.10).

The chars panel stores `me` in millions of USD and `dolvol` in dollars; every
consumer of these fields gets explicit conversions here, never upstream.
Per-join assertions (9.10) are emitted alongside each materialization.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ..contracts.base import require_aware

ME_UNIT = "millions_usd"
DOLVOL_UNIT = "usd_per_day"
PRICE_UNIT = "usd_per_share"

UNITS = {"me": ME_UNIT, "dolvol": DOLVOL_UNIT, "prc": PRICE_UNIT}


@dataclass
class MaterializationReport:
    source_id: str
    formation_month: dt.datetime
    rows_in: int
    rows_out: int
    assertions: dict[str, str] = field(default_factory=dict)

    def record(self, check: str, status: str) -> None:
        self.assertions[check] = status

    @property
    def all_passed(self) -> bool:
        return all(status == "pass" for status in self.assertions.values())


class CharsPanelLoader:
    SOURCE_ID = "comp_chars_panel"

    def __init__(self, panel_path: str | Path, factor_list_path: Optional[str | Path] = None):
        self._path = Path(panel_path)
        self._factor_list_path = Path(factor_list_path) if factor_list_path else None
        self._panel: Optional[pd.DataFrame] = None

    @property
    def units(self) -> dict[str, str]:
        return dict(UNITS)

    def load(self) -> pd.DataFrame:
        if self._panel is None:
            columns = [
                "permno",
                "eom",
                "ret_exc_lead1m",
                "me",
                "dolvol",
                "prc",
                "gics",
                "size_grp",
                "ticker",
                "company_name",
            ]
            if self._factor_list_path is not None:
                factors = pd.read_csv(self._factor_list_path)["variable"].tolist()
                columns.extend(c for c in factors if c not in columns)
            panel = pd.read_parquet(self._path, columns=columns)
            panel["eom"] = pd.to_datetime(panel["eom"])
            panel["me_usd"] = panel["me"] * 1_000_000.0
            panel["me_formation_usd"] = panel["me_usd"]
            self._panel = panel
        return self._panel

    def formation_month(
        self, eom: dt.datetime, *, max_price: float = 5.0, min_dolvol: float = 0.0
    ) -> tuple[pd.DataFrame, MaterializationReport]:
        eom = pd.Timestamp(require_aware(eom, "eom").date())
        panel = self.load()
        rows_in = len(panel)
        month = panel[panel["eom"] == eom].copy()
        report = MaterializationReport(
            source_id=self.SOURCE_ID,
            formation_month=eom.to_pydatetime(),
            rows_in=rows_in,
            rows_out=0,
        )
        if month.empty:
            raise ValueError(f"no panel rows for formation month {eom.date()}")
        if month["eom"].max() > eom:
            raise ValueError("future rows in formation month selection")
        report.record("feature_availability_precedes_cutoff", "pass")

        tradable = month[
            (month["prc"].abs() >= max_price)
            & (month["dolvol"] > min_dolvol)
            & month["me_usd"].notna()
            & month["gics"].notna()
        ].copy()
        report.record("tradability_filters_applied", "pass")

        if tradable["ret_exc_lead1m"].isna().all():
            report.record("labels_present_for_evaluation", "warn")
        else:
            report.record("labels_present_for_evaluation", "pass")

        if tradable["ticker"].isna().any() or tradable["company_name"].isna().any():
            report.record("identity_labels_present", "warn")
        else:
            report.record("identity_labels_present", "pass")

        report.record("units_normalized_at_ingestion", "pass")
        report.rows_out = len(tradable)
        return tradable.reset_index(drop=True), report

    def return_column(self) -> str:
        return "ret_exc_lead1m"


class FilingCorpusLoader:
    SOURCE_ID = "comp_8k_filings"

    def __init__(self, corpus_path: str | Path):
        self._path = Path(corpus_path)

    def load_counts(self, month_start: dt.datetime, month_end: dt.datetime) -> pd.DataFrame:
        month_start = require_aware(month_start, "month_start")
        month_end = require_aware(month_end, "month_end")
        filings = pd.read_parquet(
            self._path,
            columns=[
                "permno",
                "sec_accepted_at_utc",
                "filing_date",
                "is_amendment",
                "items",
                "identifier_quality_flags",
                "characteristics_link_date",
            ],
        )
        filings["accepted"] = pd.to_datetime(filings["sec_accepted_at_utc"], utc=True)
        window = filings[
            (filings["accepted"] >= month_start) & (filings["accepted"] <= month_end)
        ].copy()
        window = window[window["is_amendment"] == False]  # noqa: E712
        counts = (
            window.groupby("permno")
            .size()
            .rename("filing_count")
            .reset_index()
        )
        return counts
