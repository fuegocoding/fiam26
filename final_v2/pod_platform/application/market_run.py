"""Complete cash-only market runs when no strategy is admitted.

This adapter uses the existing daily reconciliation engine. It derives a
trading calendar from the dates present in a configured supplied CSV, then
records a cash-only book over that calendar. The configured cash rate is
reported as an assumption unless a caller supplies rates as actual inputs.
"""

from __future__ import annotations

import csv
import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..reporting.daily import (
    DailyInputs,
    DailyPolicy,
    DailyRate,
    reconcile_daily,
)
from .common import atomic_write_json, file_sha256


def _date(value: Any, name: str) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    try:
        if len(text) == 8 and text.isdigit():
            return dt.datetime.strptime(text, "%Y%m%d").date()
        return dt.date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD or YYYYMMDD") from exc


def _calendar(
    path: Path, start: dt.date | None, end: dt.date | None
) -> list[dt.date]:
    """Read only the source date column and return observed dates in range."""
    days: set[dt.date] = set()
    parsed_dates: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "date" not in reader.fieldnames:
            raise ValueError("calendar CSV must have a date column")
        for row_number, row in enumerate(reader, start=2):
            raw = (row.get("date") or "").strip()
            if raw in parsed_dates:
                continue
            try:
                day = _date(raw, "calendar date")
            except ValueError as exc:
                raise ValueError(
                    f"invalid calendar date on CSV row {row_number}: {raw!r}"
                ) from exc
            if day is None:
                raise ValueError(f"empty calendar date on CSV row {row_number}")
            parsed_dates.add(raw)
            if (start is None or day >= start) and (end is None or day <= end):
                days.add(day)
    result = sorted(days)
    if len(result) < 2:
        raise ValueError(
            "observed calendar range must contain at least two dates"
        )
    return result


def finish_research_run(
    settings: Mapping[str, Any],
    output_dir: str | Path,
    admitted_forecasts: Sequence[Any] = (),
) -> dict[str, Any]:
    """Write a reconciled cash-only report when no forecasts were admitted.

    Settings keys:
      * ``calendar_csv``: supplied CSV with a ``date`` column (compact dates
        are accepted).
      * ``start_cash_usd``: positive opening cash.
      * ``risk_free_annual_percent``: configured constant annual rate in
        percent. This is explicitly recorded as an assumption.
      * optional ``start_date`` and ``end_date`` bound selected source dates.

    Admitted forecasts require host-prepared, production-eligible actual market
    inputs and must be routed through the existing ControlledReplay adapter.
    This function intentionally does not create those inputs.
    """
    if admitted_forecasts:
        raise ValueError(
            "admitted forecasts require production-eligible actual market "
            "inputs through the ControlledReplay adapter"
        )

    required = {"calendar_csv", "start_cash_usd", "risk_free_annual_percent"}
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"market run settings require: {', '.join(missing)}")

    calendar_path = Path(str(settings["calendar_csv"]))
    if not calendar_path.is_file():
        raise ValueError(
            f"configured calendar CSV does not exist: {calendar_path}"
        )
    start = _date(settings.get("start_date"), "start_date")
    end = _date(settings.get("end_date"), "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date")
    trading_days = _calendar(calendar_path, start, end)

    opening_cash = Decimal(str(settings["start_cash_usd"]))
    annual_rate = Decimal(str(settings["risk_free_annual_percent"]))
    if not opening_cash.is_finite() or opening_cash <= 0:
        raise ValueError("start_cash_usd must be finite and positive")
    if not annual_rate.is_finite():
        raise ValueError("risk_free_annual_percent must be finite")

    inputs = DailyInputs(
        start_cash_usd=opening_cash,
        risk_free=[
            DailyRate(on=day, tb3ms_annual_percent=float(annual_rate))
            for day in trading_days
        ],
    )
    report = reconcile_daily(inputs, DailyPolicy(trading_days=trading_days))
    result = {
        "status": "complete_no_admissions",
        "orders": 0,
        "admitted_forecasts": 0,
        "allocation": {
            "status": "abstained_no_admitted_forecasts",
            "positions": [],
        },
        "rebalancing": {
            "status": "no_orders_without_admitted_forecasts",
            "orders": [],
        },
        "daily_report": report.as_dict(),
        "market_inputs": {
            "calendar_source": str(calendar_path.resolve()),
            "calendar_sha256": file_sha256(calendar_path),
            "calendar_basis": "dates observed in the configured CSV date column",
            "selected_range": {
                "start": trading_days[0].isoformat(),
                "end": trading_days[-1].isoformat(),
            },
            "trading_days": len(trading_days),
            "risk_free_rate": {
                "source": "configured_constant_assumption",
                "tb3ms_annual_percent": str(annual_rate),
                "observed_tb3ms": False,
            },
            "security_returns_used": False,
            "investment_evidence": False,
        },
        "limitations": [
            "No forecast was admitted; no positions or orders were created.",
            "Cash accrual uses the configured constant rate assumption, not an observed TB3MS series.",
            "The calendar is the set of dates present in the configured CSV, not an independently verified exchange calendar.",
        ],
    }
    atomic_write_json(Path(output_dir) / "market_run.json", result)
    return result
