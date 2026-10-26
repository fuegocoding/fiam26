"""Cash-only completion path for configured strategy runs."""

import csv
import json
from decimal import Decimal

import pytest

from pod_platform.application.market_run import finish_research_run


def _calendar(path, dates):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["permno", "date", "prc", "daily_return"])
        for date in dates:
            writer.writerow([1, date, 10, 0])


def test_cash_only_run_uses_observed_calendar_and_reports_assumed_rate(
    tmp_path,
):
    source = tmp_path / "daily.csv"
    _calendar(source, ["20260102", "20260105", "20260105", "20260106"])
    output = tmp_path / "output"

    result = finish_research_run(
        {
            "calendar_csv": source,
            "start_cash_usd": "1000",
            "risk_free_annual_percent": "0",
        },
        output,
    )

    report = result["daily_report"]
    assert result["status"] == "complete_no_admissions"
    assert result["orders"] == 0
    assert result["allocation"]["positions"] == []
    assert report["summary"]["days"] == 3
    assert Decimal(report["summary"]["start_nav"]) == Decimal("1000")
    assert Decimal(report["summary"]["end_nav"]) == Decimal("1000")
    assert result["market_inputs"]["risk_free_rate"]["observed_tb3ms"] is False
    assert json.loads((output / "market_run.json").read_text()) == result


def test_cash_accrual_uses_configured_constant_and_date_bounds(tmp_path):
    source = tmp_path / "daily.csv"
    _calendar(source, ["20260102", "20260105", "20260106", "20260107"])

    result = finish_research_run(
        {
            "calendar_csv": source,
            "start_cash_usd": "1000",
            "risk_free_annual_percent": "5.04",
            "start_date": "2026-01-05",
            "end_date": "20260106",
        },
        tmp_path / "output",
    )

    rows = result["daily_report"]["rows"]
    assert [row["date"] for row in rows] == ["2026-01-05", "2026-01-06"]
    assert Decimal(rows[0]["nav"]) == Decimal("1000.2")
    assert Decimal(rows[1]["nav"]) == Decimal("1000.40004")


def test_cash_only_run_fails_closed_on_missing_rate_and_admissions(tmp_path):
    source = tmp_path / "daily.csv"
    _calendar(source, ["20260102", "20260105"])
    base = {"calendar_csv": source, "start_cash_usd": "1000"}

    with pytest.raises(ValueError, match="risk_free_annual_percent"):
        finish_research_run(base, tmp_path / "output")
    with pytest.raises(ValueError, match="ControlledReplay"):
        finish_research_run(
            {**base, "risk_free_annual_percent": 0},
            tmp_path / "output",
            admitted_forecasts=[{"forecast": "real"}],
        )
