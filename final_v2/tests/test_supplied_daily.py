import csv, datetime as dt, json
from decimal import Decimal
import pytest
from pod_platform.reporting.supplied_daily import (
    read_supplied_daily,
    inputs_from_json,
    write_json,
)
from pod_platform.reporting.daily import (
    DailyInputs,
    DailyPolicy,
    DailyRate,
    DailyReturn,
    reconcile_daily,
)


def put(p, rows, header=("permno", "date", "prc", "daily_return")):
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_source_return_and_reference_price(tmp_path):
    p = tmp_path / "x.csv"
    put(p, [(1, "2026-01-02", 10, 0.1), (1, "2026-01-05", 5, 0.02)])
    x = read_supplied_daily(p, ["1"], "2026-01-02", "2026-01-05")
    assert [r.total_return for r in x.returns] == [
        Decimal(".1"),
        Decimal(".02"),
    ]
    assert x.reference_prices[("1", dt.date(2026, 1, 5))] == 5
    assert "not an exchange calendar" in x.metadata()["calendar_scope"]


def test_schema_missing_duplicate_and_cutoff(tmp_path):
    p = tmp_path / "x.csv"
    put(p, [(1, "2026-01-02", 10, 0.1)], ("permno", "date", "prc", "ret"))
    with pytest.raises(ValueError, match="schema mismatch"):
        read_supplied_daily(p, ["1"], "2026-01-01", "2026-01-05")
    put(p, [(1, "2026-01-02", 10, ""), (1, "2026-01-03", 10, 0.1)])
    with pytest.raises(ValueError, match="missing daily_return"):
        read_supplied_daily(p, ["1"], "2026-01-01", "2026-01-05")
    put(p, [(1, "2026-01-02", 10, 0.1), (1, "2026-01-02", 10, 0.1)])
    with pytest.raises(ValueError, match="duplicate"):
        read_supplied_daily(p, ["1"], "2026-01-01", "2026-01-05")
    put(p, [(1, "2026-09-01", 10, 0.1)])
    with pytest.raises(ValueError, match="descriptive=True"):
        read_supplied_daily(p, ["1"], "2026-09-01", "2026-09-01")
    assert read_supplied_daily(
        p, ["1"], "2026-09-01", "2026-09-01", descriptive=True
    )


def test_explicit_ledger_hand_calculation(tmp_path):
    p = tmp_path / "x.csv"
    put(p, [(1, "2026-01-02", 11, 0.1), (1, "2026-01-05", 12, 0.1)])
    x = read_supplied_daily(p, ["1"], "2026-01-02", "2026-01-05")
    data = {
        "start_cash_usd": "0",
        "initial_positions": {"1": ["1", "10"]},
        "risk_free": [
            {"on": "2026-01-02", "tb3ms_annual_percent": 0},
            {"on": "2026-01-05", "tb3ms_annual_percent": 0},
        ],
        "fills": [],
    }
    inputs, policy = inputs_from_json(data, x)
    report = reconcile_daily(inputs, policy)
    assert Decimal(report.rows[-1]["nav"]) == Decimal("12.100")


def test_output_overwrite_policy(tmp_path):
    p = tmp_path / "out.json"
    write_json(p, {"n": 1})
    with pytest.raises(FileExistsError):
        write_json(p, {"n": 2})
    write_json(p, {"n": 2}, overwrite=True)
    assert json.loads(p.read_text()) == {"n": 2}


def test_calendar_includes_other_security_dates_and_exposes_missing_held_outcome(
    tmp_path,
):
    p = tmp_path / "x.csv"
    put(
        p,
        [
            (1, "2026-01-02", 10, 0.1),
            (2, "2026-01-05", 10, 0.1),
            (1, "2026-01-06", 11, 0.1),
        ],
    )
    x = read_supplied_daily(p, ["1"], "2026-01-02", "2026-01-06")
    assert x.trading_days == [
        dt.date(2026, 1, 2),
        dt.date(2026, 1, 5),
        dt.date(2026, 1, 6),
    ]
    data = {
        "start_cash_usd": "0",
        "initial_positions": {"1": ["1", "10"]},
        "risk_free": [
            {"on": d.isoformat(), "tb3ms_annual_percent": 0}
            for d in x.trading_days
        ],
        "fills": [],
    }
    inputs, policy = inputs_from_json(data, x)
    with pytest.raises(Exception, match="no supplied outcome"):
        reconcile_daily(inputs, policy)


def test_later_filled_name_keeps_selected_source_returns(tmp_path):
    p = tmp_path / "x.csv"
    put(
        p,
        [
            (1, "2026-01-02", 10, 0),
            (2, "2026-01-02", 20, 0),
            (2, "2026-01-05", 22, 0.1),
        ],
    )
    x = read_supplied_daily(p, ["1", "2"], "2026-01-02", "2026-01-05")
    data = {
        "start_cash_usd": "100",
        "initial_positions": {"1": ["0", "10"]},
        "risk_free": [
            {"on": d.isoformat(), "tb3ms_annual_percent": 0}
            for d in x.trading_days
        ],
        "fills": [
            {
                "security_id": "2",
                "on": "2026-01-02",
                "quantity": "1",
                "price": "20",
                "fee_usd": "0",
            }
        ],
    }
    inputs, policy = inputs_from_json(data, x)
    assert any(
        r.security_id == "2" and r.on == dt.date(2026, 1, 5)
        for r in inputs.returns
    )


def test_negative_price_and_extra_fields_rejected(tmp_path):
    p = tmp_path / "x.csv"
    put(p, [(1, "2026-01-02", -10, 0.1)])
    with pytest.raises(ValueError, match="invalid prc"):
        read_supplied_daily(p, ["1"], "2026-01-02", "2026-01-02")
    put(p, [(1, "2026-01-02", 10, 0.1, "surplus")])
    with pytest.raises(ValueError, match="extra fields"):
        read_supplied_daily(p, ["1"], "2026-01-02", "2026-01-02")


def test_cash_only_inception_fill_accrues_stock_return_next_day(tmp_path):
    p = tmp_path / "x.csv"
    put(p, [(2, "2026-01-02", 10, 0.5), (2, "2026-01-05", 11, 0.1)])
    x = read_supplied_daily(p, ["2"], "2026-01-02", "2026-01-05")
    data = {
        "start_cash_usd": "1000",
        "initial_positions": {},
        "risk_free": [
            {"on": d.isoformat(), "tb3ms_annual_percent": 0}
            for d in x.trading_days
        ],
        "fills": [
            {
                "security_id": "2",
                "on": "2026-01-02",
                "quantity": "10",
                "price": "10",
                "fee_usd": "0",
            }
        ],
    }
    inputs, policy = inputs_from_json(data, x)
    report = reconcile_daily(inputs, policy)
    assert Decimal(report.rows[0]["nav"]) == Decimal("1000")
    assert Decimal(report.rows[1]["nav"]) == Decimal("1010")


def test_compact_dates_and_terminal_total_loss_reconcile_long_and_short(
    tmp_path,
):
    p = tmp_path / "loss.csv"
    put(p, [(1, "20260102", 0, -1)])
    source = read_supplied_daily(p, ["1"], "2026-01-02", "2026-01-02")
    assert source.returns[0].total_return == Decimal("-1")
    day = dt.date(2026, 1, 2)
    tomorrow = dt.date(2026, 1, 5)
    for quantity, cash, expected in [
        ("10", "100", "100"),
        ("-10", "300", "300"),
    ]:
        report = reconcile_daily(
            DailyInputs(
                start_cash_usd=Decimal(cash),
                initial_positions={"1": (Decimal(quantity), Decimal("10"))},
                returns=source.returns
                + [
                    DailyReturn(
                        security_id="1", on=tomorrow, total_return=Decimal("0")
                    )
                ],
                risk_free=[
                    DailyRate(on=d, tb3ms_annual_percent=0)
                    for d in [day, tomorrow]
                ],
            ),
            DailyPolicy(trading_days=[day, tomorrow]),
        )
        assert Decimal(report.rows[0]["nav"]) == Decimal(expected)
        assert abs(Decimal(report.rows[0]["nav_residual"])) < Decimal("1e-12")
        assert report.summary["zero_marked_positions"] == [
            {"security_id": "1", "quantity": quantity, "mark": "0"}
        ]
    # A zero-value position stays explicit; absence tomorrow is not silently filled.
    with pytest.raises(Exception, match="no supplied outcome"):
        reconcile_daily(
            DailyInputs(
                start_cash_usd=Decimal("100"),
                initial_positions={"1": (Decimal("10"), Decimal("10"))},
                returns=source.returns,
                risk_free=[
                    DailyRate(on=d, tb3ms_annual_percent=0)
                    for d in [day, tomorrow]
                ],
            ),
            DailyPolicy(trading_days=[day, tomorrow]),
        )
    with pytest.raises(ValueError):
        DailyReturn(security_id="1", on=day, total_return=Decimal("-1.001"))


def test_zero_reference_price_requires_total_loss_evidence(tmp_path):
    p = tmp_path / "zero.csv"
    put(p, [(1, "20260102", 0, 0)])
    with pytest.raises(ValueError, match="zero reference prc"):
        read_supplied_daily(p, ["1"], "2026-01-02", "2026-01-02")
