"""Part 8 acceptance: reporting reconciliation (spec 26.7, 19.9-19.10)."""

from __future__ import annotations

import datetime as dt

import pytest

from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.contracts.decisions import ActorKind, DecisionEvent
from pod_platform.reporting import (
    DossierBuilder,
    ExhibitSeries,
    HoldingsRow,
    RequiredTable,
    holdings_csv,
    returns_csv,
    verify_deck_numbers,
)

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)


def _table(**over) -> RequiredTable:
    base = dict(
        ending_value=1_100_000.0,
        starting_value=1_000_000.0,
        elapsed_years=1.0,
        periods_per_year=12,
        portfolio_returns=(0.02, -0.01, 0.03, -0.005, 0.01, 0.025, -0.015, 0.02, 0.01, -0.005, 0.02, 0.015),
        benchmark_returns=(0.004,) * 12,
        risk_free_returns=(0.002,) * 12,
        gross_returns=(0.022, -0.008, 0.032, -0.003, 0.012, 0.027, -0.013, 0.022, 0.012, -0.003, 0.022, 0.017),
        cost_returns=(0.001,) * 12,
        borrow_returns=(0.0005,) * 12,
        position_counts=(300,) * 12,
        gross_exposures=(1.0,) * 12,
        net_exposures=(0.0,) * 12,
        turnover_per_rebalance=(0.25,) * 12,
    )
    base.update(over)
    return RequiredTable(**base)


def test_cumulative_return_matches_compounding():
    table = _table()
    compounded = 1.0
    for r in table.portfolio_returns:
        compounded *= 1.0 + r
    assert abs(table.cumulative_return() - (compounded - 1.0)) < 1e-12


def test_cagr_and_arithmetic_reported_separately():
    table = _table()
    assert abs(table.cagr() - (table.ending_value / table.starting_value - 1.0)) < 1e-9
    assert abs(table.annualized_arithmetic_return() - 12 * sum(table.portfolio_returns) / 12) < 1e-9
    assert abs(table.cagr() - table.annualized_arithmetic_return()) > 1e-9
    result = table.as_table()
    assert "cagr_convention" in result
    assert "compounded" in result["cagr_convention"]


def test_information_ratio_cost_basis_label():
    table = _table()
    result = table.as_table()
    assert "net of modeled" in result["information_ratio_cost_basis"]
    assert result["information_ratio"] > 0


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        _table(benchmark_returns=(0.004,) * 5).cumulative_return()


def test_turnover_annualization_and_position_stats():
    table = _table()
    assert abs(table.turnover_per_rebalance_avg() - 0.25) < 1e-12
    assert abs(table.turnover_annualized() - 3.0) < 1e-12
    assert table.position_count_stats() == {"average": 300.0, "minimum": 300.0, "maximum": 300.0}
    assert table.gross_exposure_stats()["maximum"] == 1.0
    assert table.net_exposure_stats()["average"] == 0.0


def test_max_drawdown_and_max_loss():
    table = _table()
    assert table.max_drawdown() <= 0.0
    assert table.max_period_loss() <= 0.0


def test_exhibit_conventions_differ_and_are_labeled():
    series = ExhibitSeries()
    p = [0.02, -0.01, 0.03]
    b = [0.004, 0.004, 0.004]
    compounded = series.cumulative_active(p, b)
    relative = series.relative_wealth(p, b)
    assert compounded[-1].value != relative[-1].value
    assert series.convention == "compounded_active_increments"
    with pytest.raises(ValueError):
        series.cumulative_active(p, b[:2])


def test_holdings_export_requires_identity_provenance():
    good = HoldingsRow(
        security_id="sec_1",
        ticker="AAA",
        name="Acme",
        label_source="crsp_status",
        label_reference_date="2021-01-04",
        quantity=100.0,
        weight=0.01,
        side="long",
    )
    text = holdings_csv([good])
    assert "crsp_status" in text
    assert text.splitlines()[0].startswith("security_id")
    bad = HoldingsRow(
        security_id="sec_2",
        ticker="BBB",
        name="Beta",
        label_source="",
        label_reference_date="",
        quantity=100.0,
        weight=0.01,
        side="long",
    )
    with pytest.raises(PodPlatformError) as exc:
        holdings_csv([bad])
    assert exc.value.response.error.code == ErrorCode.IDENTITY_AMBIGUOUS


def test_returns_export():
    text = returns_csv([("2021-01", 0.02), ("2021-02", -0.01)])
    lines = text.strip().splitlines()
    assert lines[0] == "period,net_return"
    assert len(lines) == 3


def test_deck_numbers_must_reference_artifacts():
    verify_deck_numbers({"ir": 0.36}, {"ir": 0.36})
    with pytest.raises(PodPlatformError) as exc:
        verify_deck_numbers({"ir": 0.36}, {"sharpe": 1.27})
    assert exc.value.response.error.code == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED
    with pytest.raises(PodPlatformError):
        verify_deck_numbers({"ir": 0.36}, {"ir": 0.40})


def test_dossier_explains_position_from_stored_evidence():
    builder = DossierBuilder()
    event = DecisionEvent(
        owner="integrity",
        created_at=T0,
        event_id="ev_1",
        actor_kind=ActorKind.CODE,
        actor_id="router",
        effective_research_at=T0,
        effective_market_at=T0,
        input_artifact_ids=["sec_1"],
        input_hashes={"sec_1": "abc"},
        chosen_action="buy sec_1 forecast_forecast_1",
        rationale="signal above threshold",
        authorizing_rule_or_certificate_id="cert_1",
        resulting_artifact_id="order_order_1",
    )
    builder.record(event)
    dossier = builder.build("sec_1", T0)
    explanation = dossier.explanation()
    assert explanation["security_id"] == "sec_1"
    assert len(explanation["decision_events"]) == 1
    assert explanation["decision_events"][0]["authorizing_rule_or_certificate_id"] == "cert_1"
    assert explanation["order_ids"] == ["order_order_1"]
    assert explanation["evidence_ids"] == ["sec_1", "order_order_1"]


def test_dossier_unexplainable_position_rejected():
    builder = DossierBuilder()
    with pytest.raises(PodPlatformError) as exc:
        builder.build("sec_unknown", T0)
    assert exc.value.response.error.code == ErrorCode.SCHEMA_MISMATCH
