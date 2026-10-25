"""Offline regression tests for the replay control layer and the controlled runner."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from pod_platform.accounting.ledger import BorrowSchedule
from pod_platform.allocation import OfficialLimits
from pod_platform.contracts.errors import ErrorCode, PodPlatformError
from pod_platform.contracts.forecasts import (
    Forecast,
    ReturnForecast,
    TargetPrediction,
)
from pod_platform.contracts.sources import (
    AssessmentStatus,
    CardStatus,
    Screen,
    ScreenItem,
    SourceCard,
)
from pod_platform.orchestration.replay import (
    LEGACY_FAMILY_SOURCES,
    ArtifactWriter,
    ControlledReplay,
    FreezeAssessment,
    LiquidationEvidence,
    MonthlyAccountingReplay,
    PeriodComplianceLog,
    SourceEligibilityGate,
    SourceUse,
    account_period,
    assess_freeze,
    legacy_source_uses,
    load_fixture,
    prepare_output_dir,
    resolve_outcomes,
    utc_close,
    validate_contiguous_periods,
)
from pod_platform.research.promotion import AdmissionCertificate

UTC = dt.timezone.utc
T0 = dt.datetime(2023, 12, 31, tzinfo=UTC)
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def code_of(excinfo) -> ErrorCode:
    return excinfo.value.response.error.code


# ------------------------------------------------------------------ helpers


def _card(source_id: str, *, eligible: bool = True) -> SourceCard:
    status = (
        AssessmentStatus.ESTABLISHED
        if eligible
        else AssessmentStatus.SAMPLE_ONLY
    )
    item = ScreenItem(status=status, evidence_id="ev_1" if eligible else None)
    return SourceCard(
        owner="data",
        created_at=T0,
        source_id=source_id,
        dataset_id=f"ds_{source_id}",
        topic="test",
        snapshot_id=f"snap_{source_id}_2024_01",
        status=CardStatus.APPROVED if eligible else CardStatus.SAMPLED,
        access_confirmed=eligible,
        formation_eligible=eligible,
        entity_grain=["security"],
        screen=Screen(
            history=item, metric_lag=item, vintage=item, coverage=item
        ),
        unresolved_questions=[] if eligible else ["vintage unknown"],
    )


def _limits() -> OfficialLimits:
    return OfficialLimits(
        min_positions=1,
        max_positions=50,
        max_abs_name_weight=0.6,
        min_net=-0.9,
        max_net=0.9,
        min_short_market_cap_usd=0,
    )


def _caps(*sids) -> dict:
    return {s: Decimal("1000000000") for s in sids}


def _replay(nav="1000000") -> MonthlyAccountingReplay:
    return MonthlyAccountingReplay(
        Decimal(nav),
        borrow_schedule=BorrowSchedule(
            "flat", Decimal("0"), Decimal("0.05"), Decimal("0.05")
        ),
    )


FORMATIONS = [
    dt.date(2024, 1, 31),
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 31),
    dt.date(2024, 4, 30),
]
ENDS = [
    dt.date(2024, 2, 29),
    dt.date(2024, 3, 31),
    dt.date(2024, 4, 30),
    dt.date(2024, 5, 31),
]


def _run_periods(
    count=3, *, missing=None, post_risk_override=None, liquidations=None
):
    replay = _replay()
    log = PeriodComplianceLog(_limits())
    target = {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")}
    results = []
    for i in range(count):
        outcomes = {"AAA": 0.02, "BBB": -0.01}
        if missing == i:
            outcomes["AAA"] = float("nan")
        post = dict(target)
        if post_risk_override and i in post_risk_override:
            post = post_risk_override[i]
        results.append(
            account_period(
                replay,
                log,
                period=FORMATIONS[i].isoformat(),
                formation_at=utc_close(FORMATIONS[i]),
                end_at=utc_close(ENDS[i]),
                construction_weights=target,
                post_risk_weights=post,
                outcomes=outcomes,
                liquidations=(liquidations or {}).get(i),
                rf_monthly=0.003,
                cost_bp_per_side=10,
                market_caps=_caps("AAA", "BBB"),
            )
        )
    return replay, log, results


# ------------------------------------------------------------ source gate


def test_gate_rejects_missing_and_ineligible_cards():
    gate = SourceEligibilityGate()
    ok = _card("src_ok")
    bad = _card("src_bad", eligible=False)
    with pytest.raises(PodPlatformError) as err:
        gate.evaluate(
            [
                SourceUse("fam", "src_ok", "core", ok),
                SourceUse("fam", "src_bad", "external", bad),
            ]
        )
    assert code_of(err) == ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE
    assert "src_bad" in str(err.value)
    with pytest.raises(PodPlatformError):
        gate.evaluate([SourceUse("fam", "src_nocard", "external", None)])
    with pytest.raises(
        PodPlatformError
    ):  # card for a different source id is not accepted
        gate.evaluate([SourceUse("fam", "other", "external", ok)])
    passed = gate.evaluate([SourceUse("fam", "src_ok", "core", ok)])
    assert (
        passed.freeze_claim_allowed and passed.run_label == "FORMATION_ELIGIBLE"
    )


@pytest.mark.parametrize("family", ["demand", "traded", "gate_e7_demand"])
def test_demand_traded_gate_families_blocked_without_cards(family):
    uses = legacy_source_uses([family], {})
    with pytest.raises(PodPlatformError) as err:
        SourceEligibilityGate().evaluate(uses)
    assert code_of(err) == ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE
    uses = legacy_source_uses(
        [family],
        {s: _card(s, eligible=False) for s in LEGACY_FAMILY_SOURCES[family]},
    )
    with pytest.raises(PodPlatformError):
        SourceEligibilityGate().evaluate(uses)


def test_research_only_is_labeled_and_never_claims_freeze():
    uses = legacy_source_uses(["demand"], {})
    result = SourceEligibilityGate(research_only=True).evaluate(uses)
    assert result.run_label == "RESEARCH_ONLY_NOT_FREEZABLE"
    assert result.blocked == ["demand_card_foot_traffic"]
    assert not result.freeze_claim_allowed
    # even with every source approved, research-only cannot claim freeze
    approved = SourceEligibilityGate(research_only=True).evaluate(
        [SourceUse("fam", "src_ok", "core", _card("src_ok"))]
    )
    assert (
        approved.formation_inputs_approved and not approved.freeze_claim_allowed
    )


def test_unknown_family_has_no_lineage():
    with pytest.raises(PodPlatformError):
        legacy_source_uses(["mystery_family"], {})


# ---------------------------------------------------------------- outcomes


def _liq(sid="AAA", when=dt.datetime(2024, 2, 15, tzinfo=UTC), value=-0.3):
    return LiquidationEvidence(
        security_id=sid,
        liquidated_at=when,
        realized_excess_return=value,
        evidence_id="ev_delist_1",
        source="delisting file",
    )


def test_missing_outcome_is_never_zero_filled():
    f, e = utc_close(FORMATIONS[0]), utc_close(ENDS[0])
    with pytest.raises(PodPlatformError) as err:
        resolve_outcomes(
            ["AAA", "BBB"], {"AAA": 0.01}, formation_at=f, period_end_at=e
        )
    assert code_of(
        err
    ) == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED and "BBB" in str(err.value)
    with pytest.raises(PodPlatformError):
        resolve_outcomes(
            ["AAA"], {"AAA": float("nan")}, formation_at=f, period_end_at=e
        )
    with pytest.raises(PodPlatformError):
        resolve_outcomes(
            ["AAA"], {"AAA": None}, formation_at=f, period_end_at=e
        )
    ok = resolve_outcomes(
        ["AAA"], {"AAA": 0.0}, formation_at=f, period_end_at=e
    )  # a genuine 0.0 is fine
    assert ok.returns == {"AAA": 0.0}


def test_liquidation_evidence_must_be_dated_inside_period():
    f, e = utc_close(FORMATIONS[0]), utc_close(ENDS[0])
    resolved = resolve_outcomes(
        ["AAA"], {}, {"AAA": _liq()}, formation_at=f, period_end_at=e
    )
    assert resolved.basis["AAA"] == "liquidation:ev_delist_1"
    for when in (
        dt.datetime(2024, 3, 5, tzinfo=UTC),
        dt.datetime(2024, 1, 15, tzinfo=UTC),
    ):
        with pytest.raises(PodPlatformError) as err:
            resolve_outcomes(
                ["AAA"],
                {},
                {"AAA": _liq(when=when)},
                formation_at=f,
                period_end_at=e,
            )
        assert code_of(err) == ErrorCode.ASOF_VIOLATION
    with pytest.raises(ValidationError):  # naive timestamps are rejected
        _liq(when=dt.datetime(2024, 2, 15))


def test_periods_must_abut():
    validate_contiguous_periods(list(zip(FORMATIONS, ENDS)))
    with pytest.raises(PodPlatformError) as err:
        validate_contiguous_periods(
            [(FORMATIONS[0], ENDS[0]), (FORMATIONS[2], ENDS[2])]
        )
    assert code_of(err) == ErrorCode.ASOF_VIOLATION


@pytest.mark.parametrize(
    "missing,expected_records", [(0, 3), (1, 4 + 3), (2, 8 + 3)]
)
def test_missing_outcome_first_middle_last_month_fails_closed(
    missing, expected_records
):
    replay = _replay()
    log = PeriodComplianceLog(_limits())
    target = {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")}
    with pytest.raises(PodPlatformError) as err:
        for i in range(3):
            outcomes = {
                "AAA": float("nan") if i == missing else 0.02,
                "BBB": -0.01,
            }
            account_period(
                replay,
                log,
                period=FORMATIONS[i].isoformat(),
                formation_at=utc_close(FORMATIONS[i]),
                end_at=utc_close(ENDS[i]),
                construction_weights=target,
                post_risk_weights=target,
                outcomes=outcomes,
                liquidations=None,
                rf_monthly=0.003,
                cost_bp_per_side=10,
                market_caps=_caps("AAA", "BBB"),
            )
    assert code_of(err) == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED
    assert "AAA" in str(err.value) and ENDS[missing].isoformat() in str(
        err.value
    )
    # completed periods plus the three pre-outcome stages of the failing one are persisted
    assert len(log.records) == expected_records


def test_liquidation_evidence_allows_a_missing_month_to_proceed():
    replay, log, results = _run_periods(
        2,
        missing=1,
        liquidations={
            1: {
                "AAA": _liq(
                    when=dt.datetime(2024, 3, 10, tzinfo=UTC), value=-0.2
                )
            }
        },
    )
    assert results[1].result.liquidated == ["AAA"]
    assert "AAA" not in replay.held()  # proceeds went to cash, position closed
    assert results[1].resolved.basis["AAA"].startswith("liquidation:")


# -------------------------------------------------------------- accounting


def test_accounting_conservation_cash_borrow_cost():
    replay = _replay()
    trade = replay.rebalance_to_targets(
        {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")},
        at=utc_close(FORMATIONS[0]),
        cost_bp_per_side=Decimal("20"),
    )
    assert trade.fees == Decimal("2000")
    assert replay.ledger.trading_costs == Decimal("2000")
    assert trade.nav_post_trade == Decimal("998000")
    assert trade.turnover_one_way == pytest.approx(
        0.5
    )  # from flat: half of 1.0 traded
    resolved = resolve_outcomes(
        replay.held(),
        {"AAA": 0.10, "BBB": 0.02},
        formation_at=utc_close(FORMATIONS[0]),
        period_end_at=utc_close(ENDS[0]),
    )
    result = replay.advance(
        period_end_at=utc_close(ENDS[0]),
        rf_monthly=0.004,
        resolved=resolved,
        market_caps=_caps("AAA", "BBB"),
    )
    borrow = 500_000 * 0.05 * 21 / 252
    expected = 998_000 * 1.004 + 500_000 * (0.10 - 0.02) - borrow
    assert float(result.nav_end) == pytest.approx(expected, abs=1e-6)
    assert float(replay.ledger.borrow_costs) == pytest.approx(borrow, abs=1e-9)
    assert result.net_return == pytest.approx(
        expected / 1_000_000 - 1.0, abs=1e-12
    )
    assert result.cost_return == pytest.approx(
        0.002
    ) and result.borrow_return == pytest.approx(borrow / 1e6)
    assert result.gross_return == pytest.approx(
        result.net_return + result.cost_return + result.borrow_return
    )
    assert all(
        abs(Decimal(v)) < Decimal("1e-6")
        for v in result.reconciliation.values()
    )
    assert (
        replay.ledger.cash_income > 0
    )  # cash (incl. short proceeds) earned the risk-free rate


def test_tampered_ledger_fails_reconciliation():
    replay = _replay()
    replay.rebalance_to_targets(
        {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")},
        at=utc_close(FORMATIONS[0]),
        cost_bp_per_side=Decimal("0"),
    )
    replay.ledger.cash += Decimal("1")  # unexplained cash
    resolved = resolve_outcomes(
        replay.held(),
        {"AAA": 0.0, "BBB": 0.0},
        formation_at=utc_close(FORMATIONS[0]),
        period_end_at=utc_close(ENDS[0]),
    )
    with pytest.raises(PodPlatformError) as err:
        replay.advance(
            period_end_at=utc_close(ENDS[0]),
            rf_monthly=0.0,
            resolved=resolved,
            market_caps=_caps("AAA", "BBB"),
        )
    assert code_of(err) == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED


def test_turnover_uses_drifted_holdings_not_raw_prior_targets():
    replay = _replay()
    target = {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")}
    replay.rebalance_to_targets(
        target, at=utc_close(FORMATIONS[0]), cost_bp_per_side=Decimal("0")
    )
    resolved = resolve_outcomes(
        replay.held(),
        {"AAA": 0.10, "BBB": 0.0},
        formation_at=utc_close(FORMATIONS[0]),
        period_end_at=utc_close(ENDS[0]),
    )
    replay.advance(
        period_end_at=utc_close(ENDS[0]),
        rf_monthly=0.0,
        resolved=resolved,
        market_caps=_caps("AAA", "BBB"),
    )
    trade = replay.rebalance_to_targets(
        target, at=utc_close(FORMATIONS[1]), cost_bp_per_side=Decimal("0")
    )
    assert trade.pre_weights["AAA"] > Decimal("0.5")  # drifted up
    raw_prior_turnover = (
        0.0  # same target as last month would imply zero turnover
    )
    drift_turnover = 0.5 * sum(
        abs(float(target[k] - trade.pre_weights[k])) for k in target
    )
    assert raw_prior_turnover == 0.0 and drift_turnover > 0.01
    assert trade.turnover_one_way == pytest.approx(drift_turnover, rel=1e-9)
    assert float(replay.actual_weights()["AAA"]) == pytest.approx(
        0.5, abs=1e-3
    )  # rebalanced back to target


def test_held_name_union_is_exited_and_kept_in_accounting():
    replay = _replay()
    at0, at1 = utc_close(FORMATIONS[0]), utc_close(FORMATIONS[1])
    replay.rebalance_to_targets(
        {"AAA": Decimal("0.4"), "BBB": Decimal("-0.4"), "CCC": Decimal("0.2")},
        at=at0,
        cost_bp_per_side=Decimal("10"),
    )
    resolved = resolve_outcomes(
        replay.held(),
        {"AAA": 0.01, "BBB": 0.01, "CCC": 0.05},
        formation_at=at0,
        period_end_at=utc_close(ENDS[0]),
    )
    replay.advance(
        period_end_at=utc_close(ENDS[0]),
        rf_monthly=0.003,
        resolved=resolved,
        market_caps=_caps("AAA", "BBB", "CCC"),
    )
    assert replay.held() == {
        "AAA",
        "BBB",
        "CCC",
    }  # CCC stays in accounting although not a candidate next month
    trade = replay.rebalance_to_targets(
        {"AAA": Decimal("0.4"), "BBB": Decimal("-0.4")},
        at=at1,
        cost_bp_per_side=Decimal("10"),
    )
    assert "CCC" in {t["security_id"] for t in trade.trades}
    assert (
        "CCC" not in replay.held()
        and replay.ledger.positions["CCC"].quantity == 0
    )
    assert trade.turnover_one_way >= 0.5 * float(trade.pre_weights["CCC"])
    # a held name with no outcome in the next period still blocks the run
    resolved_missing = {"AAA": 0.01}
    with pytest.raises(PodPlatformError):
        resolve_outcomes(
            replay.held(),
            resolved_missing,
            formation_at=at1,
            period_end_at=utc_close(ENDS[1]),
        )


# -------------------------------------------------------------- compliance


def test_every_period_and_stage_recorded_with_pre_and_post_weights():
    _, log, _ = _run_periods(3)
    assert len(log.records) == 3 * 4
    assert log.periods() == [d.isoformat() for d in FORMATIONS[:3]]
    stages = {r["stage"] for r in log.records}
    assert stages == {
        "formation_target",
        "post_risk",
        "post_trade_actual",
        "period_end_drifted",
    }
    assert log.all_official_pass
    drift = [r for r in log.records if r["stage"] == "period_end_drifted"]
    assert drift[0]["pre_weights"]["AAA"] == "0.5" or Decimal(
        drift[0]["pre_weights"]["AAA"]
    ) > Decimal("0.49")
    # post-trade actual weights of period 2 start from drifted (not target) holdings
    second = [
        r
        for r in log.records
        if r["period"] == FORMATIONS[1].isoformat()
        and r["stage"] == "post_trade_actual"
    ][0]
    assert Decimal(second["pre_weights"]["AAA"]) != Decimal("0.5")
    assert all("pre_summary" in r and "post_summary" in r for r in log.records)


def test_intermediate_period_breach_after_overlay_fails_closed():
    breach = {
        "AAA": Decimal("0.7"),
        "BBB": Decimal("-0.5"),
    }  # official single-name cap in this test is 0.6
    replay = _replay()
    log = PeriodComplianceLog(_limits())
    target = {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")}
    with pytest.raises(PodPlatformError) as err:
        for i in range(3):
            account_period(
                replay,
                log,
                period=FORMATIONS[i].isoformat(),
                formation_at=utc_close(FORMATIONS[i]),
                end_at=utc_close(ENDS[i]),
                construction_weights=target,
                post_risk_weights=breach if i == 1 else target,
                outcomes={"AAA": 0.01, "BBB": 0.0},
                liquidations=None,
                rf_monthly=0.003,
                cost_bp_per_side=10,
                market_caps=_caps("AAA", "BBB"),
            )
    assert code_of(err) == ErrorCode.PORTFOLIO_INFEASIBLE
    assert FORMATIONS[1].isoformat() in str(err.value) and "post_risk" in str(
        err.value
    )
    last = log.records[-1]
    assert (
        last["failing_official"] == ["single_name"]
        and not log.all_official_pass
    )
    assert (
        last["pre_weights"]["AAA"] == "0.5"
        and last["post_weights"]["AAA"] == "0.7"
    )
    assert all(
        r["official_pass"] for r in log.records[:-1]
    )  # period one was clean, only the middle period failed
    assert len(log.records) == 4 + 2  # nothing after the breach was traded


def test_drifted_breach_between_rebalances_is_caught():
    replay = _replay()
    log = PeriodComplianceLog(
        OfficialLimits(
            min_positions=1,
            max_positions=50,
            max_abs_name_weight=0.52,
            min_short_market_cap_usd=0,
        )
    )
    target = {"AAA": Decimal("0.5"), "BBB": Decimal("-0.5")}
    with pytest.raises(PodPlatformError) as err:
        account_period(
            replay,
            log,
            period="2024-01-31",
            formation_at=utc_close(FORMATIONS[0]),
            end_at=utc_close(ENDS[0]),
            construction_weights=target,
            post_risk_weights=target,
            outcomes={"AAA": 0.20, "BBB": 0.0},
            liquidations=None,
            rf_monthly=0.0,
            cost_bp_per_side=0,
            market_caps=_caps("AAA", "BBB"),
        )
    assert "period_end_drifted" in str(err.value)
    assert log.records[-1]["stage"] == "period_end_drifted"


def test_internal_warnings_are_persisted_not_fatal():
    replay = _replay()
    log = PeriodComplianceLog(_limits())
    skewed = {
        "AAA": Decimal("0.5"),
        "BBB": Decimal("-0.2"),
    }  # net 0.3 > internal 0.10, inside official [-0.5, 0.5]
    account_period(
        replay,
        log,
        period="2024-01-31",
        formation_at=utc_close(FORMATIONS[0]),
        end_at=utc_close(ENDS[0]),
        construction_weights=skewed,
        post_risk_weights=skewed,
        outcomes={"AAA": 0.01, "BBB": 0.0},
        liquidations=None,
        rf_monthly=0.003,
        cost_bp_per_side=0,
        market_caps=_caps("AAA", "BBB"),
    )
    assert log.all_official_pass
    warned = log.internal_warnings()
    assert warned and all("abs_net_internal" in w["warnings"] for w in warned)
    assert json.loads(json.dumps(log.as_dict()))["internal_warnings"] == warned


# ------------------------------------------------------------ output files


def test_output_dir_requires_fresh_or_explicit_overwrite(tmp_path):
    out = tmp_path / "run"
    prepare_output_dir(out)
    (out / "keep.txt").write_text("precious")
    with pytest.raises(PodPlatformError):
        prepare_output_dir(out)
    with pytest.raises(PodPlatformError):
        ArtifactWriter(out)
    writer = ArtifactWriter(out, overwrite=True)
    writer.write_text("new.txt", "x")
    assert (out / "keep.txt").read_text() == "precious"  # other files preserved
    assert writer.hashes["new.txt"] == hashlib.sha256(b"x").hexdigest()
    with pytest.raises(PodPlatformError):
        writer.write_text("../escape.txt", "no")
    strict = ArtifactWriter(tmp_path / "fresh")
    strict.write_text("a.txt", "1")
    with pytest.raises(FileExistsError):
        strict.write_text("a.txt", "2")


# ------------------------------------------------------------------ freeze


def test_assess_freeze_refuses_without_measured_prerequisites():
    log = PeriodComplianceLog(_limits())
    gate = SourceEligibilityGate(research_only=True).evaluate(
        [SourceUse("f", "s", "core", _card("s"))]
    )
    assessment = assess_freeze(
        manifest=None,
        outputs={},
        gate=gate,
        certificate_count=0,
        deployed_forecast_count=7,
        accounting_reconciled=True,
        compliance=log,
        trial_count=0,
        evaluation_access_count=0,
        frozen_at=T0,
    )
    assert isinstance(assessment, FreezeAssessment) and not assessment.frozen
    joined = " | ".join(assessment.reasons)
    for needle in (
        "research-only",
        "manifest",
        "certificates",
        "official compliance",
        "trial",
    ):
        assert needle in joined


# ------------------------------------------------------ controlled fixtures

LONGS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
SHORTS = ["GGG", "HHH", "III", "JJJ", "KKK", "LLL"]
ALPHAS = dict(
    zip(
        LONGS + SHORTS,
        [
            "0.03",
            "0.028",
            "0.026",
            "0.024",
            "0.01",
            "0.008",
            "-0.03",
            "-0.028",
            "-0.026",
            "-0.024",
            "-0.01",
            "-0.008",
        ],
    )
)


def _certificate() -> AdmissionCertificate:
    return AdmissionCertificate(
        owner="integrity",
        created_at=T0,
        certificate_id="cert_a",
        model_id="model_a",
        model_hash=hashlib.sha256(b"model_a").hexdigest(),
        feature_hashes={"f": "h"},
        data_hashes={"d": "h"},
        approved_universe="test_universe",
        target_id="t1",
        horizon_trading_days=21,
        effective_start=T0,
        approved_mode="controlled_replay",
        expiry_policy="annual",
        promotion_evidence_ids=["ev_promotion_1"],
    ).sealed()


def _forecast(formation: dt.date, end: dt.date, sid: str) -> Forecast:
    cutoff = dt.datetime(
        formation.year, formation.month, formation.day, 12, tzinfo=UTC
    )
    earliest = dt.datetime(
        formation.year, formation.month, formation.day, 20, tzinfo=UTC
    )
    return Forecast(
        owner="strategy",
        created_at=cutoff,
        forecast_id=f"fc_{formation.isoformat()}_{sid}",
        strategy_id="fam_a",
        model_id="model_a",
        admission_certificate_id="cert_a",
        security_id=sid,
        information_cutoff=cutoff,
        earliest_formation_at=earliest,
        target=TargetPrediction(
            target_id="t1",
            prediction=Decimal(ALPHAS[sid]),
            unit="decimal_return",
            reference_prediction=Decimal("0"),
            reference_type="naive",
            expected_resolution_at=utc_close(end),
        ),
        return_forecast=ReturnForecast(
            expected_excess_return=Decimal(ALPHAS[sid]),
            unit="decimal_return",
            horizon_trading_days=21,
            bridge_id="bridge_1",
            uncertainty_artifact_id="unc_1",
        ),
        mechanism_ids=["mech_1"],
        source_family_ids=["src_signal"],
        expires_at=dt.datetime.combine(
            formation + dt.timedelta(days=20), dt.time(0), tzinfo=UTC
        ),
    ).sealed()


def make_fixture(
    root: Path,
    *,
    signal_card_eligible: bool = True,
    blank_outcome: tuple | None = None,
    liquidation: dict | None = None,
    with_lineage: bool = True,
    risk_name_cap: float = 0.3,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": "ctl1",
        "run_kind": "replay",
        "starting_nav": "1000000",
        "official_limits": {
            "min_positions": 4,
            "max_positions": 20,
            "max_gross": 2.0,
            "min_net": -0.5,
            "max_net": 0.5,
            "max_abs_name_weight": 0.3,
            "min_short_market_cap_usd": 100000000,
        },
        "internal_policy": {
            "target_positions": 8,
            "sector_neutral": False,
            "beta_neutral": False,
        },
        "admitted_families": ["fam_a"],
        "neutral_uncertainty": 0.1,
        "cost_model": {
            "model_id": "cost_ctl_v1",
            "half_spread_by_size_bp": {"large": 2.0},
            "eta": 0.1,
            "explicit_fee_bp": 0.5,
        },
        "execution_policy": {
            "policy_id": "exec_ctl_v1",
            "max_daily_participation": 0.2,
            "min_resize_fraction": 0.0,
            "no_trade_band": 0.0005,
            "min_economic_edge_bp": 0.0,
            "order_ttl_days": 5,
        },
        "fill_window_days": 5,
        "core_sources": ["src_prices"],
        "risk": {
            "policy_version": "risk_ctl_v1",
            "max_gross": 2.0,
            "max_abs_net": 0.5,
            "max_name_weight": risk_name_cap,
            "min_names": 4,
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    cards = [
        _card("src_prices"),
        _card("src_signal", eligible=signal_card_eligible),
    ]
    (root / "source_cards.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in cards])
    )
    (root / "certificates.json").write_text(
        json.dumps([_certificate().model_dump(mode="json")])
    )
    forecasts = [
        _forecast(f, e, sid)
        for f, e in zip(FORMATIONS, ENDS)
        for sid in LONGS + SHORTS
    ]
    (root / "forecasts.json").write_text(
        json.dumps([f.model_dump(mode="json") for f in forecasts])
    )
    with (root / "market.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "formation_date",
                "security_id",
                "ticker",
                "name",
                "label_source",
                "label_reference_date",
                "market_cap_usd",
                "beta",
                "sector",
                "tradable",
                "price",
                "adv_dollars",
                "daily_volatility",
                "size_tier",
                "outcome_excess_return",
            ]
        )
        for i, formation in enumerate(FORMATIONS):
            for sid in LONGS + SHORTS:
                outcome = (
                    (0.02 + 0.004 * (i % 2))
                    if sid in LONGS
                    else (-0.01 - 0.002 * i)
                )
                outcome = "" if blank_outcome == (i, sid) else repr(outcome)
                writer.writerow(
                    [
                        formation.isoformat(),
                        sid,
                        sid,
                        f"{sid} Corp",
                        "fixture_security_master",
                        "2024-01-01",
                        "1000000000",
                        "1.0",
                        "tech",
                        "true",
                        "100",
                        "50000000",
                        "0.02",
                        "large",
                        outcome,
                    ]
                )
    with (root / "periods.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["formation_date", "period_end", "tb3ms_annual_percent"]
        )
        for f, e in zip(FORMATIONS, ENDS):
            writer.writerow([f.isoformat(), e.isoformat(), "5.0"])
    if liquidation:
        (root / "liquidations.json").write_text(json.dumps([liquidation]))
    if with_lineage:
        (root / "lineage.json").write_text(
            json.dumps(
                {
                    "security_master_version": "secmaster_2024_01_v1",
                    "economic_graph_version": "graph_2024_01_v1",
                    "calendar_policy": "nyse_month_end",
                    "timezone_policy": "utc_close_2100",
                    "data_access_policy_version": "dap_v1",
                    "evaluation_policy_version": "eval_v1",
                    "trial_count": 3,
                    "evaluation_access_count": 0,
                }
            )
        )
    return root


def run_controlled(
    tmp_path, name="out", *, research_only=False, **fixture_kwargs
):
    fixture = load_fixture(
        make_fixture(tmp_path / f"fx_{name}", **fixture_kwargs)
    )
    runner = ControlledReplay(
        fixture, tmp_path / name, research_only=research_only
    )
    return runner.run(), tmp_path / name


def test_controlled_replay_end_to_end_conserves_and_freezes(tmp_path):
    outcome, out = run_controlled(tmp_path)
    summary = outcome["summary"]
    assert summary["freeze_status"] == "FROZEN", summary["freeze_reasons"]
    assert (
        summary["labels"]["run_kind"] == "replay"
        and summary["llm_called"] is False
    )
    assert summary["source_policy"]["run_label"] == "FORMATION_ELIGIBLE"
    for name in (
        "holdings_export.csv",
        "returns_export.csv",
        "performance_report.json",
        "compliance_report.json",
        "decision_dossier.json",
        "run_manifest.json",
        "freeze_assessment.json",
        "artifact_hashes.json",
        "holdings_by_period.csv",
        "risk_reviews.json",
        "periods.json",
        "run_summary.json",
    ):
        assert (out / name).is_file(), name
    hashes = json.loads((out / "artifact_hashes.json").read_text())
    assert (
        hashes["holdings_export.csv"]
        == hashlib.sha256(
            (out / "holdings_export.csv").read_bytes()
        ).hexdigest()
    )
    # ledger NAV equals compounded reported net returns (conservation)
    monthly = json.loads((out / "monthly.json").read_text())
    assert len(monthly) == 4
    nav = 1_000_000.0
    for row in monthly:
        nav *= 1.0 + row["net_return"]
        assert all(
            abs(Decimal(v)) < Decimal("1e-6")
            for v in row["reconciliation"].values()
        )
        assert row["trading_fees"] != "0" and row["turnover_one_way"] >= 0.0
    assert nav == pytest.approx(float(summary["ending_nav"]), rel=1e-9)
    # all periods and all stages are in the compliance report with both weight sets
    compliance = json.loads((out / "compliance_report.json").read_text())
    assert (
        len(compliance["records"]) == 4 * 4 and compliance["all_official_pass"]
    )
    assert {r["period"] for r in compliance["records"]} == {
        d.isoformat() for d in FORMATIONS
    }
    assert all(
        r["pre_weights"] is not None and r["post_weights"]
        for r in compliance["records"]
    )
    # later periods trade from drifted holdings: some turnover is below the flat-start 0.5
    assert monthly[0]["turnover_one_way"] == pytest.approx(0.5, abs=0.02)
    assert monthly[2]["turnover_one_way"] < monthly[0]["turnover_one_way"]
    # independent risk reviews and fills are in the audit trail
    reviews = json.loads((out / "risk_reviews.json").read_text())
    assert [r["decision"] for r in reviews] == ["PASS"] * 4
    periods = json.loads((out / "periods.json").read_text())
    assert periods[0]["orders"] and all(
        o["unfilled_dollars"] == "0" for o in periods[0]["orders"]
    )
    dossier = json.loads((out / "decision_dossier.json").read_text())
    assert (
        "AAA" in dossier
        and dossier["AAA"]["forecast_ids"]
        and dossier["AAA"]["risk_review_ids"]
    )
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert (
        manifest["source_snapshot_ids"]["src_signal"]
        == "snap_src_signal_2024_01"
    )
    assert manifest["config_hashes"]["config"] and "latest" not in json.dumps(
        manifest
    )


def test_controlled_replay_is_reproducible(tmp_path):
    _, out_a = run_controlled(tmp_path, "run_a")
    _, out_b = run_controlled(tmp_path, "run_b")
    assert (out_a / "artifact_hashes.json").read_bytes() == (
        out_b / "artifact_hashes.json"
    ).read_bytes()
    assert (out_a / "returns_export.csv").read_bytes() == (
        out_b / "returns_export.csv"
    ).read_bytes()


def test_unapproved_source_rejected_for_formation_and_labeled_when_research_only(
    tmp_path,
):
    fixture = load_fixture(
        make_fixture(tmp_path / "fx_blocked", signal_card_eligible=False)
    )
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(fixture, tmp_path / "strict").run()
    assert code_of(err) == ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE
    failure = json.loads((tmp_path / "strict" / "failure.json").read_text())
    assert failure["status"] == "FAILED_CLOSED" and failure["frozen"] is False

    outcome, out = run_controlled(
        tmp_path, "explore", research_only=True, signal_card_eligible=False
    )
    summary = json.loads((out / "run_summary.json").read_text())
    assert (
        summary["source_policy"]["run_label"] == "RESEARCH_ONLY_NOT_FREEZABLE"
    )
    assert summary["source_policy"]["blocked_sources"] == ["src_signal"]
    assert (
        summary["freeze_status"] == "NOT_FROZEN"
        and not (out / "run_manifest.json").exists()
    )
    assert any("research-only" in r for r in summary["freeze_reasons"])


def test_missing_lineage_blocks_freeze_without_fabricating_ids(tmp_path):
    outcome, out = run_controlled(tmp_path, with_lineage=False)
    assert outcome["summary"]["freeze_status"] == "NOT_FROZEN"
    assert "security_master_version" in outcome["summary"]["missing_lineage"]
    assert not (out / "run_manifest.json").exists()


def test_controlled_missing_outcome_mid_run_fails_closed_unless_dated_liquidation(
    tmp_path,
):
    fixture = load_fixture(
        make_fixture(tmp_path / "fx_missing", blank_outcome=(1, "AAA"))
    )
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(fixture, tmp_path / "miss").run()
    assert code_of(
        err
    ) == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED and "AAA" in str(err.value)
    partial = json.loads(
        (tmp_path / "miss" / "periods.partial.json").read_text()
    )
    assert [p["period"] for p in partial] == [
        "2024-01-31"
    ]  # first month completed, second stopped
    compliance = json.loads(
        (tmp_path / "miss" / "compliance_report.partial.json").read_text()
    )
    assert compliance["periods_checked"] == ["2024-01-31", "2024-02-29"]

    liquidation = {
        "period_formation_date": "2024-02-29",
        "security_id": "AAA",
        "liquidated_at": "2024-03-10T15:00:00+00:00",
        "realized_excess_return": -0.25,
        "evidence_id": "ev_liq_aaa",
        "source": "fixture delisting evidence",
    }
    _, out = run_controlled(
        tmp_path, "liq", blank_outcome=(1, "AAA"), liquidation=liquidation
    )
    periods = json.loads((out / "periods.json").read_text())
    assert periods[1]["outcome_basis"]["AAA"] == "liquidation:ev_liq_aaa"

    late = dict(
        liquidation, liquidated_at="2024-04-10T15:00:00+00:00"
    )  # a later quote is future information
    fixture = load_fixture(
        make_fixture(
            tmp_path / "fx_late", blank_outcome=(1, "AAA"), liquidation=late
        )
    )
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(fixture, tmp_path / "late").run()
    assert code_of(err) == ErrorCode.ASOF_VIOLATION


def test_controlled_risk_breach_in_intermediate_period_stops_the_run(tmp_path):
    def overlay(target, formation):
        if formation != dt.date(2024, 3, 31):
            return target
        bumped = dict(target)
        top = max(bumped, key=lambda k: bumped[k])
        bumped[top] = Decimal(
            "0.28"
        )  # official cap 0.3 holds, independent risk cap 0.25 breaches
        return bumped

    fixture = load_fixture(
        make_fixture(tmp_path / "fx_risk", risk_name_cap=0.25)
    )
    # baseline 0.125/name passes both, overlay breaches only the risk policy in the third period
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(fixture, tmp_path / "risk", risk_overlay=overlay).run()
    assert "independent risk review blocked 2024-03-31" in str(err.value)
    reviews = json.loads(
        (tmp_path / "risk" / "risk_reviews.partial.json").read_text()
    )
    assert [r["decision"] for r in reviews] == ["PASS", "PASS", "BLOCK"]
    periods = json.loads(
        (tmp_path / "risk" / "periods.partial.json").read_text()
    )
    assert len(periods) == 2  # the blocked book was never traded


def test_controlled_official_breach_after_overlay_is_recorded_with_pre_and_post(
    tmp_path,
):
    def overlay(target, formation):
        if formation != dt.date(2024, 2, 29):
            return target
        bumped = dict(target)
        top = max(bumped, key=lambda k: bumped[k])
        bumped[top] = Decimal("0.35")  # above the 0.3 official cap
        return bumped

    fixture = load_fixture(make_fixture(tmp_path / "fx_off"))
    with pytest.raises(PodPlatformError) as err:
        ControlledReplay(fixture, tmp_path / "off", risk_overlay=overlay).run()
    assert code_of(
        err
    ) == ErrorCode.PORTFOLIO_INFEASIBLE and "post_risk" in str(err.value)
    compliance = json.loads(
        (tmp_path / "off" / "compliance_report.partial.json").read_text()
    )
    failing = compliance["records"][-1]
    assert failing["period"] == "2024-02-29" and failing[
        "failing_official"
    ] == ["single_name"]
    assert Decimal(
        max(failing["pre_weights"].values(), key=Decimal)
    ) == Decimal("0.125")
    assert Decimal(
        max(failing["post_weights"].values(), key=Decimal)
    ) == Decimal("0.35")


def test_controlled_runner_refuses_existing_output(tmp_path):
    fixture = load_fixture(make_fixture(tmp_path / "fx"))
    out = tmp_path / "existing"
    out.mkdir()
    (out / "old.json").write_text("{}")
    with pytest.raises(PodPlatformError):
        ControlledReplay(fixture, out)
    assert (out / "old.json").read_text() == "{}"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controlled_cli_exit_codes(tmp_path, capsys):
    cli = _load_script("run_controlled_replay")
    fixture = make_fixture(tmp_path / "fx_cli")
    assert (
        cli.main({"fixture": str(fixture), "out": str(tmp_path / "cli_out")})
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert (
        printed["freeze_status"] == "FROZEN" and printed["llm_called"] is False
    )
    assert (
        cli.main({"fixture": str(fixture), "out": str(tmp_path / "cli_out")})
        == 2
    )  # existing output
    blocked = make_fixture(
        tmp_path / "fx_cli_blocked", signal_card_eligible=False
    )
    assert (
        cli.main(
            {"fixture": str(blocked), "out": str(tmp_path / "cli_blocked")}
        )
        == 2
    )
