"""Part 9 tail acceptance: compliance on actual holdings, run freeze, judge package
(spec 26.5, 26.8, 28.4)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.orchestration.freeze import freeze_run
from pod_platform.orchestration.run_manifest import RunManifest
from pod_platform.reporting import (
    ActualHolding,
    AgentEvidence,
    ComplianceEngine,
    ImplementationEvidence,
    IncrementalEvidence,
    JudgePackage,
    ResearchFunnel,
    WorkedExample,
    verify_package,
)
from pod_platform.allocation import OfficialLimits

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)
D = Decimal


def _holding(sid: str, weight: str, cap: str = "2000000000", formation_cap: str = "2000000000") -> ActualHolding:
    return ActualHolding(
        security_id=sid,
        weight=D(weight),
        market_cap_usd=D(cap),
        formation_market_cap_usd=D(formation_cap),
    )


def _compliant_holdings(n: int = 250):
    weight = D("1") / D(n * 2)
    out = []
    for i in range(n // 2):
        out.append(_holding(f"L{i}", str(weight)))
    for i in range(n // 2):
        out.append(_holding(f"S{i}", str(-weight)))
    return out


# ------------------------------------------------------------ 26.5 compliance


def test_compliant_holdings_pass():
    engine = ComplianceEngine(OfficialLimits())
    report = engine.enforce(_compliant_holdings())
    assert report.official_pass
    assert report.incidents == []


def test_name_count_floor_enforced():
    engine = ComplianceEngine(OfficialLimits())
    holdings = _compliant_holdings(n=50)
    report = engine.check(holdings)
    failing = [c.name for c in report.checks if c.status == "fail"]
    assert "position_count" in failing
    with pytest.raises(PodPlatformError) as exc:
        engine.enforce(holdings)
    assert exc.value.response.error.code == ErrorCode.PORTFOLIO_INFEASIBLE


def test_single_name_cap_enforced():
    engine = ComplianceEngine(OfficialLimits())
    holdings = _compliant_holdings()
    holdings[0] = _holding("L0", "0.08")
    with pytest.raises(PodPlatformError):
        engine.enforce(holdings)


def test_short_formation_market_cap_floor_enforced():
    engine = ComplianceEngine(OfficialLimits())
    holdings = _compliant_holdings()
    holdings[-1] = _holding("S_last", str(-D("0.002")), formation_cap="400000000")
    with pytest.raises(PodPlatformError):
        engine.enforce(holdings)


def test_duplicate_security_rejected():
    engine = ComplianceEngine(OfficialLimits())
    holdings = _compliant_holdings()
    holdings.append(_holding("L0", "0.001"))
    with pytest.raises(PodPlatformError):
        engine.check(holdings)


def test_internal_net_warn_not_fail():
    engine = ComplianceEngine(OfficialLimits())
    holdings = [ _holding("L0", "0.4"), _holding("L1", "0.4") ]
    report = engine.check(holdings, max_abs_internal_net=D("0.10"))
    internal = [c for c in report.checks if c.scope == "internal"]
    assert internal[0].status == "warn"
    assert report.official_pass is False


# ------------------------------------------------------------- 26.8 / W18 freeze


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="run_1",
        code_commit_or_tree_hash="a" * 40,
        environment_lock_hash="b" * 64,
        config_hashes={"mandate": "c" * 64},
        source_snapshot_ids={"chars_panel": "snap_001"},
        security_master_version="secmaster_v1",
        economic_graph_version="graph_v1",
        calendar_policy="trading_days_nyse",
        timezone_policy="UTC",
        data_access_policy_version="policy_v1",
        evaluation_policy_version="eval_policy_v1",
    )


def _outputs() -> dict[str, bytes]:
    return {
        "holdings_export": b"holdings",
        "returns_export": b"returns",
        "performance_report": b"perf",
        "compliance_report": b"compliance",
        "decision_dossier": b"dossier",
    }


def _freeze_kwargs(**over):
    base = dict(
        frozen_at=T0,
        formation_inputs_approved=True,
        certificate_count=10,
        deployed_forecast_count=10,
        accounting_reconciled=True,
        actual_holdings_compliant=True,
        trial_count=25,
        evaluation_access_count=3,
    )
    base.update(over)
    return base


def test_freeze_succeeds_when_acceptance_met():
    result = freeze_run(_manifest(), _outputs(), **_freeze_kwargs())
    assert result.accepted
    assert result.manifest_hash
    assert set(result.output_hashes) == set(_outputs())
    assert result.as_dict()["accepted"] is True


def test_freeze_rejects_missing_outputs():
    outputs = _outputs()
    del outputs["decision_dossier"]
    with pytest.raises(PodPlatformError) as exc:
        freeze_run(_manifest(), outputs, **_freeze_kwargs())
    assert exc.value.response.error.code == ErrorCode.ACCOUNTING_RECONCILIATION_FAILED


def test_freeze_rejects_unapproved_formation_inputs():
    with pytest.raises(PodPlatformError):
        freeze_run(_manifest(), _outputs(), **_freeze_kwargs(formation_inputs_approved=False))


def test_freeze_rejects_missing_certificates():
    with pytest.raises(PodPlatformError):
        freeze_run(
            _manifest(),
            _outputs(),
            **_freeze_kwargs(certificate_count=2, deployed_forecast_count=10),
        )


def test_freeze_rejects_unreconciled_accounting():
    with pytest.raises(PodPlatformError):
        freeze_run(_manifest(), _outputs(), **_freeze_kwargs(accounting_reconciled=False))


def test_freeze_with_disclosed_incidents_allowed():
    result = freeze_run(
        _manifest(),
        _outputs(),
        **_freeze_kwargs(actual_holdings_compliant=False, incidents_disclosed=["late_fill_2021-03"]),
    )
    assert result.accepted
    evidence = [c.evidence for c in result.checks if "implementation" in c.item][0]
    assert "late_fill_2021-03" in evidence


def test_freeze_requires_trial_counts():
    with pytest.raises(PodPlatformError):
        freeze_run(_manifest(), _outputs(), **_freeze_kwargs(trial_count=0))


# ------------------------------------------------------- 28.4 judge package


def _example(kind: str) -> WorkedExample:
    return WorkedExample(
        example_id=f"ex_{kind}",
        kind=kind,
        security_id="sec_1",
        question="does demand growth predict margins?",
        measurements="card spend growth",
        test="registered experiment exp_1",
        outcome="admitted" if kind == "complete_investment" else "rejected",
        evidence_ids=["ev_1"],
    )


def _package(**over) -> JudgePackage:
    base = dict(
        package_id="pkg_1",
        run_id="run_1",
        complete_investment_example=_example("complete_investment"),
        rejected_idea=_example("rejected_idea"),
        failed_trade=_example("failed_trade"),
        research_funnel=ResearchFunnel(
            hypotheses_registered=12,
            experiments_run=12,
            variants_scored=30,
            admitted=3,
            rejected=7,
            descriptive_only=2,
        ),
        incremental_evidence=IncrementalEvidence(
            baseline_comparison_id="baseline_1",
            strategy_contributions={"s1": "contrib_1"},
            redundancy_notes="s1/s3 overlap on earnings exposure",
        ),
        implementation_evidence=ImplementationEvidence(
            alpha_decay_curve_id="decay_1",
            liquidity_evidence_id="liq_1",
            turnover_summary="35% per rebalance",
            borrow_cost_summary="12bp annual",
            cost_sensitivity_id="sens_1",
        ),
        agent_evidence=AgentEvidence(
            counterfactual_id="twin_1",
            worked_case="exception deferral case",
            interventions=2,
            net_effect_summary="+4bp net of costs",
        ),
        official_statistics_artifact_id="stats_1",
        dossier_demo="dossier lookup for sec_1 on 2021-01-04",
        presentation_plan=[
            "product and mandate",
            "four charters",
            "one discovery",
            "what failed",
            "integration and risk",
            "agent vs twin",
            "required table",
            "future and limits",
        ],
    )
    base.update(over)
    return JudgePackage(**base)


def test_judge_package_complete():
    summary = verify_package(_package())
    assert all(summary["coverage"].values())
    assert summary["presentation_minutes"] == 8


def test_judge_package_requires_all_three_narratives():
    with pytest.raises(ValueError):
        _package(
            failed_trade=WorkedExample(
                example_id="ex_wrong",
                kind="complete_investment",
                security_id="sec_1",
                question="q",
                measurements="m",
                test="t",
                outcome="o",
                evidence_ids=["ev_1"],
            )
        )


def test_judge_package_funnel_consistency():
    with pytest.raises(ValueError):
        _package(
            research_funnel=ResearchFunnel(
                hypotheses_registered=5,
                experiments_run=3,
                variants_scored=5,
                admitted=2,
                rejected=2,
                descriptive_only=0,
            )
        )


def test_judge_package_missing_evidence_rejected():
    with pytest.raises(ValueError):
        _package(official_statistics_artifact_id="")
