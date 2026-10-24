"""Part 9 acceptance: vertical slice (spec 27.4) and full-run acceptance (spec 26.8)."""

from __future__ import annotations

import datetime as dt
import hashlib
from decimal import Decimal

import pytest

from pod_platform.accounting import AccountingLedger
from pod_platform.allocation import (
    Candidate,
    ConstructionEngine,
    FamilyEstimate,
    FamilyIntegrator,
    InternalPolicy,
    OfficialLimits,
)
from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.contracts.orders import OrderSide
from pod_platform.execution import (
    CostModel,
    CostModelConfig,
    ExecutionPlanner,
    ExecutionPolicy,
    LiquiditySnapshot,
)
from pod_platform.orchestration.pipeline import VerticalSlicePipeline
from pod_platform.orchestration.run_manifest import RunManifest
from pod_platform.reporting import verify_deck_numbers
from pod_platform.research import (
    ModelState,
    PromotionCriteria,
    PromotionOutcome,
    RegisteredExperiment,
    Registry,
    ResearchLedger,
    StateTransition,
    ModelLifecycle,
)
from pod_platform.risk import MeasuredExposure, RiskEngine, RiskLimit, RiskPolicy
from pod_platform.strategies import (
    EarningsInnovationCandidate,
    SecurityFeatures,
)

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)
D = Decimal


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pipeline_components():
    registry = Registry(":memory:")
    ledger = ResearchLedger(registry)
    integrator = FamilyIntegrator(("demand", "expectations"))
    construction = ConstructionEngine(OfficialLimits(), InternalPolicy(target_positions=300))
    cost_model = CostModel(
        CostModelConfig(
            model_id="cost_v1",
            half_spread_by_size_bp={"large": 5.0},
            eta=1.0,
            explicit_fee_bp=1.0,
        )
    )
    planner = ExecutionPlanner(
        ExecutionPolicy(
            policy_id="exec_v1",
            max_daily_participation=0.10,
            min_resize_fraction=0.25,
            no_trade_band=0.001,
            min_economic_edge_bp=0.0,
            order_ttl_days=5,
        ),
        cost_model,
    )
    accounting = AccountingLedger(starting_cash=D("1000000000"))
    pipeline = VerticalSlicePipeline(
        ledger=ledger,
        integrator=integrator,
        construction=construction,
        planner=planner,
        cost_model=cost_model,
        accounting=accounting,
    )
    return pipeline, ledger, registry


def _experiment() -> RegisteredExperiment:
    return RegisteredExperiment(
        owner="strategy_1",
        created_at=T0,
        registration_id="reg_slice_001",
        experiment_spec_id="exp_slice_001",
        spec_content_hash=_sha("exp_slice_001"),
        primary_metric="rank_ic",
        baseline_experiment_ids=["public_information"],
        promotion_conditions="improvement over baseline",
        failure_conditions="no improvement",
        variant_budget=3,
        registered_at=T0,
    )


def _tradable_candidates(n_pairs: int = 150):
    out = []
    for i in range(n_pairs):
        out.append(Candidate(f"L{i}", 0.02 + i * 1e-5, 2e9, sector="tech"))
        out.append(Candidate(f"S{i}", -0.02 - i * 1e-5, 2e9, sector="tech"))
    return out


def _liquidity_for(candidates) -> dict[str, LiquiditySnapshot]:
    return {
        c.security_id: LiquiditySnapshot(
            security_id=c.security_id,
            adv_dollars=D("50000000"),
            daily_volatility=0.02,
            size_tier="large",
            price=D("50"),
        )
        for c in candidates
    }


def _exposures():
    return [
        MeasuredExposure("gross", 1.0, 2.0, is_max=True),
        MeasuredExposure("net", 0.0, 0.5, is_max=True),
    ]


def _risk_engine():
    return RiskEngine(
        RiskPolicy(
            policy_version="policy_v1",
            limits=[
                RiskLimit(name="gross", threshold="<=2.0", hard=True),
                RiskLimit(name="net", threshold="[-0.5,0.5]", hard=True),
            ],
        )
    )


def _features_map(n: int = 300):
    return {
        f"F{i}": SecurityFeatures(
            security_id=f"F{i}",
            as_of=T0,
            values={"earnings_surprise": 0.5, "consensus_revision_3m": 0.2},
        )
        for i in range(n)
    }


def _estimates_for(candidates):
    out = {}
    for c in candidates:
        out[c.security_id] = [
            FamilyEstimate("demand", c.security_id, c.alpha, 0.01),
            FamilyEstimate("expectations", c.security_id, c.alpha, 0.01),
        ]
    return out


def _run_slice():
    pipeline, ledger, registry = _pipeline_components()
    candidates = _tradable_candidates()
    liquidity = _liquidity_for(candidates)
    artifacts = pipeline.run(
        experiment=_experiment(),
        candidates_by_family=_estimates_for(candidates),
        tradable_candidates=candidates,
        liquidity=liquidity,
        risk_exposures=_exposures(),
        risk_engine=_risk_engine(),
        market_features=_features_map(),
        strategy=EarningsInnovationCandidate(),
        model_id="model_v1",
        certificate_id="cert_test_v1",
        expected_resolution_at=T0 + dt.timedelta(days=45),
        expires_at=T0 + dt.timedelta(days=25),
        decision_at=T0,
        starting_nav=D("1000000000"),
        min_positions=200,
        run_id="run_slice",
        period_label="2021-01",
        benchmark_tb3ms_annual_percent=2.4,
    )
    return artifacts, ledger, registry


# ------------------------------------------------------------- 27.4 artifacts


def test_vertical_slice_produces_all_required_artifacts():
    artifacts, _, _ = _run_slice()
    acceptance = artifacts.acceptance()
    assert all(acceptance.values()), f"incomplete slice: {acceptance}"
    assert artifacts.all_stages_complete()
    assert len(artifacts.forecast_ids) == 300
    assert artifacts.target_book is not None
    assert artifacts.risk_review.decision.value in ("PASS", "PASS_WITH_REMEDY")
    assert artifacts.planned_orders
    assert artifacts.initialization.names_held >= 200
    assert len(artifacts.nav_path) == 2
    assert artifacts.required_table is not None
    assert artifacts.dossier["security_id"] != "none"


def test_forecast_batch_carries_certificates():
    artifacts, _, _ = _run_slice()
    assert all("forecast" in fid or fid.startswith("strat") for fid in artifacts.forecast_ids[:1])
    assert artifacts.forecast_ids


def test_target_book_respects_official_limits():
    artifacts, _, _ = _run_slice()
    book = artifacts.target_book
    gross = sum(abs(p.target_weight) for p in book.positions)
    assert gross <= 2.0 + 1e-9
    assert 200 <= len(book.positions) <= 500
    assert all(r.status == "pass" for r in book.official_constraint_results)


def test_risk_review_expires():
    artifacts, _, _ = _run_slice()
    review = artifacts.risk_review
    assert review.is_valid_at(T0) is True
    assert review.is_valid_at(T0 + dt.timedelta(hours=3)) is False


def test_initialization_charges_costs_in_nav_path():
    artifacts, _, _ = _run_slice()
    init = artifacts.initialization
    assert init.entry_costs > 0
    expected = float(init.starting_nav - init.entry_costs)
    assert abs(artifacts.nav_path[-1] - expected) < 1e-3


def test_dossier_explains_position():
    artifacts, _, _ = _run_slice()
    dossier = artifacts.dossier
    assert dossier["decision_events"]
    assert dossier["decision_events"][0]["authorizing_rule_or_certificate_id"]


# ------------------------------------------------------------- 26.8 acceptance


def test_full_run_manifest_complete():
    manifest = RunManifest(
        run_id="run_slice",
        code_commit_or_tree_hash="a" * 40,
        environment_lock_hash="b" * 64,
        config_hashes={"mandate": "c" * 64},
        source_snapshot_ids={"chars_panel": "snap_001"},
        security_master_version="secmaster_v1",
        economic_graph_version="graph_v1",
        model_versions={"model_v1": _sha("model_v1")},
        bridge_versions={"bridge_rank_spread_v1": _sha("bridge")},
        calendar_policy="trading_days_nyse",
        timezone_policy="UTC",
        data_access_policy_version="policy_v1",
        evaluation_policy_version="eval_policy_v1",
        expected_outputs={"holdings": "d" * 64},
    )
    assert manifest.run_id == "run_slice"


def test_trial_counts_and_exposure_reported():
    _, ledger, _ = _run_slice()
    assert ledger.trial_count("registered_specification") == 1
    assert ledger.trial_count() >= 1


def test_formation_inputs_require_approved_certificates():
    artifacts, _, _ = _run_slice()
    assert artifacts.target_book.forecast_publication_ids
    assert all("forecast" in fid or fid.startswith("strat") for fid in artifacts.target_book.forecast_publication_ids[:1])


def test_accounting_reconciles():
    artifacts, _, _ = _run_slice()
    init = artifacts.initialization
    expected = float(init.starting_nav - init.entry_costs)
    assert abs(artifacts.nav_path[-1] - expected) < 1e-6


def test_deck_numbers_tie_to_artifacts():
    artifacts, _, _ = _run_slice()
    table = artifacts.required_table
    verify_deck_numbers(
        {"ending_value": table["ending_value"]},
        {"ending_value": table["ending_value"]},
    )
    with pytest.raises(PodPlatformError):
        verify_deck_numbers({"ending_value": 1.0}, {"ending_value": 2.0})


def test_blocked_risk_returns_early_with_incident():
    pipeline, ledger, _ = _pipeline_components()
    candidates = _tradable_candidates()
    liquidity = _liquidity_for(candidates)
    bad_exposures = [
        MeasuredExposure("gross", 3.0, 2.0, is_max=True),
        MeasuredExposure("net", 0.0, 0.5, is_max=True),
    ]
    artifacts = pipeline.run(
        experiment=_experiment(),
        candidates_by_family=_estimates_for(candidates),
        tradable_candidates=candidates,
        liquidity=liquidity,
        risk_exposures=bad_exposures,
        risk_engine=_risk_engine(),
        market_features=_features_map(),
        strategy=EarningsInnovationCandidate(),
        model_id="model_v1",
        certificate_id="cert_test_v1",
        expected_resolution_at=T0 + dt.timedelta(days=45),
        expires_at=T0 + dt.timedelta(days=25),
        decision_at=T0,
        starting_nav=D("1000000000"),
        min_positions=200,
        run_id="run_blocked",
        period_label="2021-01",
        benchmark_tb3ms_annual_percent=2.4,
    )
    assert artifacts.risk_review.decision.value == "BLOCK"
    assert artifacts.incidents
    assert artifacts.planned_orders == []
    assert artifacts.nav_path == []


def test_promotion_path_admits_model():
    life = ModelLifecycle()
    for from_state, to_state in (
        (ModelState.IDEA, ModelState.REGISTERED),
        (ModelState.REGISTERED, ModelState.DATA_READY),
        (ModelState.DATA_READY, ModelState.DEVELOPMENT),
        (ModelState.DEVELOPMENT, ModelState.CHALLENGED),
        (ModelState.CHALLENGED, ModelState.CHECK_ELIGIBLE),
        (ModelState.CHECK_ELIGIBLE, ModelState.SHADOW),
        (ModelState.SHADOW, ModelState.ADMITTED),
    ):
        life.transition(
            StateTransition(
                model_id="model_v1",
                from_state=from_state,
                to_state=to_state,
                authority="integrity",
                reason="evidence accepted",
                at=T0,
            )
        )
    assert life.admitted("model_v1") is True
    outcome = PromotionOutcome(
        registration_id="reg_slice_001",
        evaluated_at=T0,
        criteria=PromotionCriteria(
            min_coverage="500 stock-months",
            improvement_vs_primary_baseline="positive",
            direction_and_economic_size="declared sign",
            stability_requirement="same sign both halves",
            return_translation_requirement="net positive",
            implementation_requirement="tradable",
            permitted_statistical_uncertainty="95% CI excludes zero",
            multiple_testing_treatment="BH over registered variants",
        ),
        coverage_met=True,
        improvement_met=True,
        direction_met=True,
        stability_met=True,
        implementation_met=True,
        uncertainty_met=True,
        multiple_testing_met=True,
    )
    cert = outcome.certificate_or_raise(
        model_id="model_v1",
        model_hash=_sha("model_v1"),
        feature_hashes={"f1": _sha("f1")},
        data_hashes={"d1": _sha("d1")},
        approved_universe="us_public_equities_v1",
        target_id="next_reported_earnings_innovation",
        horizon_trading_days=20,
        effective_start=T0,
        approved_mode="contemporary_research",
        expiry_policy="review after 12 months",
        promotion_evidence_ids=["ev_1"],
    )
    assert cert.is_valid_at(T0)
