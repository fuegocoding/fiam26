"""W01 acceptance: valid/invalid schema fixtures for every contract (spec 21, 26)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from pod_platform.contracts import (
    ActorKind,
    AssessmentStatus,
    CardStatus,
    DecisionEvent,
    ErrorCode,
    ExperimentSpec,
    Fill,
    Forecast,
    HypothesisSpec,
    Order,
    OrderSide,
    PodPlatformError,
    ResearchMode,
    ReturnForecast,
    RiskDecision,
    RiskReview,
    Screen,
    ScreenItem,
    SolverStatus,
    SourceCard,
    TargetBook,
    TargetPrediction,
    TargetPosition,
    ToolResponse,
    TypedError,
)
from pod_platform.orchestration.run_manifest import RunManifest

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _screen(status: AssessmentStatus = AssessmentStatus.ESTABLISHED) -> Screen:
    item = ScreenItem(status=status, evidence_id="ev_1" if status == AssessmentStatus.ESTABLISHED else None)
    return Screen(history=item, metric_lag=item, vintage=item, coverage=item)


def _source_card(**over) -> SourceCard:
    base = dict(
        owner="data",
        created_at=NOW,
        source_id="source_example",
        dataset_id="CA0093",
        topic="Contracts",
        snapshot_id="catalogue_20261024",
        status=CardStatus.SAMPLED,
        access_confirmed=False,
        formation_eligible=False,
        entity_grain=["company", "period"],
        screen=Screen(
            history=ScreenItem(status=AssessmentStatus.DOCUMENTED_ONLY, evidence_id="doc_example"),
            metric_lag=ScreenItem(status=AssessmentStatus.UNRESOLVED),
            vintage=ScreenItem(status=AssessmentStatus.UNRESOLVED),
            coverage=ScreenItem(status=AssessmentStatus.SAMPLE_ONLY, evidence_id="sample_example"),
        ),
        unresolved_questions=["Can historical versions be reconstructed?"],
    )
    base.update(over)
    return SourceCard(**base)


def _hypothesis(**over) -> HypothesisSpec:
    base = dict(
        owner="strategy_1",
        created_at=NOW,
        hypothesis_id="hyp_example_margin_001",
        mechanism_family="discount_driven_volume",
        research_mode=ResearchMode.CONTEMPORARY_RESEARCH,
        question="Does demand growth supported by discounting predict weaker margins?",
        mechanism="Promotions may increase volume while reducing profit per transaction.",
        target_id="next_reported_margin_change",
        reference_model_id="public_margin_baseline_v1",
        feature_spec_ids=["traffic_growth_v1"],
        applicability_spec_id="comparable_retail_businesses_v1",
        baseline_experiment_ids=["traffic_only"],
        falsification=["No stable improvement versus the primary baseline"],
        variant_budget=3,
        experiment_spec_id="exp_example_margin_001",
    )
    base.update(over)
    return HypothesisSpec(**base)


def _experiment(**over) -> ExperimentSpec:
    base = dict(
        owner="strategy_1",
        created_at=NOW,
        experiment_spec_id="exp_example_margin_001",
        identity={
            "hypothesis_id": "hyp_example_margin_001",
            "variant_id": "v1",
            "parent_experiment_id": None,
            "owner": "strategy_1",
        },
        claim={
            "primary_target_id": "next_reported_margin_change",
            "primary_metric": "rank_ic",
            "economic_effect": "discount-driven demand growth precedes margin compression",
            "declared_sign": "negative",
        },
        data={
            "source_snapshot_ids": ["catalogue_20261024"],
            "feature_spec_ids": ["traffic_growth_v1"],
            "identity_graph_version": "graph_v1",
        },
        sample={
            "universe_spec_id": "us_public_equities_v1",
            "eligibility_spec_id": "tradable_v1",
            "applicability_spec_id": "comparable_retail_businesses_v1",
        },
        time={
            "training_cutoff": "2018-12-31T23:59:59Z",
            "label_maturity_rule": "labels resolved by training cutoff only",
            "partitions": {"design": "2015-2018", "check": "2019-2020"},
            "purge_rule": "purge overlapping target windows at partition edges",
            "embargo_rule": "embargo equal to target window overlap",
        },
        model={
            "method_id": "ridge_rank_v1",
            "hyperparameter_space": {"alpha": [0.1, 1.0]},
            "seed_policy": "fixed_seeds_0_3",
        },
        portfolio={
            "entry_rule": "next eligible close after formation",
            "exit_rule": "end of horizon or invalidation",
            "horizon_trading_days": 20,
            "construction_policy_id": "deterministic_selection_v1",
            "hedge_policy_id": "permitted_stock_hedges_v1",
        },
        costs={
            "trade_cost_model_id": "cost_v1",
            "borrow_model_id": "borrow_v1",
            "liquidity_model_id": "liquidity_v1",
            "funding_model_id": "funding_v1",
        },
        comparisons={
            "baseline_experiment_ids": ["traffic_only"],
            "paired_comparison_policy_id": "paired_v1",
            "permitted_sensitivities": ["cost_plus_50pct"],
        },
        statistics={
            "dependence_handling": "date_clustered_se",
            "uncertainty_method": "block_bootstrap",
            "selection_accounting_policy_id": "trial_counters_v1",
        },
        admission={
            "min_coverage": "500 stock-months",
            "effect_requirement": "positive improvement over primary baseline",
            "stability_requirement": "same sign in both halves",
            "implementation_requirement": "net positive after frozen costs",
        },
        budget={
            "compute_budget": "100 cpu_hours",
            "provider_budget": "none",
            "llm_budget": "none",
            "evaluation_access_bounds": "design only",
        },
        promotion_conditions="all admission conditions met on check",
        failure_conditions="no improvement or unstable sign",
    )
    base.update(over)
    return ExperimentSpec(**base)


def _target_prediction(**over) -> TargetPrediction:
    base = dict(
        target_id="next_reported_margin_change",
        prediction=Decimal("-0.01"),
        unit="decimal_margin_change",
        reference_prediction=Decimal("-0.002"),
        reference_type="public_model",
        expected_resolution_at=dt.datetime(2020, 11, 5, 21, 30, tzinfo=UTC),
    )
    base.update(over)
    return TargetPrediction(**base)


def _return_forecast(**over) -> ReturnForecast:
    base = dict(
        expected_excess_return=Decimal("-0.004"),
        unit="decimal_return",
        horizon_trading_days=20,
        bridge_id="bridge_example_v2",
        uncertainty_artifact_id="uncertainty_example",
    )
    base.update(over)
    return ReturnForecast(**base)


def _forecast(**over) -> Forecast:
    base = dict(
        owner="strategy_1",
        created_at=NOW,
        forecast_id="forecast_example_001",
        strategy_id="strategy_1",
        model_id="model_example_v3",
        admission_certificate_id="certificate_example",
        security_id="security_example",
        information_cutoff=dt.datetime(2020, 10, 19, 20, 0, tzinfo=UTC),
        earliest_formation_at=dt.datetime(2020, 10, 20, 20, 0, tzinfo=UTC),
        target=_target_prediction(),
        return_forecast=_return_forecast(),
        mechanism_ids=["discount_driven_volume"],
        source_family_ids=["consumer_activity"],
        evidence_artifact_ids=["evidence_example_1"],
        expires_at=dt.datetime(2020, 11, 18, 21, 0, tzinfo=UTC),
    )
    base.update(over)
    return Forecast(**base)


def _target_book(**over) -> TargetBook:
    base = dict(
        owner="allocation",
        created_at=NOW,
        target_book_id="book_001",
        run_id="run_001",
        decision_at=NOW,
        forecast_publication_ids=["forecast_example_001"],
        current_book_snapshot_id="snapshot_prev",
        nav_snapshot_id="nav_001",
        price_snapshot_id="px_001",
        risk_envelope_version="envelope_v1",
        covariance_version="cov_v1",
        cost_model_version="cost_v1",
        positions=[TargetPosition(security_id="security_example", target_weight=Decimal("0.01"))],
        solver_status=SolverStatus.FEASIBLE,
        attribution_method="family_equal_contribution",
    )
    base.update(over)
    return TargetBook(**base)


def _risk_review(**over) -> RiskReview:
    base = dict(
        owner="risk",
        created_at=NOW,
        risk_review_id="review_001",
        reviewed_book_id="book_001",
        reviewed_kind="target",
        policy_version="policy_v1",
        effective_at=NOW,
        expires_at=NOW + dt.timedelta(hours=1),
        limit_checks=[
            {
                "name": "gross",
                "threshold": "<=2.0",
                "value": "1.0",
                "status": "pass",
                "evidence": "computed from target weights",
            }
        ],
        decision=RiskDecision.PASS,
    )
    base.update(over)
    return RiskReview(**base)


def _order(**over) -> Order:
    base = dict(
        owner="execution",
        created_at=NOW,
        order_id="order_001",
        target_book_id="book_001",
        security_id="security_example",
        side=OrderSide.BUY,
        quantity_intent=Decimal("1000"),
        conversion_snapshot_id="conv_001",
        allowed_interval_start=NOW,
        allowed_interval_end=NOW + dt.timedelta(days=1),
        participation_bound=Decimal("0.10"),
        expiry_at=NOW + dt.timedelta(days=1),
        cancellation_condition="end of allowed interval",
        expected_cost=Decimal("0.0005"),
        forecast_provenance_id="forecast_example_001",
    )
    base.update(over)
    return Order(**base)


def _fill(**over) -> Fill:
    base = dict(
        owner="execution",
        created_at=NOW,
        fill_id="fill_001",
        order_id="order_001",
        is_simulated=True,
        executed_at=NOW,
        quantity=Decimal("1000"),
        price=Decimal("10.5"),
        currency="USD",
        cost_breakdown={"fees": Decimal("0.0002")},
        accounting_event_id="acct_001",
    )
    base.update(over)
    return Fill(**base)


def _decision_event(**over) -> DecisionEvent:
    base = dict(
        owner="integrity",
        created_at=NOW,
        event_id="event_001",
        actor_kind=ActorKind.CODE,
        actor_id="deterministic_router",
        effective_research_at=NOW,
        effective_market_at=NOW,
        input_artifact_ids=["artifact_001"],
        input_hashes={"artifact_001": "abc123"},
        chosen_action="hold",
        rationale="no binding constraint triggered",
        authorizing_rule_or_certificate_id="rule_execution_baseline_v1",
        resulting_artifact_id="artifact_002",
    )
    base.update(over)
    return DecisionEvent(**base)


def _manifest(**over) -> RunManifest:
    base = dict(
        run_id="run_001",
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
    base.update(over)
    return RunManifest(**base)


# ---------------------------------------------------------------- valid fixtures


def test_source_card_spec_example_is_valid():
    card = _source_card()
    assert card.broker_rejects_formation() is True


def test_source_card_formation_eligible_requires_established_screen():
    card = _source_card(
        status=CardStatus.APPROVED,
        access_confirmed=True,
        formation_eligible=True,
        screen=_screen(),
        unresolved_questions=[],
    )
    assert card.broker_rejects_formation() is False


def test_hypothesis_and_experiment_register_shape():
    hyp = _hypothesis()
    exp = _experiment()
    assert hyp.sealed().verify_seal()
    assert exp.sealed().verify_seal()


def test_forecast_spec_example_is_valid():
    fc = _forecast()
    assert fc.is_expired(dt.datetime(2020, 10, 25, tzinfo=UTC)) is False
    assert fc.is_expired(dt.datetime(2020, 12, 1, tzinfo=UTC)) is True


def test_target_book_and_risk_review_valid():
    book = _target_book()
    review = _risk_review()
    assert review.is_valid_at(NOW) is True
    assert review.is_valid_at(NOW + dt.timedelta(hours=2)) is False
    assert book.solver_status == SolverStatus.FEASIBLE


def test_order_fill_decision_valid():
    order = _order()
    fill = _fill()
    event = _decision_event()
    assert order.quantity_intent is not None and order.dollar_intent is None
    assert fill.is_simulated is True
    assert event.actor_kind == ActorKind.CODE


def test_manifest_valid():
    manifest = _manifest()
    assert manifest.run_id == "run_001"


# ------------------------------------------------------------- invalid fixtures


def test_naive_timestamp_rejected():
    with pytest.raises(Exception):
        _source_card(created_at=dt.datetime(2026, 10, 24, 12, 0))


def test_unlabeled_return_unit_rejected():
    with pytest.raises(Exception):
        _return_forecast(unit="percent")
    with pytest.raises(Exception):
        _return_forecast(unit="basis_points")


def test_forecast_formation_before_cutoff_rejected():
    with pytest.raises(Exception):
        _forecast(earliest_formation_at=dt.datetime(2020, 10, 1, tzinfo=UTC))


def test_forecast_resolution_before_cutoff_rejected():
    with pytest.raises(Exception):
        _forecast(
            target=_target_prediction(expected_resolution_at=dt.datetime(2020, 10, 1, tzinfo=UTC))
        )


def test_formation_eligible_with_unresolved_screen_rejected():
    with pytest.raises(Exception):
        _source_card(
            access_confirmed=True,
            formation_eligible=True,
            screen=_screen(AssessmentStatus.UNRESOLVED),
            unresolved_questions=[],
        )


def test_infeasible_book_with_positions_rejected():
    with pytest.raises(Exception):
        _target_book(solver_status=SolverStatus.INFEASIBLE)


def test_failed_official_constraint_rejected():
    with pytest.raises(Exception):
        _target_book(
            official_constraint_results=[
                {"name": "gross", "threshold": "<=2.0", "value": "2.5", "status": "fail"}
            ]
        )


def test_pass_with_remedy_requires_remedies():
    with pytest.raises(Exception):
        _risk_review(decision=RiskDecision.PASS_WITH_REMEDY)
    with pytest.raises(Exception):
        _risk_review(decision=RiskDecision.BLOCK)


def test_pass_with_remedies_rejected():
    with pytest.raises(Exception):
        _risk_review(remedy_constraints=["reduce gross"])


def test_order_with_both_intents_rejected():
    with pytest.raises(Exception):
        _order(dollar_intent=Decimal("10000"))


def test_agent_decision_without_model_refs_rejected():
    with pytest.raises(Exception):
        _decision_event(actor_kind=ActorKind.AGENT_ROLE, actor_id="risk_agent")


def test_agent_decision_with_model_refs_valid():
    event = _decision_event(
        actor_kind=ActorKind.AGENT_ROLE,
        actor_id="risk_agent",
        model_ref="model_v1",
        prompt_ref="prompt_v1",
        cache_ref="cache_v1",
    )
    assert event.actor_kind == ActorKind.AGENT_ROLE


def test_manifest_latest_version_rejected():
    with pytest.raises(Exception):
        _manifest(security_master_version="latest")


def test_experiment_own_baseline_rejected():
    with pytest.raises(Exception):
        _experiment(
            comparisons={
                "baseline_experiment_ids": ["exp_example_margin_001"],
                "paired_comparison_policy_id": "paired_v1",
                "permitted_sensitivities": [],
            }
        )


def test_duplicate_feature_spec_rejected():
    with pytest.raises(Exception):
        _hypothesis(feature_spec_ids=["a", "a"])


# ------------------------------------------------------------------ typed errors


def test_only_transient_errors_are_retryable():
    err = ToolResponse.failure(
        summary="provider timeout",
        code=ErrorCode.TRANSIENT_PROVIDER_FAILURE,
        retryable=True,
        stop_condition="give up after 3 attempts",
    )
    assert err.error.retryable is True
    with pytest.raises(Exception):
        ToolResponse.failure(
            summary="bad source",
            code=ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE,
            retryable=True,
            stop_condition="never",
        )


def test_retryable_requires_stop_condition():
    with pytest.raises(Exception):
        ToolResponse.failure(
            summary="provider timeout",
            code=ErrorCode.TRANSIENT_PROVIDER_FAILURE,
            retryable=True,
        )


def test_error_response_shape_enforced():
    with pytest.raises(Exception):
        ToolResponse(status="error", summary="boom")
    with pytest.raises(Exception):
        ToolResponse(
            status="success",
            summary="fine",
            error=TypedError(code=ErrorCode.ASOF_VIOLATION),
        )


def test_pod_platform_error_carries_response():
    resp = ToolResponse.failure(
        summary="denied",
        code=ErrorCode.PARTITION_ACCESS_DENIED,
        stop_condition="route through Research Integrity",
    )
    exc = PodPlatformError(resp)
    assert exc.response.error.code == ErrorCode.PARTITION_ACCESS_DENIED
