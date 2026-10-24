"""Part 4 acceptance: research integrity (spec 26.3) and statistical protocol (13)."""

from __future__ import annotations

import datetime as dt
import hashlib

import pytest

from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.research import (
    LabelWindow,
    LateStartPolicy,
    ModelLifecycle,
    ModelState,
    Partition,
    PartitionPlan,
    PromotionCriteria,
    PromotionOutcome,
    RegisteredExperiment,
    Registry,
    ResearchLedger,
    StateTransition,
    TrialRecord,
    embargo_after,
    matured_labels,
    purge_between,
)

UTC = dt.timezone.utc
T0 = dt.datetime(2015, 1, 1, tzinfo=UTC)


def _when(**kw) -> dt.datetime:
    return T0 + dt.timedelta(**kw)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture()
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.sqlite")
    yield reg
    reg.close()


@pytest.fixture()
def ledger(registry):
    return ResearchLedger(registry)


def _registration(**over) -> RegisteredExperiment:
    base = dict(
        owner="strategy_1",
        created_at=T0,
        registration_id="reg_001",
        experiment_spec_id="exp_example_margin_001",
        spec_content_hash=_sha("exp_1"),
        primary_metric="rank_ic",
        baseline_experiment_ids=["traffic_only"],
        promotion_conditions="improvement over baseline on check",
        failure_conditions="no improvement or unstable sign",
        variant_budget=3,
        registered_at=T0,
    )
    base.update(over)
    return RegisteredExperiment(**base)


def _criteria(**over) -> PromotionCriteria:
    base = dict(
        min_coverage="500 stock-months",
        improvement_vs_primary_baseline="positive paired improvement",
        direction_and_economic_size="declared sign, |effect| >= 10bp/month",
        stability_requirement="same sign in both halves",
        return_translation_requirement="net positive after frozen costs",
        implementation_requirement="tradable within ADV bounds",
        permitted_statistical_uncertainty="95% CI excludes zero on paired dates",
        multiple_testing_treatment="Benjamini-Hochberg over registered variants",
    )
    base.update(over)
    return PromotionCriteria(**base)


def _outcome(**over) -> PromotionOutcome:
    base = dict(
        registration_id="reg_001",
        evaluated_at=_when(days=10),
        criteria=_criteria(),
        coverage_met=True,
        improvement_met=True,
        direction_met=True,
        stability_met=True,
        implementation_met=True,
        uncertainty_met=True,
        multiple_testing_met=True,
    )
    base.update(over)
    return PromotionOutcome(**base)


# ------------------------------------------------------ 13.2-13.3 partitions


def test_partition_plan_rejects_overlap():
    with pytest.raises(ValueError):
        PartitionPlan(
            partitions=[
                Partition(name="design", start=T0, end=_when(days=400), label="design"),
                Partition(name="check", start=_when(days=300), end=_when(days=800), label="check"),
            ]
        )


def test_partition_of_date():
    plan = PartitionPlan(
        partitions=[
            Partition(name="design", start=T0, end=_when(days=400), label="design"),
            Partition(name="check", start=_when(days=401), end=_when(days=800), label="check"),
        ]
    )
    assert plan.partition_of(_when(days=100)) == "design"
    assert plan.partition_of(_when(days=500)) == "check"
    assert plan.partition_of(_when(days=2000)) is None


def test_matured_labels_only_resolved_before_cutoff():
    labels = [
        LabelWindow(observation_at=T0, resolves_at=_when(days=10)),
        LabelWindow(observation_at=_when(days=5), resolves_at=_when(days=60)),
    ]
    mature = matured_labels(labels, _when(days=30))
    assert len(mature) == 1
    assert mature[0].resolves_at == _when(days=10)


def test_purge_removes_overlapping_target_windows():
    labels = [
        LabelWindow(observation_at=_when(days=10), resolves_at=_when(days=20)),
        LabelWindow(observation_at=_when(days=35), resolves_at=_when(days=55)),
        LabelWindow(observation_at=_when(days=100), resolves_at=_when(days=110)),
    ]
    kept, purged = purge_between(labels, _when(days=50), _when(days=90))
    assert len(kept) == 2
    assert len(purged) == 1
    assert purged[0].observation_at == _when(days=35)


def test_embargo_excludes_overlap_window():
    labels = [
        LabelWindow(observation_at=_when(days=10), resolves_at=_when(days=20)),
        LabelWindow(observation_at=_when(days=45), resolves_at=_when(days=55)),
        LabelWindow(observation_at=_when(days=200), resolves_at=_when(days=210)),
    ]
    kept, embargoed = embargo_after(labels, _when(days=40), _when(days=100))
    assert len(embargoed) == 1
    assert embargoed[0].observation_at == _when(days=45)
    assert len(kept) == 2


def test_late_start_policy():
    policy = LateStartPolicy(
        source_id="CA0060", first_eligible_at=_when(days=1500), rule="first observation + history"
    )
    assert not policy.eligible(_when(days=1000))
    assert policy.eligible(_when(days=1500))


# ------------------------------------------------------ 13.1, 13.8 registration


def test_registration_is_immutable(ledger):
    ledger.register(_registration())
    with pytest.raises(PodPlatformError):
        ledger.register(_registration())


def test_trial_counters_by_kind(ledger):
    ledger.register(_registration())
    spec_hash = _sha("exp_1")
    for i in range(4):
        ledger.record_trial(
            TrialRecord(
                kind="hyperparameter_config",
                spec_hash=spec_hash,
                detail=f"config {i}",
                counted_at=_when(days=i),
            )
        )
    ledger.record_trial(
        TrialRecord(kind="seed_restart", spec_hash=spec_hash, detail="restart", counted_at=_when(days=9))
    )
    assert ledger.trial_count() == 5
    assert ledger.trial_count("hyperparameter_config") == 4
    assert ledger.trial_count("seed_restart") == 1
    assert ledger.trial_count("hypothesis") == 0


def test_variant_budget_enforced(ledger):
    ledger.register(_registration(variant_budget=2))
    spec_hash = _sha("exp_1")
    for i in range(2):
        ledger.record_trial(
            TrialRecord(
                kind="scored_variant", spec_hash=spec_hash, detail=f"variant {i}", counted_at=_when(days=i)
            )
        )
    with pytest.raises(PodPlatformError) as exc:
        ledger.enforce_variant_budget("reg_001")
    assert exc.value.response.error.code == ErrorCode.PROVIDER_BUDGET_EXCEEDED


def test_unknown_trial_kind_rejected():
    with pytest.raises(ValueError):
        TrialRecord(kind="mystery", spec_hash=_sha("x"), detail="d", counted_at=T0)


def test_outcome_access_ledgered(ledger, registry):
    ledger.record_outcome_access("acc_1", "check", "strategy_1", "peek", 100, authorized=True)
    assert ledger.outcome_accesses() == [
        {
            "access_id": "acc_1",
            "partition": "check",
            "actor": "strategy_1",
            "purpose": "peek",
            "record_count": 100,
        }
    ]
    with pytest.raises(PodPlatformError):
        ledger.record_outcome_access("acc_2", "final", "strategy_1", "peek", 100)


# ------------------------------------------------------------ 14 lifecycle


def test_legal_lifecycle_path():
    life = ModelLifecycle()
    seq = [
        (ModelState.IDEA, ModelState.REGISTERED),
        (ModelState.REGISTERED, ModelState.DATA_READY),
        (ModelState.DATA_READY, ModelState.DEVELOPMENT),
        (ModelState.DEVELOPMENT, ModelState.CHALLENGED),
        (ModelState.CHALLENGED, ModelState.CHECK_ELIGIBLE),
        (ModelState.CHECK_ELIGIBLE, ModelState.SHADOW),
        (ModelState.SHADOW, ModelState.ADMITTED),
        (ModelState.ADMITTED, ModelState.MONITORED),
        (ModelState.MONITORED, ModelState.RETIRED),
    ]
    for from_state, to_state in seq:
        life.transition(
            StateTransition(
                model_id="model_1",
                from_state=from_state,
                to_state=to_state,
                authority="integrity",
                reason="evidence accepted",
                at=T0,
            )
        )
    assert life.state("model_1") == ModelState.RETIRED
    assert life.admitted("model_1") is False
    assert len(life.history("model_1")) == len(seq)


def test_illegal_transition_rejected():
    life = ModelLifecycle()
    with pytest.raises(ValueError):
        StateTransition(
            model_id="model_1",
            from_state=ModelState.IDEA,
            to_state=ModelState.ADMITTED,
            authority="strategy_1",
            reason="wishful thinking",
            at=T0,
        )


def test_serialized_transitions_enforced():
    life = ModelLifecycle()
    life.transition(
        StateTransition(
            model_id="model_1",
            from_state=ModelState.IDEA,
            to_state=ModelState.REGISTERED,
            authority="integrity",
            reason="registered",
            at=T0,
        )
    )
    with pytest.raises(PodPlatformError):
        life.transition(
            StateTransition(
                model_id="model_1",
                from_state=ModelState.IDEA,
                to_state=ModelState.REGISTERED,
                authority="integrity",
                reason="stale transition",
                at=_when(days=1),
            )
        )


# ------------------------------------------------------- 13.10 promotion gate


def test_promotion_denied_when_any_condition_fails():
    outcome = _outcome(stability_met=False)
    with pytest.raises(PodPlatformError) as exc:
        outcome.certificate_or_raise(
            model_id="model_1",
            model_hash=_sha("model"),
            feature_hashes={"f1": _sha("f1")},
            data_hashes={"d1": _sha("d1")},
            approved_universe="us_public_equities_v1",
            target_id="next_reported_margin_change",
            horizon_trading_days=20,
            effective_start=T0,
            approved_mode="contemporary_research",
            expiry_policy="review after 12 months",
            promotion_evidence_ids=["ev_1"],
        )
    assert exc.value.response.error.code == ErrorCode.ADMISSION_CERTIFICATE_INVALID


def test_promotion_issues_certificate_when_all_conditions_met():
    outcome = _outcome()
    assert outcome.admitted is True
    cert = outcome.certificate_or_raise(
        model_id="model_1",
        model_hash=_sha("model"),
        feature_hashes={"f1": _sha("f1")},
        data_hashes={"d1": _sha("d1")},
        approved_universe="us_public_equities_v1",
        target_id="next_reported_margin_change",
        horizon_trading_days=20,
        effective_start=T0,
        approved_mode="contemporary_research",
        expiry_policy="review after 12 months",
        promotion_evidence_ids=["ev_1"],
    )
    assert cert.is_valid_at(T0) is True
    assert cert.is_valid_at(_when(days=-1)) is False
    assert cert.model_id == "model_1"
