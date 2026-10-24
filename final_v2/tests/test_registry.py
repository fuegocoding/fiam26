"""W01 acceptance: registry immutability, idempotency, access control (spec 22.3, 26.3)."""

from __future__ import annotations

import pytest

from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.research.registry import Registry

from .test_contracts import _decision_event, _fill, _forecast, _hypothesis, _order, _target_book


@pytest.fixture()
def registry(tmp_path):
    reg = Registry(tmp_path / "registry.sqlite")
    yield reg
    reg.close()


def test_register_and_get_roundtrip(registry):
    record_id = registry.register("hypotheses", _hypothesis())
    assert record_id == "hyp_example_margin_001"
    stored = registry.get("hypotheses", record_id)
    assert stored["mechanism_family"] == "discount_driven_volume"
    assert registry.count("hypotheses") == 1


def test_duplicate_registration_rejected(registry):
    registry.register("hypotheses", _hypothesis())
    with pytest.raises(PodPlatformError):
        registry.register("hypotheses", _hypothesis())


def test_forecast_requires_existing_model_and_certificate(registry):
    with pytest.raises(PodPlatformError) as exc:
        registry.register("forecast_publications", _forecast())
    assert exc.value.response.error.code == ErrorCode.SCHEMA_MISMATCH


def test_fill_requires_existing_order(registry):
    with pytest.raises(PodPlatformError):
        registry.register("fills", _fill())
    registry.register("target_books", _target_book())
    registry.register("orders", _order())
    fill_id = registry.register("fills", _fill())
    assert fill_id == "fill_001"


def test_order_requires_existing_target_book(registry):
    with pytest.raises(PodPlatformError):
        registry.register("orders", _order())
    registry.register("target_books", _target_book())
    assert registry.register("orders", _order()) == "order_001"


def test_run_attempt_is_idempotent(registry):
    first = registry.record_run_attempt(
        attempt_id="attempt_1",
        idempotency_key="key_1",
        state="completed",
        spec_hash="h" * 64,
        partition="design",
        artifact_ids=["art_2", "art_1"],
    )
    assert first["idempotent_replay"] is False
    second = registry.record_run_attempt(
        attempt_id="attempt_2",
        idempotency_key="key_1",
        state="completed",
        spec_hash="h" * 64,
        partition="design",
        artifact_ids=["art_1"],
    )
    assert second["idempotent_replay"] is True
    assert second["attempt_id"] == "attempt_1"
    assert second["artifact_ids"] == ["art_1", "art_2"]


def test_protected_partition_denied_without_authorization(registry):
    with pytest.raises(PodPlatformError) as exc:
        registry.record_evaluation_access(
            access_id="acc_1",
            partition="check",
            actor="strategy_1",
            purpose="peek",
            record_count=10,
        )
    assert exc.value.response.error.code == ErrorCode.PARTITION_ACCESS_DENIED
    registry.record_evaluation_access(
        access_id="acc_2",
        partition="design",
        actor="strategy_1",
        purpose="development",
        record_count=10,
    )
    registry.record_evaluation_access(
        access_id="acc_3",
        partition="check",
        actor="referee",
        purpose="controlled evaluation",
        record_count=10,
        authorized=True,
    )


def test_trial_counters_are_separate_ledgers(registry):
    registry.count_trial("t1", "hypothesis", "a" * 64)
    registry.count_trial("t2", "hyperparameter_config", "a" * 64)
    registry.count_trial("t3", "hyperparameter_config", "b" * 64)
    assert registry.trial_count() == 3
    assert registry.trial_count("hyperparameter_config") == 2
    assert registry.trial_count("hypothesis") == 1


def test_decision_event_ledger_append_only(registry):
    registry.register("decision_events", _decision_event())
    assert registry.count("decision_events") == 1
    with pytest.raises(PodPlatformError):
        registry.register("decision_events", _decision_event())


def test_unknown_table_rejected(registry):
    with pytest.raises(ValueError):
        registry.register("not_a_table", _hypothesis())
