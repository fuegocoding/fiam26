"""ResearchLedger is durable through Registry: restart recovery, atomic reservations, no counter collisions."""

import datetime as dt
import sqlite3
import threading
from pathlib import Path

import pytest

from pod_platform.contracts.errors import ErrorCode, PodPlatformError
from pod_platform.papers import adaptations
from pod_platform.research.registration import RegisteredExperiment, ResearchLedger, TrialRecord
from pod_platform.research.registry import Registry

T0 = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
HASH = "ab" * 32


def _reg(registration_id="reg_1", budget=2, spec_hash=HASH):
    return RegisteredExperiment(owner="strategy_1", created_at=T0, registration_id=registration_id, experiment_spec_id="exp_1",
                                spec_content_hash=spec_hash, primary_metric="rank_ic", baseline_experiment_ids=["base"],
                                promotion_conditions="beats baseline", failure_conditions="does not", variant_budget=budget, registered_at=T0)


def _variant(detail, spec_hash=HASH):
    return TrialRecord(kind="scored_variant", spec_hash=spec_hash, detail=detail, counted_at=T0)


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "registry.sqlite"


def _open(path):
    registry = Registry(path)
    return registry, ResearchLedger(registry)


def test_registration_trials_and_budget_survive_restart(path):
    registry, ledger = _open(path)
    ledger.register(_reg())
    assert ledger.reserve_trial("reg_1", _variant("v1"))["replayed"] is False
    registry.close()

    registry, ledger = _open(path)  # fresh process: empty in-memory state
    assert ledger.registration("reg_1").variant_budget == 2
    assert ledger.trial_count() == 2 and ledger.trial_count("scored_variant") == 1
    assert ledger.variant_budget_remaining("reg_1") == 1
    known, created = ledger.ensure_registered(_reg())  # recovered from the stored record, counts not reset
    assert created is False and known.registration_id == "reg_1" and ledger.trial_count() == 2
    with pytest.raises(PodPlatformError):
        ledger.register(_reg())  # duplicate registration stays immutable
    with pytest.raises(PodPlatformError):
        ledger.ensure_registered(_reg(spec_hash="cd" * 32))  # same id, different content
    ledger.reserve_trial("reg_1", _variant("v2"))
    with pytest.raises(PodPlatformError) as exc:
        ledger.enforce_variant_budget("reg_1")
    assert exc.value.response.error.code == ErrorCode.PROVIDER_BUDGET_EXCEEDED
    registry.close()


def test_no_global_counter_collision_between_ledgers_or_restarts(path):
    (r1, l1), (r2, l2) = _open(path), _open(path)
    l1.register(_reg("reg_a", spec_hash="aa" * 32))
    l2.register(_reg("reg_b", spec_hash="bb" * 32))  # both would have used trial_1 before
    for i in range(3):
        l1.record_trial(_variant(f"x{i}", "aa" * 32))
        l2.record_trial(_variant(f"y{i}", "bb" * 32))
    assert l1.trial_count() == l2.trial_count() == 8
    assert r1.trial_count() == 8  # legacy trial_counts mirror carries the same unique ids
    r1.close(), r2.close()


def test_reservation_retry_is_free_new_variant_spends_and_refusal_is_audited(path):
    registry, ledger = _open(path)
    ledger.register(_reg(budget=2))
    first = ledger.reserve_trial("reg_1", _variant("cand:design_base"))
    retry = ledger.reserve_trial("reg_1", _variant("cand:design_base"))  # cached same experiment
    assert retry["replayed"] is True and retry["trial_id"] == first["trial_id"] and ledger.variant_budget_remaining("reg_1") == 1
    ledger.reserve_trial("reg_1", _variant("cand:placebo"))  # genuinely new variant spends before it runs
    assert ledger.variant_budget_remaining("reg_1") == 0
    before = ledger.trial_count()
    with pytest.raises(PodPlatformError) as exc:
        ledger.reserve_trial("reg_1", _variant("cand:third"))
    assert exc.value.response.error.code == ErrorCode.PROVIDER_BUDGET_EXCEEDED and ledger.trial_count() == before
    assert ledger.reserve_trial("reg_1", _variant("cand:placebo"))["replayed"] is True  # replays still succeed when exhausted
    audit = registry.research_audit_log("reservation_denied")
    assert len(audit) == 1 and audit[0]["subject"] == "reg_1" and "cand:third" in audit[0]["detail"]
    registry.close()


def test_concurrent_reservations_never_overspend(path):
    registry, ledger = _open(path)
    ledger.register(_reg(budget=3))
    outcomes = []

    def worker(i):
        reg, led = _open(path)  # separate connection per thread, as separate processes would have
        try:
            led.reserve_trial("reg_1", _variant(f"variant_{i}"))
            outcomes.append("ok")
        except PodPlatformError as exc:
            outcomes.append(exc.response.error.code)
        finally:
            reg.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert outcomes.count("ok") == 3 and outcomes.count(ErrorCode.PROVIDER_BUDGET_EXCEEDED) == 7
    assert registry.research_budgeted_used(HASH) == 3 and ledger.variant_budget_remaining("reg_1") == 0
    assert len(registry.research_audit_log("reservation_denied")) == 7
    registry.close()


def test_concurrent_identical_reservation_spends_once(path):
    registry, ledger = _open(path)
    ledger.register(_reg(budget=5))
    replays = []

    def worker():
        reg, led = _open(path)
        replays.append(led.reserve_trial("reg_1", _variant("same"))["replayed"])
        reg.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert replays.count(False) == 1 and ledger.variant_budget_remaining("reg_1") == 4
    registry.close()


def test_concurrent_registration_has_one_winner(path):
    registry, _ = _open(path)
    results = []

    def worker():
        reg, led = _open(path)
        try:
            led.register(_reg())
            results.append("ok")
        except PodPlatformError:
            results.append("duplicate")
        reg.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 1 and registry.research_trial_count("registered_specification") == 1
    registry.close()


def test_outcome_access_is_durable_idempotent_and_failures_are_audited(path):
    registry, ledger = _open(path)
    ledger.record_outcome_access("acc_1", "check", "integrity", "limited evaluation", 10, authorized=True)
    ledger.record_outcome_access("acc_1", "check", "integrity", "limited evaluation", 10, authorized=True)  # replay: one event
    with pytest.raises(PodPlatformError):
        ledger.record_outcome_access("acc_1", "check", "integrity", "a different purpose", 10, authorized=True)
    with pytest.raises(PodPlatformError) as exc:
        ledger.record_outcome_access("acc_2", "final", "strategy_1", "peek", 5)
    assert exc.value.response.error.code == ErrorCode.PARTITION_ACCESS_DENIED
    registry.close()

    registry, ledger = _open(path)
    assert [a["access_id"] for a in ledger.outcome_accesses()] == ["acc_1"]
    assert registry._conn.execute("SELECT COUNT(*) FROM evaluation_access").fetchone()[0] == 1
    assert [a["subject"] for a in registry.research_audit_log("access_refused")] == ["acc_1", "acc_2"]
    registry.close()


def test_durable_tables_reject_update_and_delete(path):
    registry, ledger = _open(path)
    ledger.register(_reg())
    registry.research_audit("probe", "reg_1", "row so the delete trigger has something to refuse")
    for sql in ("UPDATE research_trials SET kind='x'", "DELETE FROM research_trials", "UPDATE research_registrations SET spec_hash='x'",
                "DELETE FROM research_audit"):
        with pytest.raises(sqlite3.DatabaseError):
            registry._conn.execute(sql)
    registry.close()


def test_paper_adapter_no_longer_swallows_integrity_errors():
    assert "IntegrityError" not in Path(adaptations.__file__).read_text(encoding="utf-8")
