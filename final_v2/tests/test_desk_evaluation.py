"""Actual financial-outcome plumbing with declared deterministic fixtures."""

import copy
import datetime as dt
import json
from types import SimpleNamespace

import pytest

from pod_platform.application import desk_evaluation
from pod_platform.contracts.base import canonical_digest
from pod_platform.research.builtins.timeutil import (
    month_end_known_at,
    month_index,
)
from pod_platform.research.composition import (
    CompositionService,
    observation_digest,
)
from pod_platform.research.features import FeatureObservation
from pod_platform.research.jobs import open_sqlite

NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)
SOURCE = "comp_chars_panel"
LABEL = "ret_exc_lead1m"


def _observations(first, months):
    """Create labeled fixtures, never used by the live workflow."""
    rows = []
    for offset in range(months):
        month = first + offset
        for index in range(6):
            for metric, value in (
                ("x", index + 1),
                ("y", 2 * (index + 1)),
                ("base", -index - 1),
                ("native", 20 + index + offset),
                # Explicit forward outcomes are keyed by realization month.
                (LABEL, 0.002 * (index - 2) * (1 + 0.2 * (offset - 1))),
            ):
                day = month_end_known_at(month)
                rows.append(
                    FeatureObservation(
                        observation_id=f"fixture:{month}:{index}:{metric}",
                        source_id=SOURCE,
                        security_id=str(index),
                        metric=metric,
                        value=value,
                        period_end=day,
                        available_at=day,
                        version_known_at=day,
                    )
                )
    return rows


@pytest.fixture
def world(tmp_path):
    first = month_index(dt.date(2018, 1, 1))
    rows = _observations(first, 6)
    service = CompositionService(
        tmp_path / "compositions.sqlite",
        papers=None,
        namespace="pod_fundamentals",
        clock=lambda: NOW,
        code_digest="a" * 64,
    )
    nodes = [
        {
            "node_id": name,
            "operation": "input",
            "recipe": {
                "feature_id": name,
                "source_id": SOURCE,
                "metric": name,
                "operation": "latest",
            },
        }
        for name in ("x", "y", "base")
    ] + [{"node_id": "combined", "operation": "product", "inputs": ["x", "y"]}]
    plan = {
        "composition_id": "composition_fixture",
        "namespace": "pod_fundamentals",
        "scope": "research_only",
        "registered_at": NOW.isoformat(),
        "spec": {
            "nodes": nodes,
            "outputs": ["combined"],
            "baselines": ["base"],
            "target_id": "native_target",
        },
        "grant": {
            "source_metrics": {SOURCE: ["x", "y", "base", "native", LABEL]},
            "security_ids": [str(index) for index in range(6)],
            "formation_dates": [
                month_end_known_at(first + month).isoformat()
                for month in range(4)
            ],
            "targets": {
                "native_target": {
                    "source_id": SOURCE,
                    "metric": "native",
                    "horizon_months": 1,
                    "resolve_lag_days": 0,
                }
            },
        },
    }
    service._put(plan["composition_id"], "plan", plan)
    service._put(
        "result:" + plan["composition_id"], "result", {"status": "measured"}
    )
    calls = []

    def prepare(namespace, identifier):
        def load():
            with open_sqlite(service.path) as connection:
                count = connection.execute(
                    "SELECT COUNT(*) FROM compositions WHERE kind='desk_trial'"
                ).fetchone()[0]
            calls.append(count)
            return rows

        service.bind_inputs(identifier, {SOURCE: observation_digest(rows)})
        return load

    workflow = SimpleNamespace(
        application={"run_id": "fixture_run"},
        settings={
            "research": {
                "desk_evaluation": {
                    "min_training_periods": 3,
                    "min_training_psr": 0.5,
                    "min_names_per_date": 3,
                    "min_validation_psr": 0.5,
                },
                "periods": {
                    "validation_start": "2020-01-01",
                    "validation_end": "2020-12-31",
                },
            }
        },
        clock=lambda: NOW,
        loop=SimpleNamespace(compositions={"pod_fundamentals": service}),
        prepare_composition=prepare,
    )
    return SimpleNamespace(
        workflow=workflow,
        plans=[plan],
        rows=rows,
        calls=calls,
        service=service,
        first=first,
    )


def test_training_uses_shifted_return_labels_costs_and_all_variants(world):
    report = desk_evaluation.evaluate_training(world.workflow, world.plans)
    assert world.calls == [4]  # All reservations precede the first loader.
    assert report["total_attempts"] == 4
    assert report["attempts_complete"] is True
    combined = next(
        row for row in report["candidates"] if row["output_node"] == "combined"
    )
    assert combined["native_measurement_status"] == "measured"
    assert combined["status"] == "measured"
    assert combined["series"][0]["gross"] == pytest.approx(0.01)
    assert combined["series"][0]["turnover"] == 2
    assert combined["series"][0]["net"] == pytest.approx(0.008)
    assert combined["series"][0]["stress_net"] == pytest.approx(0.006)
    assert combined["series"][1]["gross"] == pytest.approx(0.012)
    assert combined["series"][1]["turnover"] == 0
    assert combined["metrics"]["deflated_sharpe_status"] == "available"
    assert (
        combined["cost_stress_2x_metrics"]["cumulative_net_return"]
        < (combined["metrics"]["cumulative_net_return"])
    )
    assert combined["eligible_for_validation"]
    assert report["eligible_candidate_ids"] == [combined["variant_id"]]
    assert report["admitted"] is False
    assert report["short_book_assumption"] == "hypothetical_research_short_book"
    assert (
        canonical_digest(
            {k: v for k, v in report.items() if k != "training_hash"}
        )
        == report["training_hash"]
    )
    json.dumps(report, allow_nan=False)


def test_missing_return_month_fails_without_compressing_frequency(world):
    missing = month_end_known_at(world.first + 2)
    world.rows[:] = [
        row
        for row in world.rows
        if not (row.metric == LABEL and row.period_end == missing)
    ]
    report = desk_evaluation.evaluate_training(world.workflow, world.plans)
    assert report["total_attempts"] == 4
    assert report["attempts_complete"] is False
    assert all(row["status"] == "failed" for row in report["candidates"])
    assert report["eligible_candidate_ids"] == []


def test_failed_component_is_counted_and_makes_dsr_unavailable(world):
    world.rows[:] = [row for row in world.rows if row.metric != "y"]
    report = desk_evaluation.evaluate_training(world.workflow, world.plans)
    assert report["total_attempts"] == 4
    assert report["attempts_complete"] is False
    measured = [c for c in report["candidates"] if c["status"] == "measured"]
    assert {row["output_node"] for row in measured} == {"x", "base"}
    assert all(
        row["metrics"]["deflated_sharpe_ratio"] is None for row in measured
    )
    assert all(row["metrics"]["total_attempts"] == 4 for row in measured)


@pytest.mark.parametrize(
    "proposal_status", ["model_failed", "held_review", None]
)
def test_unmeasured_model_hypotheses_remain_counted(world, proposal_status):
    world.service._put("desk_failed", "desk_request", {"fixture": True})
    if proposal_status:
        world.service._put(
            "desk_result:desk_failed",
            "desk_result",
            {"status": proposal_status},
        )
    report = desk_evaluation.evaluate_training(world.workflow, world.plans)
    assert report["total_attempts"] == 5
    assert len(report["registered_attempt_ids"]) == 5
    assert report["attempts_complete"] is False
    assert len(report["unmeasured_hypotheses"]) == 1
    measured = [c for c in report["candidates"] if c["status"] == "measured"]
    assert len(measured) == 4
    assert all(c["metrics"]["deflated_sharpe_ratio"] is None for c in measured)
    assert all(c["metrics"]["total_attempts"] == 5 for c in measured)


def test_same_run_cannot_expand_variant_registry_after_return_access(world):
    desk_evaluation.evaluate_training(world.workflow, world.plans)
    extra = copy.deepcopy(world.plans[0])
    extra["composition_id"] = "composition_new_hypothesis"
    world.service._put(extra["composition_id"], "plan", extra)
    with pytest.raises(ValueError, match="immutable composition"):
        desk_evaluation.evaluate_training(world.workflow, world.plans + [extra])
    assert len(world.calls) == 1


def test_training_rerun_reuses_same_fixed_registry(world):
    first = desk_evaluation.evaluate_training(world.workflow, world.plans)
    repeated = desk_evaluation.evaluate_training(world.workflow, world.plans)
    assert first == repeated


def test_global_declared_warmup_keeps_march_2015_to_september_2019(world):
    first = month_index(dt.date(2015, 1, 1))
    world.rows[:] = _observations(first, 60)
    rolling = copy.deepcopy(world.plans[0])
    rolling["composition_id"] = "composition_rolling_fixture"
    rolling["grant"]["formation_dates"] = [
        month_end_known_at(first + offset).isoformat() for offset in range(57)
    ]
    rolling["spec"]["nodes"][1]["recipe"].update(
        operation="rolling_mean", lookback=3
    )
    simple = copy.deepcopy(rolling)
    simple["composition_id"] = "composition_simple_fixture"
    simple["spec"]["nodes"][1]["recipe"].update(operation="latest", lookback=1)
    for plan in (rolling, simple):
        world.service._put(plan["composition_id"], "plan", plan)
        world.service._put(
            "result:" + plan["composition_id"],
            "result",
            {"status": "measured"},
        )
    report = desk_evaluation.evaluate_training(
        world.workflow, [rolling, simple]
    )
    assert report["warmup_periods"] == 2
    assert report["required_history_periods"] == 3
    assert report["actual_training_start"].startswith("2015-03-31")
    assert report["attempts_complete"] is True
    assert len(report["candidates"]) == 8
    for candidate in report["candidates"]:
        assert candidate["status"] == "measured"
        assert candidate["metrics"]["periods"] == 55
        assert candidate["series"][0]["formation_at"].startswith("2015-03-31")
        assert candidate["series"][-1]["formation_at"].startswith("2019-09-30")
        assert candidate["series"][0]["net"] != 0
        reserved = world.service._get(
            "desk_trial:fixture_run:" + candidate["variant_id"]
        )
        assert reserved["warmup_periods"] == 2
        assert len(reserved["formation_dates"]) == 55
        assert reserved["actual_training_start"].startswith("2015-03-31")


@pytest.mark.parametrize(
    ("operation", "lookback", "required"),
    [
        ("latest", 8, 1),
        ("lag", 3, 4),
        ("difference", 2, 3),
        ("growth", 2, 3),
        ("rolling_mean", 3, 3),
    ],
)
def test_required_history_matches_feature_recipe_semantics(
    world, operation, lookback, required
):
    plan = copy.deepcopy(world.plans[0])
    plan["spec"]["nodes"][0]["recipe"].update(
        operation=operation, lookback=lookback
    )
    assert desk_evaluation._required_history(plan) == required


def _validation_loader(world, monkeypatch, *, interrupt=False):
    calls = []
    first = month_index(dt.date(2020, 1, 1))
    rows = _observations(first, 12)

    def prepare(config, security_ids, clock, *, reservation_id, freeze_digest):
        calls.append(reservation_id)
        reserved = world.service._get(reservation_id)
        assert reserved["freeze_digest"] == freeze_digest
        assert world.service._get("desk_validation_freeze:fixture_run")
        if interrupt:
            raise KeyboardInterrupt("fixture interrupted after reservation")
        return {
            "observation_source": {"source_id": SOURCE},
            "snapshot": {"freeze_digest": freeze_digest},
        }

    def load(bindings):
        return rows

    monkeypatch.setattr(desk_evaluation, "prepare_validation_inputs", prepare)
    world.workflow._load_observations = load
    return calls


def test_validation_freezes_before_single_load_and_reuses_result(
    world, monkeypatch
):
    training = desk_evaluation.evaluate_training(world.workflow, world.plans)
    calls = _validation_loader(world, monkeypatch)
    validation = desk_evaluation.evaluate_validation(
        world.workflow, world.plans, training
    )
    assert calls == ["validation_access:fixture_run"]
    assert validation["status"] == "measured"
    assert len(validation["candidates"]) == 1
    candidate = validation["candidates"][0]
    assert candidate["status"] == "measured"
    assert candidate["metrics"]["periods"] == 11
    assert candidate["series"][0]["formation_at"].startswith("2020-01")
    assert candidate["series"][-1]["formation_at"].startswith("2020-11")
    assert candidate["metrics"]["deflated_sharpe_ratio"] is None
    assert candidate["metrics"]["total_attempts"] == 4
    assert validation["evaluation_accessed"] is False
    assert validation["admitted"] is False
    repeated = desk_evaluation.evaluate_validation(
        world.workflow, world.plans, training
    )
    assert repeated == validation
    assert len(calls) == 1


def test_interrupted_validation_cannot_reopen_access(world, monkeypatch):
    training = desk_evaluation.evaluate_training(world.workflow, world.plans)
    calls = _validation_loader(world, monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        desk_evaluation.evaluate_validation(
            world.workflow, world.plans, training
        )
    with pytest.raises(ValueError, match="already reserved"):
        desk_evaluation.evaluate_validation(
            world.workflow, world.plans, training
        )
    assert len(calls) == 1


def test_validation_rejects_changed_training_report_before_loading(
    world, monkeypatch
):
    training = desk_evaluation.evaluate_training(world.workflow, world.plans)
    calls = _validation_loader(world, monkeypatch)
    changed = copy.deepcopy(training)
    changed["candidates"][0]["metrics"]["mean_periodic_net_return"] = 1
    with pytest.raises(ValueError, match="Training evidence"):
        desk_evaluation.evaluate_validation(
            world.workflow, world.plans, changed
        )
    assert calls == []
    assert world.service._get("validation_access:fixture_run") is None


def test_validation_without_eligible_training_does_not_open_data(
    world, monkeypatch
):
    world.workflow.settings["research"]["desk_evaluation"][
        "min_training_periods"
    ] = 24
    training = desk_evaluation.evaluate_training(world.workflow, world.plans)
    calls = _validation_loader(world, monkeypatch)
    validation = desk_evaluation.evaluate_validation(
        world.workflow, world.plans, training
    )
    assert validation["status"] == "no_training_candidates"
    assert not validation["validation_accessed"]
    assert calls == []


def test_three_month_native_target_purges_validation_maturity():
    workflow = SimpleNamespace(
        settings={
            "research": {
                "periods": {
                    "validation_start": "2020-01-01",
                    "validation_end": "2020-12-31",
                }
            }
        }
    )
    dates = desk_evaluation._validation_dates(
        workflow, {"horizon_months": 3, "resolve_lag_days": 0}
    )
    assert len(dates) == 9
    assert dates[-1].month == 9


def test_count_unmeasured_deflation_enables_dsr_gate(world):
    world.service._put("desk_failed", "desk_request", {"fixture": True})
    world.workflow.settings["research"]["desk_evaluation"].update(
        deflation="count_unmeasured", training_gate="dsr", min_training_dsr=0.5
    )
    report = desk_evaluation.evaluate_training(world.workflow, world.plans)
    assert report["total_attempts"] == 5
    assert report["attempts_complete"] is True
    measured = [c for c in report["candidates"] if c["status"] == "measured"]
    assert all(
        c["metrics"]["deflated_sharpe_status"] == "available" for c in measured
    )
    for candidate in measured:
        dsr = candidate["metrics"]["deflated_sharpe_ratio"]
        assert candidate["eligible_for_validation"] == (
            "candidate" in candidate["roles"] and dsr >= 0.5
        )


def test_pooled_validation_tests_training_and_validation_together(
    world, monkeypatch
):
    world.workflow.settings["research"]["desk_evaluation"].update(
        validation_rule="pooled"
    )
    training = desk_evaluation.evaluate_training(world.workflow, world.plans)
    _validation_loader(world, monkeypatch)
    validation = desk_evaluation.evaluate_validation(
        world.workflow, world.plans, training
    )
    candidate = validation["candidates"][0]
    trained = next(
        c
        for c in training["candidates"]
        if c["variant_id"] == candidate["variant_id"]
    )
    pooled = candidate["pooled_metrics"]
    assert pooled["periods"] == (
        trained["metrics"]["periods"] + candidate["metrics"]["periods"]
    )
    assert candidate["passed_research_validation"] == (
        pooled["probabilistic_sharpe_ratio"] >= 0.5
        and candidate["metrics"]["mean_periodic_net_return"] > 0
    )


def test_unknown_gate_policy_is_rejected(world):
    world.workflow.settings["research"]["desk_evaluation"].update(
        validation_rule="lenient"
    )
    with pytest.raises(ValueError, match="Unknown training gate"):
        desk_evaluation.evaluate_training(world.workflow, world.plans)
