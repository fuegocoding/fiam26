import datetime as dt
from pathlib import Path

import pytest

from pod_platform.contracts.sources import SourceCard
from pod_platform.research.evaluation import ComparisonPlan, ForecastOutcome, RegisteredEvaluator

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 25, tzinfo=UTC)


@pytest.fixture
def card():
    path = Path(__file__).parents[1] / "configs/source_cards/COMP_chars_panel.json"
    return SourceCard.model_validate_json(path.read_text())


@pytest.fixture
def evaluator(tmp_path):
    value = RegisteredEvaluator(tmp_path / "experiments.sqlite")
    yield value
    value.close()


def plan(**changes):
    return ComparisonPlan.model_validate({
        "experiment_id": "exp1", "owner": "pod1", "hypothesis_id": "hyp1",
        "registered_at": NOW, "target_id": "revenue_growth",
        "source_ids": ["comp_chars_panel"], "variant_budget": 1,
        **changes,
    })


def outcomes():
    return [ForecastOutcome(
        security_id=str(i), formation_at=dt.datetime(2020, month, 1, tzinfo=UTC),
        feature_available_at=dt.datetime(2020, month, 1, tzinfo=UTC),
        label_resolves_at=dt.datetime(2021, month, 1, tzinfo=UTC),
        prediction=i, baseline_prediction=-i, target=i,
        net_forward_return=(-1) ** i * 0.01,
    ) for month in (1, 2, 3) for i in range(4)]


def evaluate(evaluator, card, **kwargs):
    return evaluator.evaluate("exp1", "v1", kwargs.pop("rows", outcomes()),
                              owner=kwargs.pop("owner", "pod1"),
                              evaluation_asof=NOW, source_cards=[card], **kwargs)


def test_paired_comparison_and_idempotent_replay(evaluator, card):
    evaluator.register(plan())
    result = evaluate(evaluator, card)
    assert result["paired_mean_ic_improvement"] == pytest.approx(2)
    assert result["enough_dates"]
    assert result["evaluation_label"] == "retrospective_research"
    assert result["admitted"] is False
    assert evaluate(evaluator, card) == result
    changed = outcomes()
    changed[0].prediction = 42
    with pytest.raises(ValueError, match="different inputs"):
        evaluate(evaluator, card, rows=changed)


def test_budget_and_registration_immutable(evaluator, card):
    evaluator.register(plan())
    assert evaluator.register(plan())
    with pytest.raises(ValueError, match="immutable"):
        evaluator.register(plan(variant_budget=2))
    evaluate(evaluator, card)
    with pytest.raises(ValueError, match="budget"):
        evaluator.evaluate("exp1", "v2", outcomes(), owner="pod1",
                           evaluation_asof=NOW, source_cards=[card])


def test_owner_and_protected_partition(evaluator, card):
    evaluator.register(plan(partition="check"))
    with pytest.raises(PermissionError, match="another pod"):
        evaluate(evaluator, card, owner="pod2")
    with pytest.raises(PermissionError, match="protected"):
        evaluate(evaluator, card)
    assert evaluate(evaluator, card, allowed_partitions=("check",))


def test_future_outcome_and_duplicate_rows(evaluator, card):
    evaluator.register(plan())
    rows = outcomes()
    rows[0].label_resolves_at = NOW + dt.timedelta(days=1)
    with pytest.raises(ValueError, match="not matured"):
        evaluate(evaluator, card, rows=rows)
    rows = outcomes()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(evaluator, card, rows=rows + [rows[0]])


def test_source_gate_and_explicit_exploratory_label(evaluator, card):
    blocked = card.model_copy(update={"formation_eligible": False})
    evaluator.register(plan(research_only=False))
    with pytest.raises(ValueError, match="ineligible"):
        evaluate(evaluator, blocked)
    evaluator.register(plan(experiment_id="exp2"))
    result = evaluator.evaluate("exp2", "v1", outcomes(), owner="pod1",
                                evaluation_asof=NOW, source_cards=[blocked])
    assert result["blocked_formation_sources"] == [card.source_id]
    assert result["research_only"] and not result["admitted"]


def test_flat_baseline_has_no_fabricated_ic(evaluator, card):
    evaluator.register(plan())
    rows = outcomes()
    for row in rows:
        row.baseline_prediction = 0
    result = evaluate(evaluator, card, rows=rows)
    assert not result["enough_dates"]
    assert result["paired_mean_ic_improvement"] is None


def test_temporal_and_finite_validation():
    row = outcomes()[0].model_dump()
    with pytest.raises(ValueError, match="feature not available"):
        ForecastOutcome.model_validate({**row, "feature_available_at": NOW})
    with pytest.raises(ValueError):
        ForecastOutcome.model_validate({**row, "prediction": float("nan")})


def test_prospective_cannot_relabel_historical_work(evaluator, card):
    evaluator.register(plan(partition="prospective"))
    with pytest.raises(ValueError, match="must follow registration"):
        evaluate(evaluator, card, allowed_partitions=("prospective",))
