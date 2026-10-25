import datetime as dt

import pytest

from pod_platform.research.features import FeatureObservation, FeatureRecipe, materialize

UTC = dt.timezone.utc


def obs(month, value, *, security="A", version_month=None, observation_id=None):
    at = dt.datetime(2020, month, 1, tzinfo=UTC)
    return FeatureObservation(observation_id=observation_id or f"{security}-{month}",
                              security_id=security, source_id="source", metric="sales",
                              period_end=at, available_at=at,
                              version_known_at=dt.datetime(2020, version_month or month, 1, tzinfo=UTC),
                              value=value)


def recipe(operation="growth", **changes):
    return FeatureRecipe(feature_id="growth", source_id="source", metric="sales",
                         operation=operation, **changes)


def run(r, rows, securities=None):
    return materialize(r, rows, formation_at=dt.datetime(2020, 3, 1, tzinfo=UTC),
                       security_ids=securities or ["A"])


def test_revisions_and_future_observations_do_not_leak():
    rows = [obs(1, 10), obs(2, 20), obs(4, 1000),
            obs(1, 1000, version_month=4, observation_id="revised")]
    result = run(recipe(), rows)[0]
    assert result["value"] == pytest.approx(1)
    assert result["observation_ids"] == ["A-1", "A-2"]


@pytest.mark.parametrize("operation,lookback,expected", [
    ("latest", 1, 30), ("lag", 1, 20), ("difference", 2, 20),
    ("growth", 2, 2), ("rolling_mean", 3, 20),
])
def test_declared_operators(operation, lookback, expected):
    assert run(recipe(operation, lookback=lookback),
               [obs(1, 10), obs(2, 20), obs(3, 30)])[0]["value"] == expected


def test_missing_is_not_zero_or_renormalized():
    result = run(recipe(), [obs(1, 10)], ["A", "B"])
    assert all(row["value"] is None for row in result)
    assert all(row["missing_reason"] == "insufficient_available_history" for row in result)
    assert run(recipe(), [obs(1, 0), obs(2, 2)])[0]["missing_reason"] == "zero_growth_denominator"


def test_cross_sectional_ties_and_missing():
    result = run(recipe("latest", cross_sectional_rank=True),
                 [obs(1, 10), obs(1, 10, security="B"), obs(1, 20, security="C")],
                 ["A", "B", "C", "D"])
    assert result[0]["value"] == result[1]["value"]
    assert result[2]["value"] > result[0]["value"]
    assert result[3]["value"] is None


def test_conflicting_observation_and_version_fail_closed():
    with pytest.raises(ValueError, match="conflicting content"):
        run(recipe(), [obs(1, 10), obs(1, 20)])
    with pytest.raises(ValueError, match="ambiguous versions"):
        run(recipe(), [obs(1, 10), obs(1, 20, observation_id="other")])
