"""Synthetic checks for the expert routers and the 500-name cap."""
import numpy as np
import pandas as pd
import pytest

from src.capacity_mix import routing
from src.capacity_mix.capacity import cap_names


def _returns(seed: int = 0, months: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.period_range("2021-01", periods=months, freq="M")
    return pd.DataFrame(rng.normal(0.01, 0.03, (months, 4)), index=index, columns=list("abcd"))


@pytest.mark.parametrize("kind,lookback,lam", routing.ROUTER_GRID)
def test_router_weights_use_only_past_returns(kind, lookback, lam):
    net = _returns()
    base = routing.route(net, kind, lookback, lam)
    changed = net.copy()
    changed.iloc[25:] = changed.iloc[25:] * -3.0 + 0.5  # rewrite month 25 onward
    altered = routing.route(changed, kind, lookback, lam)
    pd.testing.assert_frame_equal(base.iloc[:26], altered.iloc[:26])


@pytest.mark.parametrize("kind,lookback,lam", routing.ROUTER_GRID)
def test_router_weights_are_nonnegative_and_sum_to_one_or_cash(kind, lookback, lam):
    weights = routing.route(_returns(seed=1), kind, lookback, lam)
    assert (weights >= -1e-12).all().all()
    totals = weights.sum(axis=1)
    invested = totals > 0
    assert np.allclose(totals[invested], 1.0)
    if kind != "posgate":
        assert invested.all()


def test_router_is_equal_weight_during_burn_in():
    weights = routing.route(_returns(), "perf", 24, 0.5)
    assert np.allclose(weights.iloc[:24].to_numpy(), 0.25)


def test_posgate_goes_to_cash_when_no_expert_has_a_positive_trailing_mean():
    losers = _returns() * 0 - 0.02
    weights = routing.route(losers, "posgate", 12, 1.0)
    assert (weights.iloc[12:].sum(axis=1) == 0).all()


def test_unknown_router_kind_is_rejected():
    with pytest.raises(ValueError):
        routing.route(_returns(), "median", 12, 0.5)


def test_cap_names_keeps_the_largest_absolute_weights_per_month():
    rows = []
    for month in ("2021-01", "2021-02"):
        for permno in range(1, 8):
            sign = 1 if permno % 2 else -1
            rows.append({"target_month": pd.Period(month, "M"), "permno": permno, "target_weight": sign * permno / 100})
    capped = cap_names(pd.DataFrame(rows), max_names=3)
    assert capped.groupby("target_month").size().eq(3).all()
    assert set(capped["permno"]) == {5, 6, 7}
    assert "_abs" not in capped.columns


def test_equal_weights_selection_sums_to_one():
    weights = routing.equal_weights(["a", "b", "c", "d"], ["a", "c"])
    assert weights.to_dict() == {"a": 0.5, "b": 0.0, "c": 0.5, "d": 0.0}
