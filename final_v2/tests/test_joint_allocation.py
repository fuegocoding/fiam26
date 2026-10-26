"""Dollar construction enforces shared orders after signal evidence is frozen."""

import numpy as np
import pandas as pd
import pytest

from pod_platform.research.joint_allocation import (
    allocate_jointly,
    capital_bound,
)


def settings():
    return {
        "max_daily_participation": 0.1,
        "order_ttl_sessions": 10,
        "max_borrow_fraction_market_cap": 0.001,
        "max_borrow_adv_days": 5,
        "impact_eta": 0.5,
        "max_pod_capital": 0.5,
        "single_name_limit": 0.05,
        "gross_limit": 2,
    }


def frame(adv=1e6):
    return pd.DataFrame(
        {
            "dolvol_126d": [adv, adv],
            "me": [20000.0, 20000.0],
            "rvol_21d": [0.02, 0.02],
        }
    )


def test_capacity_uses_exact_existing_dollars_at_new_funding_size():
    target = np.array([0.04, -0.04])
    prior = target * 1e7
    empty = capital_bound(target, np.zeros(2), frame(1e5), settings(), 1e9)
    held = capital_bound(target, prior, frame(1e5), settings(), 1e9)
    assert empty == pytest.approx(2.5e6)
    assert held == pytest.approx(
        12.5e6
    )  # Borrow inventory binds before liquidity.
    # The allowed reduction exception cannot authorize larger new risk than
    # the actual carried dollars when it dominates the ten-session allowance.
    assert np.max(abs(held * target - prior)) <= max(1e5, abs(prior).max())


def test_joint_allocator_enforces_aggregate_shared_liquidity_and_pod_limits():
    weights = np.array([[0.04, -0.04], [0.04, -0.04]])
    dollars, evidence = allocate_jointly(
        [0.12, 0.11],
        [1e8, 1e8],
        ["a", "b"],
        weights,
        np.zeros(2),
        frame(1e5),
        settings(),
        1e8,
        np.eye(2) * 0.001,
    )
    combined = dollars @ weights
    assert evidence["status"] == "solved"
    assert np.max(abs(combined)) <= 1e5 + 1
    assert dollars.sum() <= 2.5e6 + 1
    assert all(dollars <= 5e7)
    assert evidence["competition_hurdle_dollars"] == 4e6
    assert evidence["expected_surplus_over_TB_plus_four_dollars"] < 0


def test_covariance_penalty_reduces_duplicate_funding_without_redefining_edge():
    s = settings() | {"impact_eta": 0}
    args = (
        [0.01, 0.01],
        [1e8, 1e8],
        ["a", "b"],
        np.array([[0.04, -0.04], [0.04, -0.04]]),
        np.zeros(2),
        frame(1e9),
        s,
        1e8,
    )
    independent, _ = allocate_jointly(*args, np.eye(2) * 0.04, risk_aversion=10)
    duplicate, _ = allocate_jointly(
        *args, np.ones((2, 2)) * 0.04, risk_aversion=10
    )
    assert duplicate.sum() < independent.sum()
    assert independent.sum() < 1e8


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_zero_adv_incumbent_does_not_authorize_new_exposure(side):
    unit = np.array([side, 0.0])
    carried = np.array([side * 100.0, 0.0])
    s = settings() | {
        "max_pod_capital": 1.0,
        "single_name_limit": 0.5,
        "impact_eta": 0,
    }
    bound = capital_bound(unit, carried, frame(0), s, 1000.0)
    assert bound <= 100.0
    amounts, _ = allocate_jointly(
        [0.1], [1000.0], ["a"], [unit], carried, frame(0), s, 1000.0, [[0.0]]
    )
    assert amounts[0] <= 100.0 + 0.01


def test_recurrent_impact_is_charged_even_when_current_order_is_unchanged():
    s = settings() | {"impact_eta": 0}
    weights = np.array([[0.04, -0.04]])
    args = (
        [0.1],
        [1e8],
        ["a"],
        weights,
        weights[0] * 1e7,
        frame(1e9),
        s,
        1e8,
        [[0.001]],
    )
    without, _ = allocate_jointly(*args, recurring_impacts=[0.0])
    with_cost, evidence = allocate_jointly(*args, recurring_impacts=[1e-4])
    assert with_cost[0] < without[0]
    assert evidence["prior_estimated_annual_recurring_impact_dollars"] > 0
    assert evidence["one_time_current_order_impact_dollars"] == 0
