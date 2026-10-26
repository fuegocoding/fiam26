"""Edge must not depend on fund size; capital must respond continuously to costs."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pod_platform.application.edge_research import host_control
from pod_platform.research.edge_capacity import (
    allocate_continuously,
    current_capacity,
    measure_edge,
    optimal_capital,
)
from pod_platform.research.pod_plans import plan_weights
from pod_platform.research.training_tactical import DailyMarket, replay_daily


def _fixture():
    dates = pd.bdate_range("2020-01-02", periods=7)
    frame = pd.DataFrame(
        {
            "me": [20_000.0, 20_000.0],
            "dolvol_126d": [100_000.0, 100_000.0],
            "rvol_21d": [0.02, 0.02],
            "bidaskhl_21d": [0.0, 0.0],
        }
    )
    market = DailyMarket(
        dates,
        np.zeros((7, 2)),
        np.ones((7, 2), dtype=bool),
        np.zeros(7, dtype=int),
        [],
        [frame],
    )
    settings = {
        "cost_bps_one_way": 0,
        "large_cap_borrow_threshold_usd": 10_000_000_000,
        "large_cap_borrow_bps": 0,
        "other_borrow_bps": 0,
        "impact_eta": 0,
        "max_daily_participation": 0.1,
        "order_ttl_sessions": 10,
        "max_borrow_fraction_market_cap": 0.001,
        "max_borrow_adv_days": 5,
        "capacity_execution": True,
        "strategy_capital_usd": 1000,
        "gross_limit": 2,
        "single_name_limit": 0.05,
        "universe_size": 500,
        "execution_no_trade_band": 0,
        "execution_gross_buffer": 1,
    }
    return market, settings


def test_unit_edge_is_invariant_to_capital_and_trading_volume():
    market, settings = _fixture()
    market.returns[2:] = [0.01, -0.01]
    schedule = {1: np.array([0.04, -0.04])}
    original = measure_edge(market, schedule, settings)
    market.frames[0]["dolvol_126d"] *= 0.01
    changed = measure_edge(
        market, schedule, {**settings, "strategy_capital_usd": 1_000_000_000}
    )
    np.testing.assert_array_equal(original["net"], changed["net"])
    assert changed["hard_capacity_usd"][1] < original["hard_capacity_usd"][1]
    assert changed["capital_assumption"] is None
    assert original["net"][1] == 0
    assert original["net"][2] > 0
    expensive = measure_edge(
        market,
        schedule,
        {**settings, "cost_bps_one_way": 100, "large_cap_borrow_bps": 1000},
    )
    np.testing.assert_array_equal(original["gross"], expensive["gross"])
    assert expensive["net"].sum() < original["net"].sum()


def test_rule_weights_are_invariant_to_capital_and_adv():
    rng = np.random.default_rng(77)
    shape = (5, 600)
    frames = [
        pd.DataFrame(
            {
                "me": np.full(600, 20_000.0),
                "gics": ["10"] * 300 + ["20"] * 300,
                "beta_60m": np.ones(600),
                "dolvol_126d": np.full(600, 10_000_000.0),
            }
        )
        for _ in range(5)
    ]
    panel = SimpleNamespace(
        features={"cash_at": rng.normal(size=shape)},
        mask=np.ones(shape, dtype=bool),
        frames=frames,
    )
    plan = host_control("cash", {"cash_at": 1}, 3, "sector_beta")
    settings = {
        "universe_size": 500,
        "gross_limit": 2,
        "single_name_limit": 0.05,
        "strategy_capital_usd": 10_000_000,
    }
    original = plan_weights(plan, panel, settings)
    for frame in frames:
        frame["dolvol_126d"] *= 0.0001
    changed = plan_weights(
        plan, panel, {**settings, "strategy_capital_usd": 900_000_000}
    )
    np.testing.assert_array_equal(original, changed)


def test_continuous_sizing_stops_when_marginal_edge_is_exhausted():
    edge, impact = 0.1, 0.00001
    amount = optimal_capital(edge, impact, 1_000_000_000)
    assert amount == pytest.approx((edge / (1.5 * impact)) ** 2)
    assert edge - 1.5 * impact * np.sqrt(amount) == pytest.approx(0)
    assert amount % 10_000_000 != 0
    assert optimal_capital(edge, impact, amount / 2) == amount / 2
    assert optimal_capital(-0.01, impact, 1_000_000_000) == 0


def test_competing_strategies_share_a_budget_and_can_leave_cash():
    allocated = allocate_continuously(
        [0.1, 0.08], [0.00001, 0.00002], [1e9, 1e9], 1e7
    )
    assert allocated.sum() == pytest.approx(1e7)
    marginal = np.array([0.1, 0.08]) - 1.5 * np.array(
        [0.00001, 0.00002]
    ) * np.sqrt(allocated)
    assert marginal[0] == pytest.approx(marginal[1])
    exhausted = allocate_continuously(
        [0.1, -0.01], [0.00001, 0], [1e9, 1e9], 1e9
    )
    assert exhausted[1] == 0
    assert exhausted.sum() < 1e9


def test_inventory_caps_capital_after_edge_selection():
    market, settings = _fixture()
    target = np.array([0.04, -0.04])
    bound = current_capacity(
        target, np.zeros(2), market.frames[0], settings, 1e9
    )
    assert bound == pytest.approx(2_500_000)
    assert optimal_capital(0.2, 0, bound) == bound
    market.frames[0].loc[1, "me"] = 499
    assert (
        current_capacity(target, np.zeros(2), market.frames[0], settings, 1e9)
        == 0
    )


def test_adaptive_orders_use_previous_nav_and_fixed_decision_dollars():
    market, settings = _fixture()
    observed_nav = []

    def decide(nav, carried):
        observed_nav.append(nav)
        return np.array([0.02, -0.02])

    schedule = {1: np.array([0.04, -0.04]), 2: decide}
    before = replay_daily(market, schedule, settings, keep_holdings=True)
    market.returns[2] = [0.5, -0.5]
    changed = replay_daily(market, schedule, settings, keep_holdings=True)
    assert observed_nav[0] == observed_nav[1]
    first_nav = 1000 * np.prod(1 + before["net"][:3])
    second_nav = 1000 * np.prod(1 + changed["net"][:3])
    np.testing.assert_allclose(
        before["execution_targets"][2] * first_nav,
        changed["execution_targets"][2] * second_nav,
    )
