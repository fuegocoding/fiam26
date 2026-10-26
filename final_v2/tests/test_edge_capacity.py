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


def test_expired_reversal_can_close_but_cannot_open_opposite_position():
    market, settings = _fixture()
    market.frames[0]["dolvol_126d"] = 20.0
    settings.update(
        strategy_capital_usd=100.0,
        single_name_limit=0.5,
        order_ttl_sessions=2,
        execution_no_trade_band=0,
        execution_gross_buffer=1,
    )
    # Initial fill completes in two days. A later opposite target expires while
    # reducing the original book: the old position can finish closing only.
    result = replay_daily(
        market,
        {0: np.array([0.04, -0.04]), 2: np.array([-0.05, 0.05])},
        settings,
        keep_holdings=True,
    )
    assert result["holdings"][3, 0] >= -1e-10
    assert np.min(result["holdings"][4:, 0]) >= -1e-10


def test_native_sleeve_updates_preserve_other_pending_order_expiry_and_replay():
    from pod_platform.research.sleeve_execution import (
        RecordedInstruction,
        SleeveExecutionBook,
    )

    market, settings = _fixture()
    frame = pd.concat([market.frames[0], market.frames[0]], ignore_index=True)
    frame["dolvol_126d"] = 20.0
    market.frames = [frame]
    market.returns = np.zeros((len(market.dates), 4))
    market.tradable = np.ones_like(market.returns, bool)
    settings.update(
        strategy_capital_usd=100.0,
        single_name_limit=0.5,
        order_ttl_sessions=2,
        execution_no_trade_band=0,
        execution_gross_buffer=1,
    )
    book = SleeveExecutionBook(4)
    schedule = {
        0: lambda nav, carried: book.set_targets(
            0, {"monthly": np.array([10.0, -10.0, 0.0, 0.0])}
        ),
        1: lambda nav, carried: None,  # An unfunded daily research review.
        3: lambda nav, carried: book.set_targets(
            3, {"daily": np.array([0.0, 0.0, 2.0, -2.0])}
        ),
    }
    result = replay_daily(
        market,
        schedule,
        {**settings, "execution_observer": book},
        keep_holdings=True,
    )
    assert set(result["execution_targets"]) == {0, 3}
    np.testing.assert_allclose(
        result["holdings"][2:, :2],
        np.tile([0.04, -0.04], (len(market.dates) - 2, 1)),
    )
    replay = replay_daily(
        market,
        {
            day: RecordedInstruction(
                target, result["execution_renewal_masks"][day]
            )
            for day, target in result["execution_targets"].items()
        },
        settings,
        keep_holdings=True,
    )
    np.testing.assert_array_equal(result["net"], replay["net"])
    np.testing.assert_array_equal(result["holdings"], replay["holdings"])
    np.testing.assert_allclose(
        sum(book.positions.values()),
        result["holdings"][-1] * result["execution"]["final_nav_usd"],
        atol=1e-6,
    )


def test_daily_component_returns_reconcile_cash_positions_and_costs():
    market, settings = _fixture()
    market.cash_returns = np.repeat(0.0001, len(market.dates))
    market.returns[2:] = [0.001, -0.002]
    settings.update(cost_bps_one_way=10, large_cap_borrow_bps=50)
    result = replay_daily(market, {1: np.array([0.04, -0.04])}, settings)
    a = result["attribution"]
    np.testing.assert_allclose(
        result["net"],
        a["long_return"]
        + a["short_return"]
        + a["cash_return"]
        - a["trading_cost_return"]
        - a["borrow_cost_return"],
        atol=1e-15,
    )


@pytest.mark.parametrize("gross_buffer", [1.0, 0.98])
def test_shared_symbol_review_cannot_revive_an_expired_other_sleeve_order(
    gross_buffer,
):
    from pod_platform.research.sleeve_execution import (
        RecordedInstruction,
        SleeveExecutionBook,
    )

    market, settings = _fixture()
    market.frames[0]["dolvol_126d"] = 20.0
    settings.update(
        strategy_capital_usd=100.0,
        single_name_limit=0.5,
        order_ttl_sessions=2,
        execution_no_trade_band=0,
        execution_gross_buffer=gross_buffer,
    )
    book = SleeveExecutionBook(
        2, gross_buffer=gross_buffer, order_ttl_sessions=2
    )
    schedule = {
        0: lambda nav, carried: book.set_targets(
            0, {"monthly": np.array([10.0, -10.0])}
        ),
        3: lambda nav, carried: book.set_targets(
            3, {"daily": np.array([1.0, -1.0])}
        ),
    }
    result = replay_daily(
        market,
        schedule,
        {**settings, "execution_observer": book},
        keep_holdings=True,
    )
    # The monthly sleeve fills $2 per day for two sessions, then expires. A
    # new sleeve's order for the same symbols must add only its own dollars.
    np.testing.assert_allclose(book.positions["monthly"], [4.0, -4.0])
    np.testing.assert_allclose(
        book.positions["daily"], np.array([1.0, -1.0]) * gross_buffer
    )
    np.testing.assert_allclose(result["holdings"][2], [0.04, -0.04])
    expected = np.array([4.0 + gross_buffer, -4.0 - gross_buffer]) / 100.0
    np.testing.assert_allclose(
        result["holdings"][3:],
        np.tile(expected, (len(market.dates) - 3, 1)),
        atol=1e-8,
    )
    replay = replay_daily(
        market,
        {
            day: RecordedInstruction(
                target, result["execution_renewal_masks"][day]
            )
            for day, target in result["execution_targets"].items()
        },
        settings,
        keep_holdings=True,
    )
    np.testing.assert_array_equal(result["net"], replay["net"])
    np.testing.assert_array_equal(result["holdings"], replay["holdings"])


@pytest.mark.parametrize(
    ("desired", "retained"), [(5.0, 5.0), (-5.0, 0.0), (20.0, 10.0)]
)
def test_expired_sleeve_target_preserves_reductions_and_cancels_new_risk(
    desired, retained
):
    from pod_platform.research.sleeve_execution import SleeveExecutionBook

    book = SleeveExecutionBook(1, gross_buffer=0.98, order_ttl_sessions=2)
    book.set_targets(0, {"monthly": np.array([desired / 0.98])})
    book.positions["monthly"][:] = 10.0
    book.expire(2)
    np.testing.assert_allclose(book.targets["monthly"] * 0.98, [retained])
