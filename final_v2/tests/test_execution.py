"""Part 7 acceptance: portfolio and execution invariants (spec 26.5, 18)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from pod_platform.contracts import ErrorCode, PodPlatformError
from pod_platform.contracts.orders import OrderSide
from pod_platform.execution import (
    CostModel,
    CostModelConfig,
    DesiredTrade,
    ExecutionPlanner,
    ExecutionPolicy,
    FillEngine,
    LiquiditySnapshot,
    initialize_book,
)

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)
D = Decimal


def _cost_model() -> CostModel:
    return CostModel(
        CostModelConfig(
            model_id="cost_v1",
            half_spread_by_size_bp={"large": 5.0, "small": 20.0},
            eta=1.0,
            explicit_fee_bp=1.0,
        )
    )


def _policy(**over) -> ExecutionPolicy:
    base = dict(
        policy_id="exec_v1",
        max_daily_participation=0.10,
        min_resize_fraction=0.25,
        no_trade_band=0.005,
        min_economic_edge_bp=20.0,
        order_ttl_days=5,
    )
    base.update(over)
    return ExecutionPolicy(**base)


def _liquidity(security_id: str, adv: str = "50000000", price: str = "50") -> LiquiditySnapshot:
    return LiquiditySnapshot(
        security_id=security_id,
        adv_dollars=D(adv),
        daily_volatility=0.02,
        size_tier="large",
        price=D(price),
    )


def _desire(security_id: str, target: str, current: str = "0", nav: str = "1000000000", edge: str = "5000") -> DesiredTrade:
    return DesiredTrade(
        security_id=security_id,
        side=OrderSide.BUY,
        target_weight=D(target),
        current_weight=D(current),
        nav=D(nav),
        declared_edge_bp=D(edge),
    )


def test_participation_never_exceeded():
    planner = ExecutionPlanner(_policy(), _cost_model())
    liquidity = {"AAA": _liquidity("AAA")}
    orders = planner.plan(
        [_desire("AAA", "0.002", nav="1000000000", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    assert len(orders) == 1
    order = orders[0]
    assert order.daily_fill_dollars <= D("5000000")
    assert order.days_to_complete == 1


def test_large_order_split_into_feasible_daily_fills():
    planner = ExecutionPlanner(_policy(), _cost_model())
    liquidity = {"AAA": _liquidity("AAA", adv="1000000")}
    orders = planner.plan(
        [_desire("AAA", "0.05", nav="1000000000", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    order = orders[0]
    assert order.days_to_complete > 1
    assert order.daily_fill_dollars <= D("100000")
    assert order.days_to_complete * order.daily_fill_dollars >= abs(order.order_dollars)


def test_no_trade_band_skips_small_resizes():
    planner = ExecutionPlanner(_policy(), _cost_model())
    liquidity = {"AAA": _liquidity("AAA")}
    orders = planner.plan(
        [_desire("AAA", "0.0501", current="0.05", nav="1000000000", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    assert orders == []


def test_entry_requires_edge_over_round_trip_cost():
    planner = ExecutionPlanner(_policy(), _cost_model())
    liquidity = {"AAA": _liquidity("AAA")}
    thin_edge = planner.plan(
        [_desire("AAA", "0.05", nav="1000000000", edge="1")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    assert thin_edge == []
    rich_edge = planner.plan(
        [_desire("AAA", "0.05", nav="1000000000", edge="500")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    assert len(rich_edge) == 1


def test_missing_liquidity_rejected():
    planner = ExecutionPlanner(_policy(), _cost_model())
    with pytest.raises(PodPlatformError) as exc:
        planner.plan(
            [_desire("BBB", "0.05", edge="5000")],
            {},
            decision_at=T0,
            target_book_id="book_1",
        )
    assert exc.value.response.error.code == ErrorCode.LIQUIDITY_UNAVAILABLE


def test_partial_fill_remaining_stays_pending():
    planner = ExecutionPlanner(_policy(), _cost_model())
    engine = FillEngine(_cost_model())
    liquidity = {"AAA": _liquidity("AAA", adv="1000000")}
    orders = planner.plan(
        [_desire("AAA", "0.05", nav="1000000000", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    state = engine.open(orders[0])
    engine.fill_day(state, liquidity["AAA"], fill_dollars=D("100000"))
    assert state.remaining_dollars > 0
    assert state.pending is True
    engine.assert_no_assumed_holdings([state])


def test_filled_order_is_not_pending():
    planner = ExecutionPlanner(_policy(), _cost_model())
    engine = FillEngine(_cost_model())
    liquidity = {"AAA": _liquidity("AAA")}
    orders = planner.plan(
        [_desire("AAA", "0.05", nav="1000000000", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    state = engine.open(orders[0])
    engine.fill_day(state, liquidity["AAA"])
    assert state.pending is False
    assert state.filled_dollars == abs(orders[0].order_dollars)


def test_expired_order_cannot_fill():
    planner = ExecutionPlanner(_policy(), _cost_model())
    engine = FillEngine(_cost_model())
    liquidity = {"AAA": _liquidity("AAA", adv="1000000")}
    orders = planner.plan(
        [_desire("AAA", "0.05", nav="1000000000", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    state = engine.open(orders[0])
    engine.fill_day(state, liquidity["AAA"], fill_dollars=D("1000"))
    engine.expire(state, T0)
    assert state.pending is False
    with pytest.raises(ValueError):
        engine.fill_day(state, liquidity["AAA"])


def test_cost_rate_increases_with_participation():
    costs = _cost_model()
    liquidity = _liquidity("AAA", adv="1000000")
    small = costs.cost_rate(liquidity, D("10000"))
    large = costs.cost_rate(liquidity, D("900000"))
    assert large > small


def test_initialization_charges_entry_costs():
    planner = ExecutionPlanner(_policy(), _cost_model())
    engine = FillEngine(_cost_model())
    liquidity = {f"AAA{i}": _liquidity(f"AAA{i}") for i in range(5)}
    desires = [_desire(f"AAA{i}", "0.01", edge="5000") for i in range(5)]
    orders = planner.plan(desires, liquidity, decision_at=T0, target_book_id="book_1")
    result = initialize_book(
        orders,
        liquidity,
        engine,
        starting_nav=D("1000000000"),
        min_positions=5,
        inception_at=T0,
    )
    assert result.entry_costs > 0
    assert result.names_held == 5
    assert result.pending_orders == []


def test_initialization_rejects_sub_minimum_buildup():
    planner = ExecutionPlanner(_policy(), _cost_model())
    engine = FillEngine(_cost_model())
    liquidity = {"AAA0": _liquidity("AAA0")}
    orders = planner.plan(
        [_desire("AAA0", "0.01", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    with pytest.raises(PodPlatformError) as exc:
        initialize_book(
            orders,
            liquidity,
            engine,
            starting_nav=D("1000000000"),
            min_positions=200,
            inception_at=T0,
        )
    assert exc.value.response.error.code == ErrorCode.PORTFOLIO_INFEASIBLE


def test_unfilled_quantity_never_becomes_holding():
    engine = FillEngine(_cost_model())
    planner = ExecutionPlanner(_policy(), _cost_model())
    liquidity = {"AAA": _liquidity("AAA", adv="1000000")}
    orders = planner.plan(
        [_desire("AAA", "0.05", nav="1000000000", edge="5000")],
        liquidity,
        decision_at=T0,
        target_book_id="book_1",
    )
    state = engine.open(orders[0])
    engine.fill_day(state, liquidity["AAA"], fill_dollars=D("1"))
    engine.expire(state, T0 + dt.timedelta(days=6))
    engine.assert_no_assumed_holdings([state])
    assert state.remaining_dollars > 0
    assert state.expired is True
