"""Cost model and fixed-rule execution planner
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 18.1-18.3, 18.7-18.8).

Deterministic baseline: no-trade region around incumbents, minimum
economically worthwhile resize, daily participation bounds, order expiry,
and participation-split fills. Orders exceeding participation limits are
split into feasible daily fills; an order never assumes unlimited size
behind a capped impact rate.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from typing import Optional, Sequence

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from ..contracts.orders import Order, OrderSide

ZERO = Decimal("0")


class CostModelConfig(StrictModel):
    model_id: str = Field(min_length=1)
    half_spread_by_size_bp: dict[str, float] = Field(min_length=1)
    eta: float = Field(gt=0)
    explicit_fee_bp: float = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> "CostModelConfig":
        for tier, bp in self.half_spread_by_size_bp.items():
            if bp < 0:
                raise ValueError(f"half spread for {tier!r} cannot be negative")
        return self


class ExecutionPolicy(StrictModel):
    policy_id: str = Field(min_length=1)
    max_daily_participation: float = Field(gt=0, le=1)
    min_resize_fraction: float = Field(ge=0, le=1)
    no_trade_band: float = Field(ge=0, le=1)
    min_economic_edge_bp: float = Field(ge=0)
    order_ttl_days: int = Field(ge=1)


@dataclass(frozen=True)
class LiquiditySnapshot:
    security_id: str
    adv_dollars: Decimal
    daily_volatility: float
    size_tier: str
    price: Decimal

    def __post_init__(self) -> None:
        if self.adv_dollars <= ZERO:
            raise ValueError("ADV must be positive")
        if self.daily_volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.price <= ZERO:
            raise ValueError("price must be positive")


@dataclass(frozen=True)
class DesiredTrade:
    security_id: str
    side: OrderSide
    target_weight: Decimal
    current_weight: Decimal
    nav: Decimal
    declared_edge_bp: Decimal = ZERO

    def order_dollars(self) -> Decimal:
        return (self.target_weight - self.current_weight) * self.nav

    def is_entry(self) -> bool:
        return self.current_weight == ZERO


@dataclass
class PlannedOrder:
    order_id: str
    security_id: str
    side: OrderSide
    order_dollars: Decimal
    quantity: Decimal
    expected_cost: Decimal
    daily_fill_dollars: Decimal
    days_to_complete: int
    priority: int


class CostModel:
    def __init__(self, config: CostModelConfig):
        self._config = config

    @property
    def model_id(self) -> str:
        return self._config.model_id

    def cost_rate(self, liquidity: LiquiditySnapshot, order_dollars: Decimal) -> Decimal:
        if order_dollars < ZERO:
            raise ValueError("order dollars must be non-negative")
        half_spread = Decimal(str(self._config.half_spread_by_size_bp.get(liquidity.size_tier, 0.0)))
        participation = float(order_dollars / liquidity.adv_dollars)
        impact = self._config.eta * liquidity.daily_volatility * math.sqrt(participation)
        total_bp = float(half_spread) + impact * 10000.0 + self._config.explicit_fee_bp
        return Decimal(str(total_bp)) / Decimal("10000")

    def expected_cost(self, liquidity: LiquiditySnapshot, order_dollars: Decimal) -> Decimal:
        return abs(order_dollars) * self.cost_rate(liquidity, order_dollars)


class ExecutionPlanner:
    def __init__(self, policy: ExecutionPolicy, cost_model: CostModel):
        self._policy = policy
        self._costs = cost_model

    def plan(
        self,
        desires: Sequence[DesiredTrade],
        liquidity: dict[str, LiquiditySnapshot],
        *,
        decision_at: dt.datetime,
        target_book_id: str,
    ) -> list[PlannedOrder]:
        require_aware(decision_at, "decision_at")
        planned: list[PlannedOrder] = []
        for index, desire in enumerate(desires):
            snapshot = liquidity.get(desire.security_id)
            if snapshot is None:
                raise PodPlatformError(
                    ToolResponse.failure(
                        summary=f"no liquidity snapshot for {desire.security_id}",
                        code=ErrorCode.LIQUIDITY_UNAVAILABLE,
                        stop_condition="No order may be planned without dated liquidity evidence.",
                    )
                )
            delta = desire.order_dollars()
            if delta == ZERO:
                continue
            abs_delta = abs(delta)
            weight_delta = abs(desire.target_weight - desire.current_weight)
            if (
                not desire.is_entry()
                and weight_delta <= Decimal(str(self._policy.no_trade_band))
            ):
                continue
            expected = self._costs.expected_cost(snapshot, abs_delta)
            if desire.is_entry():
                round_trip_bp = self._costs.cost_rate(snapshot, abs_delta) * Decimal("20000")
                if desire.declared_edge_bp <= round_trip_bp:
                    continue
            max_daily = snapshot.adv_dollars * Decimal(str(self._policy.max_daily_participation))
            if max_daily <= ZERO:
                raise PodPlatformError(
                    ToolResponse.failure(
                        summary=f"zero daily capacity for {desire.security_id}",
                        code=ErrorCode.LIQUIDITY_UNAVAILABLE,
                        stop_condition="Feasible participation must be positive.",
                    )
                )
            days = max(1, int((abs_delta / max_daily).to_integral_value(rounding=ROUND_CEILING)))
            planned.append(
                PlannedOrder(
                    order_id=f"{target_book_id}_{desire.security_id}_{index}",
                    security_id=desire.security_id,
                    side=desire.side,
                    order_dollars=delta,
                    quantity=abs_delta / snapshot.price,
                    expected_cost=expected,
                    daily_fill_dollars=abs_delta / Decimal(days),
                    days_to_complete=days,
                    priority=0 if delta * desire.target_weight > ZERO else 1,
                )
            )
        planned.sort(key=lambda o: (o.priority, -abs(o.order_dollars)))
        return planned


@dataclass
class FillState:
    order_id: str
    security_id: str
    side: OrderSide
    remaining_dollars: Decimal
    filled_dollars: Decimal = ZERO
    costs_paid: Decimal = ZERO
    expired: bool = False

    @property
    def pending(self) -> bool:
        return self.remaining_dollars > ZERO and not self.expired


class FillEngine:
    def __init__(self, cost_model: CostModel):
        self._costs = cost_model

    def open(self, order: PlannedOrder) -> FillState:
        return FillState(
            order_id=order.order_id,
            security_id=order.security_id,
            side=order.side,
            remaining_dollars=abs(order.order_dollars),
        )

    def fill_day(
        self,
        state: FillState,
        liquidity: LiquiditySnapshot,
        *,
        fill_dollars: Optional[Decimal] = None,
    ) -> Decimal:
        if state.expired:
            raise ValueError("cannot fill an expired order")
        if state.remaining_dollars <= ZERO:
            return ZERO
        attempt = fill_dollars if fill_dollars is not None else state.remaining_dollars
        if attempt < ZERO:
            raise ValueError("fill dollars cannot be negative")
        filled = min(attempt, state.remaining_dollars)
        cost = self._costs.expected_cost(liquidity, filled)
        state.filled_dollars += filled
        state.costs_paid += cost
        state.remaining_dollars -= filled
        return cost

    def expire(self, state: FillState, at: dt.datetime) -> None:
        require_aware(at, "at")
        if state.remaining_dollars > ZERO:
            state.expired = True

    def assert_no_assumed_holdings(self, states: Sequence[FillState]) -> None:
        for state in states:
            if state.remaining_dollars < ZERO:
                raise ValueError("unfilled quantity never becomes an assumed holding")
