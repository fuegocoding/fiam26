"""Initialization convention (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 18.8).

The evaluation start establishes actual initial positions through the same
fill and cost rules as any later trade. No cost-free, fully invested target
portfolio and no multi-day sub-minimum buildup reported as compliant.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from ..contracts.base import require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from .planner import FillEngine, LiquiditySnapshot, PlannedOrder

ZERO = Decimal("0")


@dataclass
class InitializationResult:
    filled_dollars: Decimal = ZERO
    entry_costs: Decimal = ZERO
    names_held: int = 0
    pending_orders: list[str] = field(default_factory=list)
    starting_nav: Decimal = ZERO


def initialize_book(
    orders: Sequence[PlannedOrder],
    liquidity: dict[str, LiquiditySnapshot],
    engine: FillEngine,
    *,
    starting_nav: Decimal,
    min_positions: int,
    inception_at: dt.datetime,
    closing_batch: bool = True,
) -> InitializationResult:
    require_aware(inception_at, "inception_at")
    if starting_nav <= ZERO:
        raise ValueError("starting NAV must be positive")
    result = InitializationResult(starting_nav=starting_nav)
    filled_names: set[str] = set()
    for order in orders:
        snapshot = liquidity.get(order.security_id)
        if snapshot is None:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"no liquidity snapshot for {order.security_id} at inception",
                    code=ErrorCode.LIQUIDITY_UNAVAILABLE,
                    stop_condition="Initialization uses the same fill and cost rules as any other trade.",
                )
            )
        state = engine.open(order)
        cost = engine.fill_day(
            state, snapshot, fill_dollars=abs(order.order_dollars)
        )
        result.filled_dollars += state.filled_dollars
        result.entry_costs += cost
        filled_names.add(order.security_id)
        if state.pending:
            result.pending_orders.append(order.order_id)
    result.names_held = len(filled_names)
    if closing_batch and result.names_held < min_positions:
        raise PodPlatformError(
            ToolResponse.failure(
                summary=f"inception batch holds {result.names_held} names below the {min_positions} floor",
                code=ErrorCode.PORTFOLIO_INFEASIBLE,
                stop_condition="A multi-day sub-minimum buildup cannot be reported as compliant.",
            )
        )
    return result
