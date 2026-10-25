"""Deterministic accounting ledger (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 19.1, 19.4).

Self-financing convention: short sale proceeds are credited to cash at the
convention price, the short liability is the negative marked quantity, and the
remaining cash balance earns the configured cash return. Borrow fees accrue on
actual borrowed notional and reduce cash. NAV = cash + sum(quantity * mark).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from ..contracts.base import require_aware
from ..contracts.orders import OrderSide

ZERO = Decimal("0")


@dataclass(frozen=True)
class BorrowSchedule:
    schedule_id: str
    large_cap_threshold_usd: Decimal
    large_cap_annual_rate: Decimal
    other_annual_rate: Decimal
    day_count: str = "trading_252"

    def annual_rate(self, market_cap_usd: Decimal) -> Decimal:
        return (
            self.large_cap_annual_rate
            if market_cap_usd >= self.large_cap_threshold_usd
            else self.other_annual_rate
        )


@dataclass
class Position:
    security_id: str
    quantity: Decimal = ZERO
    last_mark: Optional[Decimal] = None
    last_mark_at: Optional[dt.datetime] = None
    borrowed_notional: Decimal = ZERO
    borrow_accrued: Decimal = ZERO

    def market_value(self) -> Decimal:
        if self.last_mark is None:
            raise ValueError(f"{self.security_id} has no mark price")
        return self.quantity * self.last_mark

    def short_notional(self) -> Decimal:
        if self.quantity >= ZERO:
            return ZERO
        if self.last_mark is None:
            raise ValueError(
                f"{self.security_id} is short with no mark; the liability cannot be deleted"
            )
        return -self.quantity * self.last_mark


@dataclass
class AccountingEvent:
    event_id: str
    kind: str
    effective_at: dt.datetime
    detail: dict = field(default_factory=dict)


class AccountingLedger:
    def __init__(self, starting_cash: Decimal, currency: str = "USD"):
        if starting_cash < ZERO:
            raise ValueError("starting cash cannot be negative")
        self.currency = currency
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.events: list[AccountingEvent] = []
        self.trading_costs = ZERO
        self.borrow_costs = ZERO
        self.funding_costs = ZERO
        self.cash_income = ZERO

    def position(self, security_id: str) -> Position:
        return self.positions.setdefault(
            security_id, Position(security_id=security_id)
        )

    def execute(
        self,
        event_id: str,
        security_id: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        fees: Decimal,
        executed_at: dt.datetime,
    ) -> None:
        executed_at = require_aware(executed_at, "executed_at")
        if quantity <= ZERO:
            raise ValueError("executed quantity must be positive")
        if price <= ZERO:
            raise ValueError("execution price must be positive")
        if fees < ZERO:
            raise ValueError("fees cannot be negative")
        position = self.position(security_id)
        if side in (OrderSide.BUY, OrderSide.COVER):
            signed = quantity
        else:
            signed = -quantity
        proceeds = signed * price
        if side == OrderSide.SHORT:
            position.borrowed_notional += quantity * price
        if side == OrderSide.COVER:
            position.borrowed_notional = max(
                ZERO, position.borrowed_notional - quantity * price
            )
        position.quantity += signed
        self.cash -= proceeds
        self.cash -= fees
        self.trading_costs += fees
        if position.quantity == ZERO:
            position.borrowed_notional = ZERO
        self.events.append(
            AccountingEvent(
                event_id=event_id,
                kind=f"execute_{side.value.lower()}",
                effective_at=executed_at,
                detail={
                    "security_id": security_id,
                    "quantity": str(quantity),
                    "price": str(price),
                    "fees": str(fees),
                },
            )
        )

    def mark(self, security_id: str, price: Decimal, at: dt.datetime) -> None:
        at = require_aware(at, "at")
        if price <= ZERO:
            raise ValueError("mark price must be positive")
        position = self.position(security_id)
        position.last_mark = price
        position.last_mark_at = at

    def apply_split(
        self,
        event_id: str,
        security_id: str,
        numerator: int,
        denominator: int,
        at: dt.datetime,
    ) -> None:
        at = require_aware(at, "at")
        if numerator <= 0 or denominator <= 0:
            raise ValueError("split ratio must be positive")
        position = self.position(security_id)
        if position.quantity == ZERO:
            raise ValueError("cannot split a zero position")
        before = position.market_value()
        ratio = Decimal(numerator) / Decimal(denominator)
        position.quantity *= ratio
        if position.last_mark is not None:
            position.last_mark /= ratio
        after = position.market_value()
        if before != after:
            raise ValueError("corporate action changed economic value")
        self.events.append(
            AccountingEvent(
                event_id=event_id,
                kind="split",
                effective_at=at,
                detail={
                    "security_id": security_id,
                    "ratio": f"{numerator}/{denominator}",
                },
            )
        )

    def accrue_cash(
        self, annual_rate: Decimal, periods_per_year: int, at: dt.datetime
    ) -> None:
        at = require_aware(at, "at")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        income = self.cash * annual_rate / Decimal(periods_per_year)
        self.cash += income
        self.cash_income += income
        self.events.append(
            AccountingEvent(
                event_id=f"cash_{at.date().isoformat()}",
                kind="cash_income",
                effective_at=at,
                detail={"income": str(income)},
            )
        )

    def accrue_borrow(
        self,
        schedule: BorrowSchedule,
        market_caps: dict[str, Decimal],
        days: int,
        at: dt.datetime,
    ) -> Decimal:
        at = require_aware(at, "at")
        if days < 0:
            raise ValueError("days cannot be negative")
        divisor = Decimal(252 if schedule.day_count == "trading_252" else 365)
        total = ZERO
        for security_id, position in self.positions.items():
            notional = position.short_notional()
            if notional == ZERO:
                continue
            cap = market_caps.get(security_id, ZERO)
            rate = schedule.annual_rate(cap)
            fee = notional * rate * Decimal(days) / divisor
            position.borrow_accrued += fee
            total += fee
        self.cash -= total
        self.borrow_costs += total
        self.events.append(
            AccountingEvent(
                event_id=f"borrow_{at.date().isoformat()}",
                kind="borrow_accrual",
                effective_at=at,
                detail={"amount": str(total), "days": str(days)},
            )
        )
        return total

    def charge_funding(self, amount: Decimal, at: dt.datetime) -> None:
        if amount < ZERO:
            raise ValueError("funding cost cannot be negative")
        self.cash -= amount
        self.funding_costs += amount
        self.events.append(
            AccountingEvent(
                event_id=f"funding_{require_aware(at, 'at').date().isoformat()}",
                kind="funding_cost",
                effective_at=require_aware(at, "at"),
                detail={"amount": str(amount)},
            )
        )

    def marked_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.quantity != ZERO]

    def gross_exposure(self) -> Decimal:
        return sum(
            (abs(p.market_value()) for p in self.marked_positions()), ZERO
        )

    def net_exposure(self) -> Decimal:
        return sum((p.market_value() for p in self.marked_positions()), ZERO)

    def short_liabilities(self) -> Decimal:
        return sum((p.short_notional() for p in self.marked_positions()), ZERO)

    def nav(self) -> Decimal:
        marked_assets = sum(
            (
                p.market_value()
                for p in self.marked_positions()
                if p.quantity > ZERO
            ),
            ZERO,
        )
        return marked_assets + self.cash - self.short_liabilities()

    def assert_nav_identity(self) -> None:
        recomputed = self.cash + sum(
            (p.market_value() for p in self.marked_positions()), ZERO
        )
        if recomputed != self.nav():
            raise ValueError(
                "NAV identity failed: cash + marked quantities != marked assets + cash - liabilities"
            )

    def positions_table(self) -> list[dict]:
        rows = []
        for position in sorted(
            self.marked_positions(), key=lambda p: p.security_id
        ):
            rows.append(
                {
                    "security_id": position.security_id,
                    "quantity": str(position.quantity),
                    "mark": str(position.last_mark),
                    "market_value": str(position.market_value()),
                    "borrow_accrued": str(position.borrow_accrued),
                }
            )
        return rows
