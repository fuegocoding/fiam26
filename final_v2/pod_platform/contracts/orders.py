"""Order and fill contracts (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.9, 18.7)."""

from __future__ import annotations

import datetime as dt
import enum
from decimal import Decimal
from typing import Optional

from pydantic import Field, model_validator

from .base import DurableModel, StrictModel, finite_decimal, require_aware


class OrderSide(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"


class Order(DurableModel):
    ID_FIELD = "order_id"

    order_id: str = Field(min_length=1)
    target_book_id: str = Field(min_length=1)
    security_id: str = Field(min_length=1)
    side: OrderSide
    quantity_intent: Optional[Decimal] = None
    dollar_intent: Optional[Decimal] = None
    conversion_snapshot_id: str = Field(min_length=1)
    allowed_interval_start: dt.datetime
    allowed_interval_end: dt.datetime
    participation_bound: Decimal
    risk_bounds: dict[str, str] = Field(default_factory=dict)
    expiry_at: dt.datetime
    cancellation_condition: str = Field(min_length=1)
    expected_cost: Decimal
    forecast_provenance_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "Order":
        for name in ("quantity_intent", "dollar_intent"):
            value = getattr(self, name)
            if value is not None:
                finite_decimal(value, name)
        if (self.quantity_intent is None) == (self.dollar_intent is None):
            raise ValueError(
                "an order carries exactly one of quantity_intent or dollar_intent"
            )
        if self.quantity_intent is not None and self.quantity_intent == 0:
            raise ValueError("order quantity intent cannot be zero")
        if self.dollar_intent is not None and self.dollar_intent == 0:
            raise ValueError("order dollar intent cannot be zero")
        finite_decimal(self.participation_bound, "participation_bound")
        finite_decimal(self.expected_cost, "expected_cost")
        if not (Decimal("0") < self.participation_bound <= Decimal("1")):
            raise ValueError("participation_bound must be in (0, 1]")
        require_aware(self.allowed_interval_start, "allowed_interval_start")
        require_aware(self.allowed_interval_end, "allowed_interval_end")
        require_aware(self.expiry_at, "expiry_at")
        if self.allowed_interval_end <= self.allowed_interval_start:
            raise ValueError("allowed execution interval must be non-empty")
        if self.expiry_at > self.allowed_interval_end:
            raise ValueError(
                "expiry cannot lie beyond the allowed execution interval"
            )
        return self


class Fill(DurableModel):
    ID_FIELD = "fill_id"

    fill_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    is_simulated: bool
    executed_at: dt.datetime
    quantity: Decimal
    price: Decimal
    currency: str = Field(min_length=1)
    cost_breakdown: dict[str, Decimal] = Field(default_factory=dict)
    liquidity_evidence_id: Optional[str] = None
    accounting_event_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "Fill":
        require_aware(self.executed_at, "executed_at")
        finite_decimal(self.quantity, "quantity")
        finite_decimal(self.price, "price")
        if self.quantity == 0:
            raise ValueError("a fill cannot have zero quantity")
        if self.price <= 0:
            raise ValueError("a fill price must be positive")
        if self.currency.upper() != self.currency or len(self.currency) != 3:
            raise ValueError("currency must be a 3-letter ISO code")
        for key, value in self.cost_breakdown.items():
            finite_decimal(value, f"cost_breakdown[{key}]")
            if value < 0:
                raise ValueError("cost components cannot be negative")
        return self


class PendingExposure(StrictModel):
    order_id: str = Field(min_length=1)
    unfilled_quantity: Decimal

    @model_validator(mode="after")
    def _check(self) -> "PendingExposure":
        finite_decimal(self.unfilled_quantity, "unfilled_quantity")
        return self
