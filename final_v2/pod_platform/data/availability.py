"""Availability rules: evidence-based metric eligibility (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 9.2, 9.4)."""

from __future__ import annotations

import datetime as dt
import enum
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, require_aware


class RuleKind(str, enum.Enum):
    SOURCE_PUBLICATION = "source_publication"
    PERIOD_END_PLUS_LAG = "period_end_plus_lag"
    PLATFORM_DELIVERY = "platform_delivery"


class UsageMode(str, enum.Enum):
    SOURCE_PUBLICATION_HISTORICAL = "source_publication_historical"
    PLATFORM_DELIVERY_HISTORICAL = "platform_delivery_historical"
    DESCRIPTIVE_ONLY = "descriptive_only"


class AvailabilityRule(DurableModel):
    ID_FIELD = "rule_id"

    rule_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    rule_kind: RuleKind
    lag_days: int = Field(ge=0)
    evidence_id: str = Field(min_length=1)
    revision_completeness: str = Field(min_length=1)
    usage_mode: UsageMode
    earliest_start: Optional[dt.datetime] = None

    @model_validator(mode="after")
    def _check(self) -> "AvailabilityRule":
        if self.earliest_start is not None:
            require_aware(self.earliest_start, "earliest_start")
        if (
            self.rule_kind == RuleKind.PERIOD_END_PLUS_LAG
            and self.lag_days == 0
        ):
            raise ValueError("period_end_plus_lag requires a positive lag")
        return self


def compute_available_at(
    rule: AvailabilityRule,
    *,
    source_published_at: Optional[dt.datetime] = None,
    platform_delivered_at: Optional[dt.datetime] = None,
    period_end: Optional[dt.datetime] = None,
) -> dt.datetime:
    if rule.rule_kind == RuleKind.SOURCE_PUBLICATION:
        if source_published_at is None:
            raise ValueError(
                "source_publication rule requires source_published_at"
            )
        return require_aware(source_published_at, "source_published_at")
    if rule.rule_kind == RuleKind.PERIOD_END_PLUS_LAG:
        if period_end is None:
            raise ValueError("period_end_plus_lag rule requires period_end")
        period_end = require_aware(period_end, "period_end")
        return period_end + dt.timedelta(days=rule.lag_days)
    if platform_delivered_at is None:
        raise ValueError(
            "platform_delivery rule requires platform_delivered_at"
        )
    return require_aware(platform_delivered_at, "platform_delivered_at")
