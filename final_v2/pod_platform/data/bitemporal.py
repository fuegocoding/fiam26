"""Bitemporal observation store (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 9.1, 9.3).

Every observation carries its economic clocks (event_time, period) and its
knowledge clocks (source_published_at, version_known_at, ingested_at). An as-of
query selects the latest version eligible at the requested cutoff — never the
latest row in the store. First reports and later restatements are separate rows.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, require_aware


class Observation(DurableModel):
    ID_FIELD = "observation_id"

    observation_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    event_time: Optional[dt.datetime] = None
    period_start: Optional[dt.datetime] = None
    period_end: Optional[dt.datetime] = None
    source_published_at: Optional[dt.datetime] = None
    platform_delivered_at: Optional[dt.datetime] = None
    ingested_at: dt.datetime
    version_known_at: dt.datetime
    available_at: dt.datetime
    value: Any = None
    unit: Optional[str] = None
    revision_of: Optional[str] = None

    @model_validator(mode="after")
    def _check_clocks(self) -> "Observation":
        for name in ("event_time", "period_start", "period_end", "source_published_at", "platform_delivered_at"):
            value = getattr(self, name)
            if value is not None:
                require_aware(value, name)
        require_aware(self.ingested_at, "ingested_at")
        require_aware(self.version_known_at, "version_known_at")
        require_aware(self.available_at, "available_at")
        if self.available_at > self.version_known_at:
            raise ValueError("a version cannot be known before the information is available")
        if self.period_start is not None and self.period_end is not None:
            if self.period_end < self.period_start:
                raise ValueError("period_end cannot precede period_start")
        if self.version == 1 and self.revision_of is not None:
            raise ValueError("a first report cannot be a revision_of anything")
        if self.version > 1 and self.revision_of is None:
            raise ValueError("a restatement must reference the observation it revises")
        return self

    @property
    def business_key(self) -> tuple[str, str, str]:
        return (self.source_id, self.metric, self.entity_id)


class BitemporalStore:
    def __init__(self) -> None:
        self._rows: dict[str, Observation] = {}

    def insert(self, observation: Observation) -> str:
        if observation.observation_id in self._rows:
            raise ValueError(f"duplicate observation_id {observation.observation_id}")
        self._rows[observation.observation_id] = observation
        return observation.observation_id

    def get(self, observation_id: str) -> Optional[Observation]:
        return self._rows.get(observation_id)

    def asof_versions(
        self,
        information_cutoff: dt.datetime,
        *,
        source_id: Optional[str] = None,
        metric: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> list[Observation]:
        cutoff = require_aware(information_cutoff, "information_cutoff")
        eligible = [
            row
            for row in self._rows.values()
            if row.available_at <= cutoff and row.version_known_at <= cutoff
        ]
        if source_id is not None:
            eligible = [row for row in eligible if row.source_id == source_id]
        if metric is not None:
            eligible = [row for row in eligible if row.metric == metric]
        if entity_id is not None:
            eligible = [row for row in eligible if row.entity_id == entity_id]
        latest: dict[tuple[str, str, str], Observation] = {}
        for row in eligible:
            key = row.business_key
            incumbent = latest.get(key)
            if incumbent is None or (row.version_known_at, row.version) > (
                incumbent.version_known_at,
                incumbent.version,
            ):
                latest[key] = row
        return sorted(latest.values(), key=lambda r: r.observation_id)

    def eligibility_predicate(self, observation: Observation, information_cutoff: dt.datetime) -> bool:
        cutoff = require_aware(information_cutoff, "information_cutoff")
        return observation.available_at <= cutoff and observation.version_known_at <= cutoff

    def __len__(self) -> int:
        return len(self._rows)
