"""A bounded, as-of feature language for registered research proposals.

Operations consume observations, not generated Python. Lookbacks count distinct
measurement periods (they do not assume a calendar frequency). Missing windows
remain missing. Revisions known after formation are never visible.
"""
from __future__ import annotations

import datetime as dt
import math
import statistics
from bisect import bisect_left, bisect_right
from typing import Literal

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, require_aware


class FeatureObservation(StrictModel):
    observation_id: str = Field(min_length=1)
    security_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    period_end: dt.datetime
    available_at: dt.datetime
    version_known_at: dt.datetime
    value: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def clocks(self):
        for name in ("period_end", "available_at", "version_known_at"):
            require_aware(getattr(self, name), name)
        if self.available_at < self.period_end:
            raise ValueError("observation cannot be available before its measurement period ends")
        return self


class FeatureRecipe(StrictModel):
    feature_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    operation: Literal["latest", "lag", "difference", "growth", "rolling_mean"]
    lookback: int = Field(default=1, ge=1, le=120)
    cross_sectional_rank: bool = False


def materialize(recipe: FeatureRecipe, observations: list[FeatureObservation], *,
                formation_at: dt.datetime, security_ids: list[str]) -> list[dict]:
    cutoff = require_aware(formation_at, "formation_at")
    if len(set(security_ids)) != len(security_ids):
        raise ValueError("universe contains duplicate securities")
    universe = set(security_ids)
    versions: dict[tuple, FeatureObservation] = {}
    identities: dict[str, dict] = {}
    for observation in observations:
        serialized = observation.model_dump(mode="json")
        if observation.observation_id in identities and identities[observation.observation_id] != serialized:
            raise ValueError("observation ID refers to conflicting content")
        identities[observation.observation_id] = serialized
        if (observation.source_id != recipe.source_id or observation.metric != recipe.metric
                or observation.security_id not in universe):
            continue
        if max(observation.available_at, observation.version_known_at, observation.period_end) > cutoff:
            continue
        key = (observation.security_id, observation.period_end)
        previous = versions.get(key)
        if previous is not None and previous.version_known_at == observation.version_known_at:
            if previous.value != observation.value:
                raise ValueError("ambiguous versions at the same knowledge timestamp")
        if previous is None or observation.version_known_at > previous.version_known_at:
            versions[key] = observation
    grouped: dict[str, list[FeatureObservation]] = {}
    for observation in versions.values():
        grouped.setdefault(observation.security_id, []).append(observation)
    result = []
    for security in security_ids:
        history = sorted(grouped.get(security, []), key=lambda o: o.period_end)
        operation = recipe.operation
        required = (1 if operation == "latest" else recipe.lookback
                    if operation == "rolling_mean" else recipe.lookback + 1)
        window = history[-required:]
        value, reason = None, None
        if len(window) < required:
            reason = "insufficient_available_history"
        elif operation == "latest":
            value = window[-1].value
        elif operation == "lag":
            value = window[0].value
        elif operation == "difference":
            value = window[-1].value - window[0].value
        elif operation == "growth":
            if window[0].value == 0:
                reason = "zero_growth_denominator"
            else:
                value = window[-1].value / window[0].value - 1
        else:
            value = statistics.mean(o.value for o in window)
        if value is not None and not math.isfinite(value):
            value, reason = None, "non_finite_result"
        result.append({
            "security_id": security, "feature_id": recipe.feature_id,
            "formation_at": cutoff.isoformat(), "value": value, "missing_reason": reason,
            "observation_ids": [o.observation_id for o in window],
            "latest_evidence_at": max((max(o.available_at, o.version_known_at)
                                       for o in window), default=None),
        })
    if recipe.cross_sectional_rank:
        valid = [row for row in result if row["value"] is not None]
        values = sorted(row["value"] for row in valid)
        for row in valid:
            lo, hi = bisect_left(values, row["value"]), bisect_right(values, row["value"])
            # Average tied percentile, centered at zero; missing values stay missing.
            row["value"] = ((lo + hi + 1) / 2) / len(values) - 0.5
    return result
