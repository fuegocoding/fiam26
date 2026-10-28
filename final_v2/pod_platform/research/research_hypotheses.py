"""Observable, falsifiable prediction declarations for fresh agent research."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator

from ..contracts.base import StrictModel


class PredictionContract(StrictModel):
    """Describe the event and outcome independently of its trading construction.

    The runtime checks feature availability and measures the declared effect.
    Text alone never establishes that an event occurred or a strategy has edge.
    """

    trigger: str = Field(min_length=20, max_length=1000)
    expected_direction: Literal["long", "short", "both"]
    target: Literal["peer_relative_return"]
    comparator: Literal["industry_size_peers"]
    event_features: list[str] = Field(min_length=1, max_length=147)
    economic_rationale: str = Field(min_length=20, max_length=1000)
    falsification: str = Field(min_length=20, max_length=1000)

    @field_validator("trigger", "economic_rationale", "falsification")
    @classmethod
    def substantive_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 20:
            raise ValueError("Prediction declarations need substantive text")
        return value

    @field_validator("event_features")
    @classmethod
    def observable_feature_names(cls, values: list[str]) -> list[str]:
        forbidden = {
            "permno", "eom", "date", "daily_return", "returns", "return",
            "forward_return", "label", "outcome", "target", "model_forecast",
        }
        if len(set(values)) != len(values):
            raise ValueError("Declare each observable event feature once")
        for name in values:
            if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
                raise ValueError("Use literal registered observable feature names")
            if name in forbidden:
                raise ValueError("Prediction triggers require observable inputs, not identifiers or outcomes")
        return sorted(values)
