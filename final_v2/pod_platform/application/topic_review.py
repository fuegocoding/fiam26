"""A research-stopping judgment kept separate from strategy generation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..contracts.base import StrictModel


class FrontierReview(StrictModel):
    action: Literal["continue", "retain"]
    diagnosis: str = Field(min_length=20, max_length=4000)
    evidence_strategy_ids: list[str] = Field(min_length=1, max_length=6)
    hypothesis: str | None = Field(default=None, min_length=20, max_length=1000)
    proposed_test: str | None = Field(default=None, min_length=20, max_length=1000)
    falsification: str | None = Field(default=None, min_length=20, max_length=1000)
    distinction_from_prior_tests: str | None = Field(default=None, min_length=20, max_length=1000)

    @field_validator("diagnosis", "hypothesis", "proposed_test", "falsification",
        "distinction_from_prior_tests", mode="before")
    @classmethod
    def substantive_text(cls, value):
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def explicit_test_or_retention(self):
        lead = (self.hypothesis, self.proposed_test, self.falsification,
            self.distinction_from_prior_tests)
        if len(set(self.evidence_strategy_ids)) != len(self.evidence_strategy_ids):
            raise ValueError("Reference each measured strategy once")
        if self.action == "continue" and not all(value and value.strip() for value in lead):
            raise ValueError("Further research requires a distinct hypothesis, test and falsification")
        if self.action == "retain" and any(value is not None for value in lead):
            raise ValueError("Retaining this topic cannot also authorize another test")
        return self
