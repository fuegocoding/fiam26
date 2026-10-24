"""Hypothesis specification contract (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.4, 11.1)."""

from __future__ import annotations

import enum

from pydantic import Field, model_validator

from .base import DurableModel


class ResearchMode(str, enum.Enum):
    CONTEMPORARY_RESEARCH = "contemporary_research"
    HISTORICAL_REPLAY = "historical_replay"


class HypothesisSpec(DurableModel):
    ID_FIELD = "hypothesis_id"

    hypothesis_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    parent_id: str | None = None
    mechanism_family: str = Field(min_length=1)
    research_mode: ResearchMode
    question: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    reference_model_id: str = Field(min_length=1)
    feature_spec_ids: list[str] = Field(min_length=1)
    applicability_spec_id: str = Field(min_length=1)
    baseline_experiment_ids: list[str] = Field(min_length=1)
    falsification: list[str] = Field(min_length=1)
    variant_budget: int = Field(ge=1)
    experiment_spec_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_no_duplicate_specs(self) -> "HypothesisSpec":
        if len(set(self.feature_spec_ids)) != len(self.feature_spec_ids):
            raise ValueError("feature_spec_ids must be unique")
        if len(set(self.baseline_experiment_ids)) != len(self.baseline_experiment_ids):
            raise ValueError("baseline_experiment_ids must be unique")
        return self

    @model_validator(mode="after")
    def _check_amendment_link(self) -> "HypothesisSpec":
        if self.parent_id is not None and self.parent_id == self.hypothesis_id:
            raise ValueError("parent_id cannot reference the hypothesis itself")
        return self
