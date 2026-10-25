"""Source card contract (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.3, 8.4)."""

from __future__ import annotations

import enum
from typing import Optional

from pydantic import Field, model_validator

from .base import DurableModel, StrictModel


class AssessmentStatus(str, enum.Enum):
    DOCUMENTED_ONLY = "documented_only"
    SAMPLE_ONLY = "sample_only"
    UNRESOLVED = "unresolved"
    ESTABLISHED = "established"


class CardStatus(str, enum.Enum):
    CANDIDATE = "CANDIDATE"
    DOCUMENTED = "DOCUMENTED"
    SAMPLED = "SAMPLED"
    ASSESSED = "ASSESSED"
    APPROVED = "APPROVED"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"


class ScreenItem(StrictModel):
    status: AssessmentStatus
    evidence_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_evidence(self) -> "ScreenItem":
        if self.status == AssessmentStatus.ESTABLISHED and not self.evidence_id:
            raise ValueError(
                "an established screen item must reference evidence"
            )
        return self


class Screen(StrictModel):
    history: ScreenItem
    metric_lag: ScreenItem
    vintage: ScreenItem
    coverage: ScreenItem

    @property
    def all_established(self) -> bool:
        return all(
            item.status == AssessmentStatus.ESTABLISHED
            for item in (
                self.history,
                self.metric_lag,
                self.vintage,
                self.coverage,
            )
        )


class SourceCard(DurableModel):
    ID_FIELD = "source_id"

    source_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    status: CardStatus
    access_confirmed: bool = False
    formation_eligible: bool = False
    entity_grain: list[str] = Field(min_length=1)
    screen: Screen
    unresolved_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_formation_eligibility(self) -> "SourceCard":
        if self.formation_eligible and not self.access_confirmed:
            raise ValueError("formation_eligible requires access_confirmed")
        if self.formation_eligible and not self.screen.all_established:
            raise ValueError(
                "formation_eligible requires established history, metric_lag, vintage and coverage screens"
            )
        if self.formation_eligible and self.unresolved_questions:
            raise ValueError("formation_eligible forbids unresolved questions")
        return self

    def broker_rejects_formation(self) -> bool:
        return not self.formation_eligible
