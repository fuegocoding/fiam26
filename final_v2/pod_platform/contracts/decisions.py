"""Decision event contract (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.10, 20.8)."""

from __future__ import annotations

import datetime as dt
import enum
from typing import Optional

from pydantic import Field, model_validator

from .base import DurableModel, require_aware


class ActorKind(str, enum.Enum):
    CODE = "code"
    AGENT_ROLE = "agent_role"
    HUMAN = "human"


class DecisionEvent(DurableModel):
    ID_FIELD = "event_id"

    event_id: str = Field(min_length=1)
    parent_id: str | None = None
    actor_kind: ActorKind
    actor_id: str = Field(min_length=1)
    effective_research_at: dt.datetime
    effective_market_at: dt.datetime
    input_artifact_ids: list[str] = Field(min_length=1)
    input_hashes: dict[str, str] = Field(min_length=1)
    proposed_alternatives: list[str] = Field(default_factory=list)
    chosen_action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    authorizing_rule_or_certificate_id: str = Field(min_length=1)
    model_ref: Optional[str] = None
    prompt_ref: Optional[str] = None
    cache_ref: Optional[str] = None
    resulting_artifact_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "DecisionEvent":
        require_aware(self.effective_research_at, "effective_research_at")
        require_aware(self.effective_market_at, "effective_market_at")
        if self.parent_id is not None and self.parent_id == self.event_id:
            raise ValueError("parent_id cannot reference the event itself")
        if self.actor_kind == ActorKind.AGENT_ROLE:
            missing = [
                name
                for name, value in (
                    ("model_ref", self.model_ref),
                    ("prompt_ref", self.prompt_ref),
                    ("cache_ref", self.cache_ref),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "an agent decision must record model/prompt/cache references: "
                    + ", ".join(missing)
                )
        return self
