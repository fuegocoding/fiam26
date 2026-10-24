"""Security master and dated entity resolution (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 9.5-9.6)."""

from __future__ import annotations

import datetime as dt
import enum
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse


class ResolutionState(str, enum.Enum):
    EXACT_DATED = "EXACT_DATED"
    REVIEWED_DATED = "REVIEWED_DATED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class IdentifierType(str, enum.Enum):
    PERMNO = "permno"
    CIK = "cik"
    TICKER = "ticker"
    SHARE_CLASS = "share_class"
    ISSUER = "issuer"
    BRAND = "brand"


class IdentifierLink(DurableModel):
    ID_FIELD = "link_id"

    link_id: str = Field(min_length=1)
    security_id: str = Field(min_length=1)
    identifier_type: IdentifierType
    identifier_value: str = Field(min_length=1)
    valid_from: dt.datetime
    valid_to: dt.datetime
    first_known_at: dt.datetime
    resolution_state: ResolutionState
    evidence_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_interval(self) -> "IdentifierLink":
        require_aware(self.valid_from, "valid_from")
        require_aware(self.valid_to, "valid_to")
        require_aware(self.first_known_at, "first_known_at")
        if self.valid_to <= self.valid_from:
            raise ValueError("valid_to must lie after valid_from")
        if self.resolution_state in (ResolutionState.EXACT_DATED, ResolutionState.REVIEWED_DATED):
            if not self.evidence_id:
                raise ValueError("a dated resolution requires evidence")
        return self

    def valid_at(self, when: dt.datetime) -> bool:
        when = require_aware(when, "when")
        return self.valid_from <= when < self.valid_to

    def known_by(self, when: dt.datetime) -> bool:
        return self.first_known_at <= require_aware(when, "when")


class MappingProposal(StrictModel):
    source_identifier_type: IdentifierType
    source_identifier_value: str = Field(min_length=1)
    candidate_security_ids: list[str] = Field(min_length=1)
    valid_from: dt.datetime
    valid_to: dt.datetime
    evidence_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: ResolutionState

    @model_validator(mode="after")
    def _check(self) -> "MappingProposal":
        require_aware(self.valid_from, "valid_from")
        require_aware(self.valid_to, "valid_to")
        if self.valid_to <= self.valid_from:
            raise ValueError("valid_to must lie after valid_from")
        return self

    @property
    def admissible_for_formation(self) -> bool:
        return (
            self.review_status in (ResolutionState.EXACT_DATED, ResolutionState.REVIEWED_DATED)
            and len(self.candidate_security_ids) == 1
        )


class SecurityMaster:
    def __init__(self) -> None:
        self._links: dict[str, IdentifierLink] = {}

    def add_link(self, link: IdentifierLink) -> str:
        if link.link_id in self._links:
            raise ValueError(f"duplicate link_id {link.link_id}")
        self._links[link.link_id] = link
        return link.link_id

    def candidates(
        self,
        identifier_type: IdentifierType,
        identifier_value: str,
        at: dt.datetime,
    ) -> list[IdentifierLink]:
        at = require_aware(at, "at")
        return [
            link
            for link in self._links.values()
            if link.identifier_type == identifier_type
            and link.identifier_value == identifier_value
            and link.valid_at(at)
            and link.known_by(at)
        ]

    def resolve(
        self,
        identifier_type: IdentifierType,
        identifier_value: str,
        at: dt.datetime,
    ) -> str:
        links = self.candidates(identifier_type, identifier_value, at)
        usable = [
            link
            for link in links
            if link.resolution_state in (ResolutionState.EXACT_DATED, ResolutionState.REVIEWED_DATED)
        ]
        if not links:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"no dated link for {identifier_type.value}={identifier_value!r} at {at.isoformat()}",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="Unresolved identifiers stay out of formation until mapped.",
                )
            )
        if not usable:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"identifier {identifier_value!r} exists only in ambiguous/unresolved links",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="Ambiguous records stay out of formation until resolved.",
                )
            )
        security_ids = {link.security_id for link in usable}
        if len(security_ids) > 1:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"identifier {identifier_value!r} maps to multiple securities at {at.isoformat()}: {sorted(security_ids)}",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="A one-to-many mapping must be resolved to a single dated share class before use.",
                )
            )
        return usable[0].security_id

    def issuer_of(self, security_id: str, at: dt.datetime) -> Optional[str]:
        at = require_aware(at, "at")
        for link in self._links.values():
            if (
                link.security_id == security_id
                and link.identifier_type == IdentifierType.ISSUER
                and link.valid_at(at)
                and link.known_by(at)
            ):
                return link.identifier_value
        return None

    def security_link_valid(self, security_id: str, at: dt.datetime) -> bool:
        at = require_aware(at, "at")
        return any(
            link.security_id == security_id
            and link.resolution_state in (ResolutionState.EXACT_DATED, ResolutionState.REVIEWED_DATED)
            and link.valid_at(at)
            and link.known_by(at)
            for link in self._links.values()
        )

    def __len__(self) -> int:
        return len(self._links)
