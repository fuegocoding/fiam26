"""Model lifecycle and admission certificates (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 13.10, 14)."""

from __future__ import annotations

import datetime as dt
import enum
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse


class ModelState(str, enum.Enum):
    IDEA = "IDEA"
    REGISTERED = "REGISTERED"
    DATA_READY = "DATA_READY"
    DEVELOPMENT = "DEVELOPMENT"
    CHALLENGED = "CHALLENGED"
    CHECK_ELIGIBLE = "CHECK_ELIGIBLE"
    SHADOW = "SHADOW"
    ADMITTED = "ADMITTED"
    MONITORED = "MONITORED"
    REVISED = "REVISED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"
    REJECTED = "REJECTED"
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"


ALLOWED_TRANSITIONS: dict[ModelState, frozenset[ModelState]] = {
    ModelState.IDEA: frozenset({ModelState.REGISTERED, ModelState.REJECTED}),
    ModelState.REGISTERED: frozenset({ModelState.DATA_READY, ModelState.REJECTED}),
    ModelState.DATA_READY: frozenset({ModelState.DEVELOPMENT, ModelState.REJECTED}),
    ModelState.DEVELOPMENT: frozenset({ModelState.CHALLENGED, ModelState.REJECTED, ModelState.DESCRIPTIVE_ONLY}),
    ModelState.CHALLENGED: frozenset(
        {ModelState.CHECK_ELIGIBLE, ModelState.REJECTED, ModelState.DESCRIPTIVE_ONLY, ModelState.DEVELOPMENT}
    ),
    ModelState.CHECK_ELIGIBLE: frozenset({ModelState.SHADOW, ModelState.REJECTED, ModelState.DESCRIPTIVE_ONLY}),
    ModelState.SHADOW: frozenset({ModelState.ADMITTED, ModelState.REJECTED, ModelState.DESCRIPTIVE_ONLY}),
    ModelState.ADMITTED: frozenset({ModelState.MONITORED, ModelState.SUSPENDED, ModelState.RETIRED}),
    ModelState.MONITORED: frozenset(
        {ModelState.REVISED, ModelState.SUSPENDED, ModelState.RETIRED, ModelState.ADMITTED}
    ),
    ModelState.REVISED: frozenset({ModelState.DEVELOPMENT, ModelState.RETIRED}),
    ModelState.SUSPENDED: frozenset({ModelState.MONITORED, ModelState.RETIRED}),
    ModelState.RETIRED: frozenset(),
    ModelState.REJECTED: frozenset(),
    ModelState.DESCRIPTIVE_ONLY: frozenset({ModelState.RETIRED}),
}


class StateTransition(StrictModel):
    model_id: str = Field(min_length=1)
    from_state: ModelState
    to_state: ModelState
    authority: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    at: dt.datetime

    @model_validator(mode="after")
    def _check(self) -> "StateTransition":
        require_aware(self.at, "at")
        allowed = ALLOWED_TRANSITIONS[self.from_state]
        if self.to_state not in allowed:
            raise ValueError(
                f"illegal transition {self.from_state.value} -> {self.to_state.value}"
            )
        return self


class AdmissionCertificate(DurableModel):
    ID_FIELD = "certificate_id"

    certificate_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_hashes: dict[str, str] = Field(min_length=1)
    data_hashes: dict[str, str] = Field(min_length=1)
    approved_universe: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    horizon_trading_days: int = Field(ge=1)
    effective_start: dt.datetime
    approved_mode: str = Field(min_length=1)
    risk_restrictions: list[str] = Field(default_factory=list)
    capacity_restrictions: list[str] = Field(default_factory=list)
    expiry_policy: str = Field(min_length=1)
    known_limitations: list[str] = Field(default_factory=list)
    promotion_evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "AdmissionCertificate":
        require_aware(self.effective_start, "effective_start")
        return self

    def is_valid_at(self, when: dt.datetime) -> bool:
        return require_aware(when, "when") >= self.effective_start


class PromotionCriteria(StrictModel):
    min_coverage: str = Field(min_length=1)
    improvement_vs_primary_baseline: str = Field(min_length=1)
    direction_and_economic_size: str = Field(min_length=1)
    stability_requirement: str = Field(min_length=1)
    return_translation_requirement: str = Field(min_length=1)
    implementation_requirement: str = Field(min_length=1)
    permitted_statistical_uncertainty: str = Field(min_length=1)
    multiple_testing_treatment: str = Field(min_length=1)


class PromotionOutcome(StrictModel):
    registration_id: str = Field(min_length=1)
    evaluated_at: dt.datetime
    criteria: PromotionCriteria
    coverage_met: bool
    improvement_met: bool
    direction_met: bool
    stability_met: bool
    implementation_met: bool
    uncertainty_met: bool
    multiple_testing_met: bool

    @model_validator(mode="after")
    def _check(self) -> "PromotionOutcome":
        require_aware(self.evaluated_at, "evaluated_at")
        return self

    @property
    def admitted(self) -> bool:
        return all(
            (
                self.coverage_met,
                self.improvement_met,
                self.direction_met,
                self.stability_met,
                self.implementation_met,
                self.uncertainty_met,
                self.multiple_testing_met,
            )
        )

    def certificate_or_raise(self, model_id: str, model_hash: str, **kwargs) -> AdmissionCertificate:
        if not self.admitted:
            missing = [
                name
                for name in (
                    "coverage_met",
                    "improvement_met",
                    "direction_met",
                    "stability_met",
                    "implementation_met",
                    "uncertainty_met",
                    "multiple_testing_met",
                )
                if not getattr(self, name)
            ]
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"promotion denied for {self.registration_id}: {', '.join(missing)}",
                    code=ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                    stop_condition="If promotion conditions are not configured and met, the evaluator cannot issue an admission certificate.",
                )
            )
        return AdmissionCertificate(
            owner="integrity",
            created_at=self.evaluated_at,
            certificate_id=f"cert_{self.registration_id}",
            model_id=model_id,
            model_hash=model_hash,
            **kwargs,
        )


class ModelLifecycle:
    def __init__(self) -> None:
        self._states: dict[str, ModelState] = {}
        self._transitions: list[StateTransition] = []

    def state(self, model_id: str) -> ModelState:
        return self._states.get(model_id, ModelState.IDEA)

    def transition(self, event: StateTransition) -> ModelState:
        current = self.state(event.model_id)
        if event.from_state != current:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"{event.model_id} is in {current.value}, not {event.from_state.value}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="State transitions are events in an append-only ledger; concurrent transitions must be serialized.",
                )
            )
        self._states[event.model_id] = event.to_state
        self._transitions.append(event)
        return event.to_state

    def history(self, model_id: str) -> list[StateTransition]:
        return [t for t in self._transitions if t.model_id == model_id]

    def admitted(self, model_id: str) -> bool:
        return self.state(model_id) in (ModelState.ADMITTED, ModelState.MONITORED)
