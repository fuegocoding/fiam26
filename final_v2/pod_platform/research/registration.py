"""Registration-before-outcome-access and trial accounting (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 13.1, 13.8)."""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from .registry import Registry

TRIAL_KINDS = (
    "hypothesis",
    "registered_specification",
    "scored_variant",
    "hyperparameter_config",
    "seed_restart",
    "fold_evaluation",
    "check_window_access",
    "final_evaluation_access",
    "failed_infrastructure_attempt",
)


class RegisteredExperiment(DurableModel):
    ID_FIELD = "registration_id"

    registration_id: str = Field(min_length=1)
    experiment_spec_id: str = Field(min_length=1)
    spec_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_metric: str = Field(min_length=1)
    baseline_experiment_ids: list[str] = Field(min_length=1)
    promotion_conditions: str = Field(min_length=1)
    failure_conditions: str = Field(min_length=1)
    variant_budget: int = Field(ge=1)
    registered_at: dt.datetime

    @model_validator(mode="after")
    def _check(self) -> "RegisteredExperiment":
        require_aware(self.registered_at, "registered_at")
        return self


class TrialRecord(StrictModel):
    kind: str
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    detail: str = Field(min_length=1)
    counted_at: dt.datetime

    @model_validator(mode="after")
    def _check(self) -> "TrialRecord":
        if self.kind not in TRIAL_KINDS:
            raise ValueError(f"unknown trial kind {self.kind!r}")
        require_aware(self.counted_at, "counted_at")
        return self


class ResearchLedger:
    def __init__(self, registry: Registry):
        self._registry = registry
        self._registrations: dict[str, RegisteredExperiment] = {}
        self._outcome_access: list[dict] = []
        self._trials: list[TrialRecord] = []

    def register(self, experiment: RegisteredExperiment) -> str:
        if experiment.registration_id in self._registrations:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"registration {experiment.registration_id} already exists and is immutable",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="Amendments create new IDs linked to parents.",
                )
            )
        self._registrations[experiment.registration_id] = experiment
        self.record_trial(
            TrialRecord(
                kind="registered_specification",
                spec_hash=experiment.spec_content_hash,
                detail=f"registered {experiment.registration_id}",
                counted_at=experiment.registered_at,
            )
        )
        return experiment.registration_id

    def registration(self, registration_id: str) -> RegisteredExperiment:
        return self._registrations[registration_id]

    def record_trial(self, trial: TrialRecord) -> None:
        self._trials.append(trial)
        self._registry.count_trial(f"trial_{len(self._trials)}", trial.kind, trial.spec_hash)

    def trial_count(self, kind: Optional[str] = None) -> int:
        if kind is None:
            return len(self._trials)
        return sum(1 for trial in self._trials if trial.kind == kind)

    def record_outcome_access(
        self,
        access_id: str,
        partition: str,
        actor: str,
        purpose: str,
        record_count: int,
        *,
        authorized: bool = False,
    ) -> str:
        self._registry.record_evaluation_access(
            access_id, partition, actor, purpose, record_count, authorized=authorized
        )
        self._outcome_access.append(
            {
                "access_id": access_id,
                "partition": partition,
                "actor": actor,
                "purpose": purpose,
                "record_count": record_count,
            }
        )
        return access_id

    def outcome_accesses(self) -> list[dict]:
        return list(self._outcome_access)

    def variant_budget_remaining(self, registration_id: str) -> int:
        experiment = self._registrations[registration_id]
        used = sum(
            1
            for trial in self._trials
            if trial.kind in ("scored_variant", "hyperparameter_config", "seed_restart")
            and trial.spec_hash == experiment.spec_content_hash
        )
        return experiment.variant_budget - used

    def enforce_variant_budget(self, registration_id: str) -> None:
        remaining = self.variant_budget_remaining(registration_id)
        if remaining <= 0:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"variant budget exhausted for {registration_id}",
                    code=ErrorCode.PROVIDER_BUDGET_EXCEEDED,
                    stop_condition="A new registered variant is required; silent retries do not reset the budget.",
                )
            )
