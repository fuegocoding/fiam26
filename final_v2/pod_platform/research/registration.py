"""Registration-before-outcome-access and trial accounting (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 13.1, 13.8).

Registrations, trials, variant budgets and outcome accesses live in the ``Registry`` database, so counts survive
restarts and are shared by every process that opens the same file. Budget reservations are atomic (check and
insert in one write transaction); refused reservations/accesses are kept in ``research_audit``.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, StrictModel, canonical_digest, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from .registry import BUDGETED_TRIAL_KINDS, Registry

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
        # read-through cache of immutable registrations (the registry table is the source of truth)
        self._registrations: dict[str, RegisteredExperiment] = {}
        # process-local overlay kept only for subclasses that bring their own storage (DurableResearchLedger);
        # the base class never writes to these two.
        self._outcome_access: list[dict] = []
        self._trials: list[TrialRecord] = []

    # ------------------------------------------------------------ registration
    def register(self, experiment: RegisteredExperiment) -> str:
        """Durably register; the registration and its trial are stored in one transaction.

        A repeat of an existing id fails even with identical content (use ``ensure_registered`` to recover)."""
        self._registry.research_register(
            experiment.registration_id,
            experiment.spec_content_hash,
            experiment.registered_at.isoformat(),
            experiment.model_dump(mode="json"),
            self._trial_row(
                TrialRecord(
                    kind="registered_specification",
                    spec_hash=experiment.spec_content_hash,
                    detail=f"registered {experiment.registration_id}",
                    counted_at=experiment.registered_at,
                )
            ),
        )
        self._registrations[experiment.registration_id] = experiment
        return experiment.registration_id

    def ensure_registered(self, experiment: RegisteredExperiment) -> tuple[RegisteredExperiment, bool]:
        """Return the stored registration if the id is known (never resetting counts), else register it.

        Known id with a different spec hash is refused. Returns ``(registration, created)``."""
        try:
            known = self.registration(experiment.registration_id)
        except KeyError:
            try:
                self.register(experiment)
                return self.registration(experiment.registration_id), True
            except PodPlatformError:
                # lost a race with a concurrent registrar: fall through to the stored record
                known = self.registration(experiment.registration_id)
        if known.spec_content_hash != experiment.spec_content_hash:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"registration {experiment.registration_id} exists with a different specification hash",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="Amendments create new IDs linked to parents.",
                )
            )
        return known, False

    def registration(self, registration_id: str) -> RegisteredExperiment:
        cached = self._registrations.get(registration_id)
        if cached is not None:
            return cached
        stored = self._registry.research_registration(registration_id)
        if stored is None:
            raise KeyError(registration_id)
        record = RegisteredExperiment.model_validate(stored)
        self._registrations[registration_id] = record
        return record

    # ------------------------------------------------------------------ trials
    @staticmethod
    def _trial_row(trial: TrialRecord) -> dict:
        return {"kind": trial.kind, "spec_hash": trial.spec_hash, "detail": trial.detail, "counted_at": trial.counted_at.isoformat()}

    def record_trial(self, trial: TrialRecord, registration_id: Optional[str] = None) -> None:
        row = self._trial_row(trial)
        self._registry.research_add_trial(row["kind"], row["spec_hash"], row["detail"], row["counted_at"], registration_id)

    def reserve_trial(self, registration_id: str, trial: TrialRecord, *, reservation_key: Optional[str] = None) -> dict:
        """Spend one trial BEFORE the work runs. Atomic against the variant budget.

        The same ``reservation_key`` (default: registration, kind, spec hash and detail) is one logical spend, so
        retrying a cached identical experiment costs nothing, while a new variant detail spends a new slot."""
        registration = self.registration(registration_id)
        key = reservation_key or canonical_digest(
            {"registration": registration_id, "kind": trial.kind, "spec_hash": trial.spec_hash, "detail": trial.detail}
        )
        row = self._trial_row(trial)
        return self._registry.research_reserve(
            registration_id, registration.spec_content_hash, registration.variant_budget,
            row["kind"], row["detail"], row["counted_at"], key,
            extra_used=sum(1 for t in self._trials if t.kind in BUDGETED_TRIAL_KINDS and t.spec_hash == registration.spec_content_hash),
        )

    def trial_count(self, kind: Optional[str] = None) -> int:
        legacy = len(self._trials) if kind is None else sum(1 for trial in self._trials if trial.kind == kind)
        return self._registry.research_trial_count(kind) + legacy

    # ---------------------------------------------------------- outcome access
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
        self._registry.research_record_access(access_id, partition, actor, purpose, record_count, authorized=authorized)
        return access_id

    def outcome_accesses(self) -> list[dict]:
        return self._registry.research_accesses() + list(self._outcome_access)

    # ------------------------------------------------------------------ budget
    def variant_budget_remaining(self, registration_id: str) -> int:
        experiment = self.registration(registration_id)
        used = self._registry.research_budgeted_used(experiment.spec_content_hash) + sum(
            1 for trial in self._trials if trial.kind in BUDGETED_TRIAL_KINDS and trial.spec_hash == experiment.spec_content_hash
        )
        return experiment.variant_budget - used

    def enforce_variant_budget(self, registration_id: str) -> None:
        remaining = self.variant_budget_remaining(registration_id)
        if remaining <= 0:
            self._registry.research_audit("budget_check_failed", registration_id, "variant budget exhausted")
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"variant budget exhausted for {registration_id}",
                    code=ErrorCode.PROVIDER_BUDGET_EXCEEDED,
                    stop_condition="A new registered variant is required; silent retries do not reset the budget.",
                )
            )
