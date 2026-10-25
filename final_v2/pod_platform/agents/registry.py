"""Language-model registry with temporal eligibility (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 20.12-20.14, 20.19).

Policy A (release-gated, decision C5): a checkpoint may only serve at a
simulated date after its verified publication. Cutoff-qualified backcast
(Policy B) is disabled by default and requires an explicit eligibility
certificate per checkpoint. Distinct evidence fields (20.12) are kept
separate; a base cutoff never approves a later chat/distilled derivative.
"""

from __future__ import annotations

import datetime as dt
import enum
from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import DurableModel, StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse


class TemporalPolicy(str, enum.Enum):
    RELEASE_GATED = "release_gated"
    CUTOFF_QUALIFIED_BACKCAST = "cutoff_qualified_backcast"


class CutoffEvidence(StrictModel):
    pretraining_corpus_boundary: Optional[str] = None
    post_training_boundary: Optional[str] = None
    teacher_provenance: Optional[str] = None
    claimed_knowledge_cutoff: Optional[str] = None
    checkpoint_released_at: dt.datetime
    retrieval_boundary: Optional[str] = None
    local_adaptation_boundary: Optional[str] = None
    evidence_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "CutoffEvidence":
        require_aware(self.checkpoint_released_at, "checkpoint_released_at")
        return self


class ModelCheckpoint(DurableModel):
    ID_FIELD = "checkpoint_id"

    checkpoint_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    exact_model_id: str = Field(min_length=1)
    checkpoint_revision: str = Field(min_length=1)
    weight_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_evidence_url: str = Field(min_length=1)
    cutoff_evidence: CutoffEvidence
    tokenizer_revision: str = Field(min_length=1)
    chat_template_revision: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    serving_engine: str = Field(min_length=1)
    context_limit_verified: int = Field(ge=1)
    reasoning_config: str = Field(min_length=1)
    benchmark_artifact_id: str = Field(min_length=1)
    admitted: bool = False

    @property
    def released_at(self) -> dt.datetime:
        return self.cutoff_evidence.checkpoint_released_at


class AdoptionRecord(DurableModel):
    ID_FIELD = "adoption_id"

    adoption_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    role: str = Field(pattern=r"^(lead|challenger|extractor)$")
    temporal_policy: TemporalPolicy
    effective_start: dt.datetime
    eligibility_certificate_id: Optional[str] = None
    shadow_comparison_id: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "AdoptionRecord":
        require_aware(self.effective_start, "effective_start")
        if self.temporal_policy == TemporalPolicy.CUTOFF_QUALIFIED_BACKCAST:
            if not self.eligibility_certificate_id:
                raise ValueError(
                    "Policy B requires an explicit per-checkpoint eligibility certificate"
                )
        return self


class ModelRegistry:
    def __init__(self, *, allow_cutoff_qualified_backcast: bool = False):
        self._allow_policy_b = allow_cutoff_qualified_backcast
        self._checkpoints: dict[str, ModelCheckpoint] = {}
        self._adoptions: list[AdoptionRecord] = []

    def register(self, checkpoint: ModelCheckpoint) -> str:
        if checkpoint.checkpoint_id in self._checkpoints:
            raise ValueError(f"duplicate checkpoint_id {checkpoint.checkpoint_id}")
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        return checkpoint.checkpoint_id

    def checkpoint(self, checkpoint_id: str) -> ModelCheckpoint:
        return self._checkpoints[checkpoint_id]

    def adopt(self, record: AdoptionRecord) -> str:
        checkpoint = self._checkpoints.get(record.checkpoint_id)
        if checkpoint is None:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"unknown checkpoint {record.checkpoint_id}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="Register and verify the checkpoint before adoption.",
                )
            )
        if not checkpoint.admitted:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"checkpoint {record.checkpoint_id} has not passed the task benchmark gate",
                    code=ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                    stop_condition="A proposed upgrade passes only through the fixed-task benchmark (20.18).",
                )
            )
        if record.temporal_policy == TemporalPolicy.CUTOFF_QUALIFIED_BACKCAST and not self._allow_policy_b:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary="cutoff-qualified backcast is disabled for this registry",
                    code=ErrorCode.ASOF_VIOLATION,
                    stop_condition="Policy B runs are separately labelled and explicitly enabled (decision C5).",
                )
            )
        self._adoptions.append(record)
        return record.adoption_id

    def eligible_at(
        self, checkpoint_id: str, simulated_at: dt.datetime, *, policy: TemporalPolicy = TemporalPolicy.RELEASE_GATED
    ) -> bool:
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            return False
        simulated_at = require_aware(simulated_at, "simulated_at")
        if policy == TemporalPolicy.RELEASE_GATED:
            return checkpoint.released_at <= simulated_at
        return self._allow_policy_b and checkpoint.released_at <= simulated_at

    def route(
        self,
        role: str,
        simulated_at: dt.datetime,
        *,
        policy: TemporalPolicy = TemporalPolicy.RELEASE_GATED,
    ) -> str:
        simulated_at = require_aware(simulated_at, "simulated_at")
        candidates = [
            record
            for record in self._adoptions
            if record.role == role
            and record.effective_start <= simulated_at
            and record.temporal_policy == policy
            and self.eligible_at(record.checkpoint_id, simulated_at, policy=policy)
        ]
        if not candidates:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"no eligible {role} checkpoint at {simulated_at.isoformat()}",
                    code=ErrorCode.ASOF_VIOLATION,
                    stop_condition="Use the deterministic fallback; a release-gated route rejects checkpoints published after the simulated date.",
                )
            )
        latest = max(candidates, key=lambda r: r.effective_start)
        return latest.checkpoint_id

    def adoption_history(self) -> list[AdoptionRecord]:
        return list(self._adoptions)
