"""Typed error response and error codes (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.11-21.12)."""

from __future__ import annotations

import enum
from typing import Any, Optional

from pydantic import Field, model_validator

from .base import StrictModel

RETRYABLE_CODES = frozenset({"TRANSIENT_PROVIDER_FAILURE"})


class ErrorCode(str, enum.Enum):
    ASOF_VIOLATION = "ASOF_VIOLATION"
    LABEL_NOT_MATURE = "LABEL_NOT_MATURE"
    PARTITION_ACCESS_DENIED = "PARTITION_ACCESS_DENIED"
    SOURCE_NOT_FORMATION_ELIGIBLE = "SOURCE_NOT_FORMATION_ELIGIBLE"
    IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    FORECAST_EXPIRED = "FORECAST_EXPIRED"
    ADMISSION_CERTIFICATE_INVALID = "ADMISSION_CERTIFICATE_INVALID"
    PORTFOLIO_INFEASIBLE = "PORTFOLIO_INFEASIBLE"
    RISK_REVIEW_REQUIRED = "RISK_REVIEW_REQUIRED"
    LIQUIDITY_UNAVAILABLE = "LIQUIDITY_UNAVAILABLE"
    TRANSIENT_PROVIDER_FAILURE = "TRANSIENT_PROVIDER_FAILURE"
    ACCOUNTING_RECONCILIATION_FAILED = "ACCOUNTING_RECONCILIATION_FAILED"


class TypedError(StrictModel):
    code: ErrorCode
    retryable: bool = False
    stop_condition: Optional[str] = None

    @model_validator(mode="after")
    def _check_retryability(self) -> "TypedError":
        if self.retryable and self.code.value not in RETRYABLE_CODES:
            raise ValueError(
                f"only genuinely transient errors receive bounded retries; {self.code.value} cannot be retryable"
            )
        if self.retryable and not self.stop_condition:
            raise ValueError(
                "a retryable error must declare its stop condition"
            )
        return self


class ToolResponse(StrictModel):
    status: str = Field(pattern=r"^(success|warning|error)$")
    summary: str = Field(min_length=1)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    error: Optional[TypedError] = None

    @model_validator(mode="after")
    def _check_shape(self) -> "ToolResponse":
        if self.status == "error" and self.error is None:
            raise ValueError("an error response must carry a typed error")
        if self.status != "error" and self.error is not None:
            raise ValueError("only an error response may carry a typed error")
        return self

    @classmethod
    def failure(
        cls,
        summary: str,
        code: ErrorCode,
        *,
        retryable: bool = False,
        stop_condition: Optional[str] = None,
        provenance: Optional[dict[str, Any]] = None,
        limitations: Optional[list[str]] = None,
        next_actions: Optional[list[str]] = None,
    ) -> "ToolResponse":
        return cls(
            status="error",
            summary=summary,
            provenance=provenance or {},
            limitations=limitations or [],
            next_actions=next_actions or [],
            error=TypedError(
                code=code, retryable=retryable, stop_condition=stop_condition
            ),
        )


class PodPlatformError(Exception):
    def __init__(self, response: ToolResponse):
        if response.status != "error" or response.error is None:
            raise ValueError("PodPlatformError requires an error ToolResponse")
        self.response = response
        super().__init__(response.summary)
