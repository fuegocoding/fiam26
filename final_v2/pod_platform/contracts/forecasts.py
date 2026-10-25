"""Forecast publication contract (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.6, 15.2)."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from pydantic import Field, model_validator

from .base import DurableModel, StrictModel, finite_decimal, require_aware

RETURN_UNIT = "decimal_return"
KNOWN_TARGET_UNITS = frozenset(
    {
        "decimal_margin_change",
        "decimal_growth_rate",
        "decimal_surprise",
        "probability",
        "decimal_return",
        "centered_rank",
        "usd_per_share",
    }
)


class TargetPrediction(StrictModel):
    target_id: str = Field(min_length=1)
    prediction: Decimal
    unit: str = Field(min_length=1)
    reference_prediction: Decimal
    reference_type: str = Field(min_length=1)
    expected_resolution_at: dt.datetime

    @model_validator(mode="after")
    def _check(self) -> "TargetPrediction":
        finite_decimal(self.prediction, "prediction")
        finite_decimal(self.reference_prediction, "reference_prediction")
        if self.unit not in KNOWN_TARGET_UNITS:
            raise ValueError(
                f"unit {self.unit!r} is not a declared target unit"
            )
        require_aware(self.expected_resolution_at, "expected_resolution_at")
        return self


class ReturnForecast(StrictModel):
    expected_excess_return: Decimal
    unit: str = Field(min_length=1)
    horizon_trading_days: int = Field(ge=1)
    bridge_id: str = Field(min_length=1)
    uncertainty_artifact_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "ReturnForecast":
        finite_decimal(self.expected_excess_return, "expected_excess_return")
        if self.unit != RETURN_UNIT:
            raise ValueError(
                "return forecasts must be stored in decimal_return; convert at boundaries"
            )
        return self


class Forecast(DurableModel):
    ID_FIELD = "forecast_id"

    forecast_id: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    admission_certificate_id: str = Field(min_length=1)
    security_id: str = Field(min_length=1)
    information_cutoff: dt.datetime
    earliest_formation_at: dt.datetime
    target: TargetPrediction
    return_forecast: ReturnForecast
    mechanism_ids: list[str] = Field(min_length=1)
    source_family_ids: list[str] = Field(min_length=1)
    event_id: str | None = None
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    expires_at: dt.datetime

    @model_validator(mode="after")
    def _check_temporal_order(self) -> "Forecast":
        require_aware(self.information_cutoff, "information_cutoff")
        require_aware(self.earliest_formation_at, "earliest_formation_at")
        require_aware(self.expires_at, "expires_at")
        if self.earliest_formation_at < self.information_cutoff:
            raise ValueError("formation cannot precede the information cutoff")
        if self.target.expected_resolution_at <= self.information_cutoff:
            raise ValueError(
                "expected resolution must be estimated from information available at the cutoff and lie after it"
            )
        if self.expires_at <= self.information_cutoff:
            raise ValueError("expires_at must lie after the information cutoff")
        return self

    @model_validator(mode="after")
    def _check_no_future_evidence(self) -> "Forecast":
        for artifact_id in self.evidence_artifact_ids:
            if not artifact_id:
                raise ValueError("evidence artifact ids must be non-empty")
        return self

    def is_expired(self, as_of: dt.datetime) -> bool:
        return require_aware(as_of, "as_of") >= self.expires_at
