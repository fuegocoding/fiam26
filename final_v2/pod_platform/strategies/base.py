"""Strategy candidate interface and belief-to-return bridge
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 4, 15.1-15.3)."""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from pydantic import model_validator

from ..contracts.base import StrictModel, require_aware
from ..contracts.forecasts import Forecast, ReturnForecast, TargetPrediction


class Charter(str, enum.Enum):
    FUNDAMENTALS_EXPECTATIONS = "business_fundamentals_expectations"
    CORPORATE_EVENTS = "corporate_events_capital_allocation"
    VALUATION_RELATIVE_VALUE = "valuation_fundamental_relative_value"
    MARKET_DYNAMICS = "market_dynamics_information_diffusion"


@dataclass(frozen=True)
class SecurityFeatures:
    security_id: str
    as_of: dt.datetime
    values: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_aware(self.as_of, "as_of")

    def get(self, key: str) -> float:
        if key not in self.values:
            raise KeyError(
                f"feature {key!r} is not declared for {self.security_id}"
            )
        return self.values[key]


@dataclass(frozen=True)
class Belief:
    security_id: str
    target_id: str
    prediction: Decimal
    unit: str
    reference_prediction: Decimal
    reference_type: str
    information_cutoff: dt.datetime
    expected_resolution_at: dt.datetime

    def __post_init__(self) -> None:
        require_aware(self.information_cutoff, "information_cutoff")
        require_aware(self.expected_resolution_at, "expected_resolution_at")
        if self.expected_resolution_at <= self.information_cutoff:
            raise ValueError(
                "expected resolution must lie after the information cutoff"
            )


class RankSpreadBridge(StrictModel):
    bridge_id: str
    horizon_trading_days: int
    slope_bp_per_unit: Decimal
    unit: str = "decimal_return"

    @model_validator(mode="after")
    def _check_unit(self) -> "RankSpreadBridge":
        if self.unit != "decimal_return":
            raise ValueError(
                "bridge outputs must be decimal_return; convert at boundaries"
            )
        return self

    def expected_excess_return(
        self, belief: Belief, scaled_score: Decimal
    ) -> Decimal:
        return (
            self.slope_bp_per_unit * scaled_score / Decimal(10000)
        ).quantize(Decimal("0.000001"))


class StrategyCandidate:
    charter: Charter
    candidate_id: str
    mechanism_id: str
    source_family_ids: tuple[str, ...]
    target_id: str
    unit: str
    reference_type: str
    uses_pretrained_model: bool
    bridge: RankSpreadBridge

    def score(
        self, features: SecurityFeatures, expected_resolution_at: dt.datetime
    ) -> Optional[Belief]:
        raise NotImplementedError

    def scaled_score(self, belief: Belief) -> Decimal:
        raise NotImplementedError

    def forecast(
        self,
        features: SecurityFeatures,
        *,
        model_id: str,
        admission_certificate_id: str,
        expected_resolution_at: dt.datetime,
        expires_at: dt.datetime,
    ) -> Optional[Forecast]:
        belief = self.score(features, expected_resolution_at)
        if belief is None:
            return None
        scaled = self.scaled_score(belief)
        return Forecast(
            owner=self.candidate_id,
            created_at=features.as_of,
            forecast_id=f"{self.candidate_id}_{features.security_id}_{features.as_of.date().isoformat()}",
            strategy_id=self.candidate_id,
            model_id=model_id,
            admission_certificate_id=admission_certificate_id,
            security_id=features.security_id,
            information_cutoff=belief.information_cutoff,
            earliest_formation_at=features.as_of,
            target=TargetPrediction(
                target_id=belief.target_id,
                prediction=belief.prediction,
                unit=belief.unit,
                reference_prediction=belief.reference_prediction,
                reference_type=belief.reference_type,
                expected_resolution_at=expected_resolution_at,
            ),
            return_forecast=ReturnForecast(
                expected_excess_return=self.bridge.expected_excess_return(
                    belief, scaled
                ),
                unit="decimal_return",
                horizon_trading_days=self.bridge.horizon_trading_days,
                bridge_id=self.bridge.bridge_id,
                uncertainty_artifact_id=f"uncertainty_{self.candidate_id}",
            ),
            mechanism_ids=[self.mechanism_id],
            source_family_ids=list(self.source_family_ids),
            evidence_artifact_ids=[f"features_{features.security_id}"],
            expires_at=expires_at,
        )
