"""Independent risk envelope (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 17).

Risk reviews target or actual books against a declared policy version and
emit the RiskReview contract (21.8). Execution cannot accept a book from an
expired review or a changed target without revalidation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Sequence

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, require_aware
from ..contracts.portfolio import LimitCheck, RiskDecision, RiskReview, StressResult, TargetBook


class RiskLimit(StrictModel):
    name: str = Field(min_length=1)
    threshold: str = Field(min_length=1)
    hard: bool = True


class RiskPolicy(StrictModel):
    policy_version: str = Field(min_length=1)
    limits: list[RiskLimit] = Field(min_length=1)
    review_ttl_minutes: int = Field(ge=1, default=60)

    @model_validator(mode="after")
    def _check(self) -> "RiskPolicy":
        names = [limit.name for limit in self.limits]
        if len(set(names)) != len(names):
            raise ValueError("limit names must be unique")
        return self


@dataclass(frozen=True)
class MeasuredExposure:
    name: str
    value: float
    threshold: float
    is_max: bool
    uncertainty: Optional[float] = None

    def status(self) -> str:
        if self.is_max:
            return "pass" if self.value <= self.threshold + 1e-12 else "fail"
        return "pass" if self.value >= self.threshold - 1e-12 else "fail"


class RiskEngine:
    def __init__(self, policy: RiskPolicy):
        self._policy = policy

    def review(
        self,
        book: TargetBook,
        exposures: Sequence[MeasuredExposure],
        *,
        stress_results: Sequence[StressResult] = (),
        effective_at: dt.datetime,
        reviewed_kind: str = "target",
        agent_recommendation_ref: Optional[str] = None,
    ) -> RiskReview:
        require_aware(effective_at, "effective_at")
        by_name = {exposure.name: exposure for exposure in exposures}
        checks: list[LimitCheck] = []
        for limit in self._policy.limits:
            exposure = by_name.get(limit.name)
            if exposure is None:
                raise ValueError(f"missing measured exposure for limit {limit.name!r}")
            checks.append(
                LimitCheck(
                    name=limit.name,
                    threshold=limit.threshold,
                    value=str(exposure.value),
                    status=exposure.status(),
                    evidence=f"measured from book {book.target_book_id}",
                )
            )
        failures_hard = [
            check
            for check, limit in zip(checks, self._policy.limits)
            if check.status == "fail" and limit.hard
        ]
        failures_soft = [
            check
            for check, limit in zip(checks, self._policy.limits)
            if check.status == "fail" and not limit.hard
        ]
        if failures_hard:
            decision = RiskDecision.BLOCK
            remedies = [f"remedy {check.name}: {check.threshold}" for check in failures_hard]
        elif failures_soft:
            decision = RiskDecision.PASS_WITH_REMEDY
            remedies = [f"remedy soft limit {check.name}: {check.threshold}" for check in failures_soft]
        else:
            decision = RiskDecision.PASS
            remedies = []
        return RiskReview(
            owner="risk",
            created_at=effective_at,
            risk_review_id=f"review_{book.target_book_id}",
            reviewed_book_id=book.target_book_id,
            reviewed_kind=reviewed_kind,
            policy_version=self._policy.policy_version,
            effective_at=effective_at,
            expires_at=effective_at + dt.timedelta(minutes=self._policy.review_ttl_minutes),
            measured_exposures={e.name: str(e.value) for e in exposures},
            exposure_uncertainty={
                e.name: str(e.uncertainty) for e in exposures if e.uncertainty is not None
            },
            limit_checks=checks,
            stress_results=list(stress_results),
            decision=decision,
            remedy_constraints=remedies,
            agent_recommendation_ref=agent_recommendation_ref,
        )
