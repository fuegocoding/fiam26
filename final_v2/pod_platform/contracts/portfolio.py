"""Target book and risk review contracts (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.7-21.8)."""

from __future__ import annotations

import datetime as dt
import enum
from decimal import Decimal

from pydantic import Field, model_validator

from .base import DurableModel, StrictModel, finite_decimal, require_aware


class TargetPosition(StrictModel):
    security_id: str = Field(min_length=1)
    target_weight: Decimal

    @model_validator(mode="after")
    def _check(self) -> "TargetPosition":
        finite_decimal(self.target_weight, "target_weight")
        return self


class ConstraintResult(StrictModel):
    name: str = Field(min_length=1)
    threshold: str = Field(min_length=1)
    value: str = Field(min_length=1)
    status: str = Field(pattern=r"^(pass|fail|warn)$")


class Relaxation(StrictModel):
    step: int = Field(ge=1)
    description: str = Field(min_length=1)
    estimated_cost: str = Field(min_length=1)


class SolverStatus(str, enum.Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    RELAXED = "RELAXED"
    INFEASIBLE = "INFEASIBLE"


class TargetBook(DurableModel):
    ID_FIELD = "target_book_id"

    target_book_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    decision_at: dt.datetime
    forecast_publication_ids: list[str] = Field(min_length=1)
    current_book_snapshot_id: str = Field(min_length=1)
    nav_snapshot_id: str = Field(min_length=1)
    price_snapshot_id: str = Field(min_length=1)
    risk_envelope_version: str = Field(min_length=1)
    covariance_version: str = Field(min_length=1)
    cost_model_version: str = Field(min_length=1)
    positions: list[TargetPosition] = Field(default_factory=list)
    diagnostics: dict[str, str] = Field(default_factory=dict)
    official_constraint_results: list[ConstraintResult] = Field(default_factory=list)
    internal_constraint_results: list[ConstraintResult] = Field(default_factory=list)
    solver_status: SolverStatus
    relaxations: list[Relaxation] = Field(default_factory=list)
    attribution_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "TargetBook":
        require_aware(self.decision_at, "decision_at")
        if self.solver_status == SolverStatus.INFEASIBLE and self.positions:
            raise ValueError("an infeasible book cannot carry target positions")
        if self.solver_status != SolverStatus.INFEASIBLE and not self.positions:
            raise ValueError("a feasible target book must carry positions")
        if self.solver_status == SolverStatus.RELAXED and not self.relaxations:
            raise ValueError("a relaxed solve must record its relaxations")
        if self.relaxations and self.solver_status not in (
            SolverStatus.RELAXED,
            SolverStatus.FEASIBLE,
            SolverStatus.OPTIMAL,
        ):
            raise ValueError("relaxations are only valid on a solved book")
        names = [p.security_id for p in self.positions]
        if len(set(names)) != len(names):
            raise ValueError("target positions must be unique per security")
        if any(r.status == "fail" for r in self.official_constraint_results):
            raise ValueError("official constraints may never fail on a published target book")
        return self


class RiskDecision(str, enum.Enum):
    PASS = "PASS"
    PASS_WITH_REMEDY = "PASS_WITH_REMEDY"
    BLOCK = "BLOCK"
    INFEASIBLE = "INFEASIBLE"


class LimitCheck(StrictModel):
    name: str = Field(min_length=1)
    threshold: str = Field(min_length=1)
    value: str = Field(min_length=1)
    status: str = Field(pattern=r"^(pass|fail|warn)$")
    evidence: str = Field(min_length=1)


class StressResult(StrictModel):
    scenario_id: str = Field(min_length=1)
    impact: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class RiskReview(DurableModel):
    ID_FIELD = "risk_review_id"

    risk_review_id: str = Field(min_length=1)
    reviewed_book_id: str = Field(min_length=1)
    reviewed_kind: str = Field(pattern=r"^(target|actual)$")
    policy_version: str = Field(min_length=1)
    effective_at: dt.datetime
    expires_at: dt.datetime
    measured_exposures: dict[str, str] = Field(default_factory=dict)
    exposure_uncertainty: dict[str, str] = Field(default_factory=dict)
    limit_checks: list[LimitCheck] = Field(min_length=1)
    stress_results: list[StressResult] = Field(default_factory=list)
    decision: RiskDecision
    remedy_constraints: list[str] = Field(default_factory=list)
    agent_recommendation_ref: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "RiskReview":
        require_aware(self.effective_at, "effective_at")
        require_aware(self.expires_at, "expires_at")
        if self.expires_at <= self.effective_at:
            raise ValueError("expires_at must lie after effective_at")
        if self.decision in (RiskDecision.PASS_WITH_REMEDY, RiskDecision.BLOCK) and not self.remedy_constraints:
            raise ValueError(f"{self.decision.value} requires remedy constraints")
        if self.decision == RiskDecision.PASS and self.remedy_constraints:
            raise ValueError("a clean PASS cannot attach remedy constraints")
        return self

    def is_valid_at(self, when: dt.datetime) -> bool:
        when = require_aware(when, "when")
        return self.effective_at <= when < self.expires_at
