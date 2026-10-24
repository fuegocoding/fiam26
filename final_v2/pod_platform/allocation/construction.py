"""Deterministic portfolio construction with official constraints
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 16.5-16.11; decision C4).

Construction policy: rank candidates by integrated alpha, select with an
incumbent retention band, size continuously under limits. Official limits
are never silently relaxed; internal soft objectives follow the declared
relaxation order and every relaxation is recorded.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Optional, Sequence

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from ..contracts.portfolio import (
    ConstraintResult,
    Relaxation,
    SolverStatus,
    TargetBook,
    TargetPosition,
)


class OfficialLimits(StrictModel):
    min_positions: int = Field(ge=1, default=200)
    max_positions: int = Field(ge=1, default=500)
    max_gross: float = Field(gt=0, default=2.0)
    min_net: float = -0.5
    max_net: float = 0.5
    max_abs_name_weight: float = Field(gt=0, default=0.05)
    min_short_market_cap_usd: float = Field(ge=0, default=500_000_000.0)

    @model_validator(mode="after")
    def _check(self) -> "OfficialLimits":
        if self.max_positions < self.min_positions:
            raise ValueError("max_positions must be at least min_positions")
        if self.min_net > self.max_net:
            raise ValueError("min_net must not exceed max_net")
        return self


class InternalPolicy(StrictModel):
    target_positions: int = Field(ge=1, default=300)
    target_gross: float = Field(gt=0, default=1.0)
    target_net: float = 0.0
    max_abs_net: float = Field(ge=0, default=0.10)
    retention_band: float = Field(ge=0.0, default=0.5)

    @model_validator(mode="after")
    def _check(self) -> "InternalPolicy":
        if self.retention_band >= 2.0:
            raise ValueError("retention band must be a tight band around k, not an unbounded keep-all")
        return self


@dataclass(frozen=True)
class Candidate:
    security_id: str
    alpha: float
    market_cap_usd: float
    tradable: bool = True
    incumbent: bool = False
    sector: str = "unknown"


@dataclass
class ConstructionResult:
    status: SolverStatus
    positions: list[TargetPosition] = field(default_factory=list)
    official_results: list[ConstraintResult] = field(default_factory=list)
    internal_results: list[ConstraintResult] = field(default_factory=list)
    relaxations: list[Relaxation] = field(default_factory=list)
    diagnostics: dict[str, str] = field(default_factory=dict)


RELAXATION_ORDER = (
    "reduce desired gross",
    "reduce changes from incumbents",
    "relax internal style target within tolerance",
    "reconsider candidate selection",
    "return infeasible",
)


class ConstructionEngine:
    def __init__(self, official: OfficialLimits, internal: InternalPolicy):
        self._official = official
        self._internal = internal

    def build(
        self,
        candidates: Sequence[Candidate],
        *,
        run_id: str,
        decision_at: dt.datetime,
        forecast_publication_ids: Sequence[str],
        current_book_snapshot_id: str,
        nav_snapshot_id: str,
        price_snapshot_id: str,
        risk_envelope_version: str,
        covariance_version: str,
        cost_model_version: str,
        attribution_method: str,
        owner: str = "allocation",
    ) -> TargetBook:
        result = self._construct(candidates)
        if result.status == SolverStatus.INFEASIBLE:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary="construction infeasible under official limits",
                    code=ErrorCode.PORTFOLIO_INFEASIBLE,
                    stop_condition="Official limits are never silently relaxed; reconsider candidates or report infeasibility.",
                )
            )
        return TargetBook(
            owner=owner,
            created_at=decision_at,
            target_book_id=f"book_{run_id}",
            run_id=run_id,
            decision_at=decision_at,
            forecast_publication_ids=list(forecast_publication_ids),
            current_book_snapshot_id=current_book_snapshot_id,
            nav_snapshot_id=nav_snapshot_id,
            price_snapshot_id=price_snapshot_id,
            risk_envelope_version=risk_envelope_version,
            covariance_version=covariance_version,
            cost_model_version=cost_model_version,
            positions=result.positions,
            diagnostics=result.diagnostics,
            official_constraint_results=result.official_results,
            internal_constraint_results=result.internal_results,
            solver_status=result.status,
            relaxations=result.relaxations,
            attribution_method=attribution_method,
        )

    def _construct(self, candidates: Sequence[Candidate]) -> ConstructionResult:
        official = self._official
        internal = self._internal
        tradable = [c for c in candidates if c.tradable]
        shortable = [c for c in tradable if c.market_cap_usd >= official.min_short_market_cap_usd]

        longs = sorted(
            (c for c in tradable if c.alpha > 0),
            key=lambda c: (-c.alpha, c.security_id),
        )
        shorts = sorted(
            (c for c in shortable if c.alpha < 0),
            key=lambda c: (c.alpha, c.security_id),
        )

        result = ConstructionResult(status=SolverStatus.FEASIBLE)
        k = min(internal.target_positions // 2, len(longs), len(shorts))
        if k * 2 < official.min_positions:
            result.status = SolverStatus.INFEASIBLE
            result.official_results.append(
                ConstraintResult(
                    name="position_count",
                    threshold=f">= {official.min_positions}",
                    value=str(k * 2),
                    status="fail",
                )
            )
            return result

        selected_longs = self._select_with_retention(longs, k)
        selected_shorts = self._select_with_retention(shorts, k)

        weight = Decimal(str(internal.target_gross)) / Decimal(2 * k)
        weight = weight.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
        if weight > Decimal(str(official.max_abs_name_weight)):
            weight = Decimal(str(official.max_abs_name_weight)).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
            result.relaxations.append(
                Relaxation(
                    step=1,
                    description="reduced desired gross to satisfy the single-name cap",
                    estimated_cost="lower gross exposure than requested",
                )
            )
            result.status = SolverStatus.RELAXED

        positions = [
            TargetPosition(security_id=c.security_id, target_weight=weight)
            for c in selected_longs
        ] + [
            TargetPosition(security_id=c.security_id, target_weight=-weight)
            for c in selected_shorts
        ]

        gross = sum((abs(p.target_weight) for p in positions), Decimal("0"))
        net = sum((p.target_weight for p in positions), Decimal("0"))
        count = len(positions)

        tol = Decimal("0.000000000001")
        max_gross = Decimal(str(official.max_gross))
        min_net = Decimal(str(official.min_net))
        max_net = Decimal(str(official.max_net))
        max_name = Decimal(str(official.max_abs_name_weight))
        result.official_results = [
            ConstraintResult(
                name="position_count",
                threshold=f"{official.min_positions}..{official.max_positions}",
                value=str(count),
                status="pass" if official.min_positions <= count <= official.max_positions else "fail",
            ),
            ConstraintResult(
                name="gross",
                threshold=f"<= {official.max_gross}",
                value=str(gross),
                status="pass" if gross <= max_gross + tol else "fail",
            ),
            ConstraintResult(
                name="net",
                threshold=f"[{official.min_net}, {official.max_net}]",
                value=str(net),
                status="pass" if min_net - tol <= net <= max_net + tol else "fail",
            ),
            ConstraintResult(
                name="max_abs_name_weight",
                threshold=f"<= {official.max_abs_name_weight}",
                value=str(weight),
                status="pass" if weight <= max_name + tol else "fail",
            ),
            ConstraintResult(
                name="short_market_cap",
                threshold=f">= {official.min_short_market_cap_usd}",
                value=str(min((c.market_cap_usd for c in selected_shorts), default=0.0)),
                status="pass",
            ),
        ]
        if any(r.status == "fail" for r in result.official_results):
            result.status = SolverStatus.INFEASIBLE
            return result

        result.internal_results = [
            ConstraintResult(
                name="abs_net",
                threshold=f"<= {internal.max_abs_net}",
                value=str(abs(net)),
                status="pass" if abs(net) <= Decimal(str(internal.max_abs_net)) + tol else "warn",
            ),
        ]
        result.positions = positions
        result.diagnostics = {
            "selected_longs": str(len(selected_longs)),
            "selected_shorts": str(len(selected_shorts)),
            "name_weight": str(weight),
            "gross": str(gross),
            "net": str(net),
            "relaxation_order": " -> ".join(RELAXATION_ORDER),
        }
        return result

    def _select_with_retention(self, ranked: Sequence[Candidate], k: int) -> list[Candidate]:
        if k <= 0:
            return []
        incumbents = [c for c in ranked[: int(k * (1.0 + self._internal.retention_band))] if c.incumbent]
        selected: list[Candidate] = []
        seen: set[str] = set()
        for candidate in incumbents[:k]:
            selected.append(candidate)
            seen.add(candidate.security_id)
        for candidate in ranked:
            if len(selected) >= k:
                break
            if candidate.security_id not in seen:
                selected.append(candidate)
                seen.add(candidate.security_id)
        return selected
