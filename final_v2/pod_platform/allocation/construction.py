"""Deterministic portfolio construction with official constraints
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 16.5-16.11; decision C4).

Construction policy: rank candidates by integrated alpha, select with an
incumbent retention band, size continuously under limits. Official limits
are never silently relaxed; internal soft objectives follow the declared
relaxation order and every relaxation is recorded.
"""

from __future__ import annotations

import datetime as dt
import math
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
    sector_neutral: bool = True
    min_sector_names: int = Field(ge=1, default=4)
    beta_neutral: bool = True
    beta_tilt: float = Field(ge=0.0, le=1.0, default=0.25)
    beta_lower: float = Field(gt=0, default=0.2)
    beta_upper: float = Field(gt=0, default=3.0)
    max_short_interest: float = Field(ge=0.0, le=1.5, default=0.15)
    min_price_usd: float = Field(ge=0.0, default=0.0)
    execution_days: int = Field(ge=1, default=5)
    long_adv_participation: float = Field(ge=0.0, le=1.0, default=0.0)
    short_adv_participation: float = Field(ge=0.0, le=1.0, default=0.0)
    require_verified_labels: bool = False
    capital_usd: float = Field(gt=0.0, default=1_000_000_000.0)
    enforce_max_abs_net: bool = False
    allow_all_ranked_sides: bool = False
    ceil_retention_zone: bool = False

    @model_validator(mode="after")
    def _check(self) -> "InternalPolicy":
        if self.retention_band >= 2.0:
            raise ValueError("retention band must be a tight band around k, not an unbounded keep-all")
        if self.beta_upper <= self.beta_lower:
            raise ValueError("beta_upper must exceed beta_lower")
        return self


@dataclass(frozen=True)
class Candidate:
    security_id: str
    alpha: float
    market_cap_usd: float
    tradable: bool = True
    incumbent: bool = False
    incumbent_long: Optional[bool] = None
    incumbent_short: Optional[bool] = None
    sector: str = "unknown"
    beta: float = 1.0
    short_interest_pct: float = float("nan")
    price_usd: float = 10.0
    adv_usd: float = 1_000_000_000.0
    verified_label: bool = True
    size_grp: str = "large"


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
        n_per_side = max(1, internal.target_positions // 2)
        one_usd = internal.capital_usd * (internal.target_gross / 2.0) / n_per_side

        base_tradable = [
            c
            for c in candidates
            if c.tradable
            and c.price_usd >= internal.min_price_usd
            and (not internal.require_verified_labels or c.verified_label)
        ]
        tradable = [
            c
            for c in base_tradable
            if internal.long_adv_participation <= 0.0
            or (c.adv_usd * internal.long_adv_participation * internal.execution_days >= one_usd)
        ]
        shortable = [
            c
            for c in base_tradable
            if c.market_cap_usd >= official.min_short_market_cap_usd
            and not (c.short_interest_pct >= internal.max_short_interest)
            and (
                internal.short_adv_participation <= 0.0
                or (c.adv_usd * internal.short_adv_participation * internal.execution_days >= one_usd)
            )
        ]

        def _tie_key(sid: str):
            return (0, int(sid)) if sid.isdigit() else (1, sid)

        if internal.allow_all_ranked_sides:
            longs = sorted(tradable, key=lambda c: (-c.alpha, _tie_key(c.security_id)))
            shorts = sorted(shortable, key=lambda c: (c.alpha, _tie_key(c.security_id)))
        else:
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

        if internal.sector_neutral:
            sector_counts: dict[str, int] = {}
            for c in list(longs) + list(shorts):
                sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1
            eligible_counts = {s: n for s, n in sector_counts.items() if n >= internal.min_sector_names}
            quota = _quota(eligible_counts, k)
            selected_longs: list[Candidate] = []
            selected_shorts: list[Candidate] = []
            for sector, k_sector in quota.items():
                if k_sector <= 0:
                    continue
                lo = [c for c in longs if c.sector == sector]
                so = [c for c in shorts if c.sector == sector]
                k_pair = min(k_sector, len(lo), len(so))
                if k_pair <= 0:
                    continue
                picked_shorts = self._select_with_retention(so, k_pair, side="short")
                short_ids = {c.security_id for c in picked_shorts}
                picked_longs = self._select_with_retention(
                    [c for c in lo if c.security_id not in short_ids], k_pair, side="long"
                )
                k_pair = min(len(picked_shorts), len(picked_longs))
                selected_longs += picked_longs[:k_pair]
                selected_shorts += picked_shorts[:k_pair]
        else:
            selected_longs = self._select_with_retention(longs, k, side="long")
            selected_shorts = self._select_with_retention(shorts, k, side="short")

        if not selected_longs or not selected_shorts:
            result.status = SolverStatus.INFEASIBLE
            result.official_results.append(
                ConstraintResult(
                    name="selection",
                    threshold="non-empty long and short books",
                    value=f"longs={len(selected_longs)} shorts={len(selected_shorts)}",
                    status="fail",
                )
            )
            return result

        half = Decimal(str(internal.target_gross)) / Decimal(2)
        long_notional = half
        short_notional = half
        if internal.beta_neutral:
            beta_low = Decimal(str(internal.beta_lower))
            beta_high = Decimal(str(internal.beta_upper))
            long_betas = [
                min(max(Decimal(str(c.beta)), beta_low), beta_high) for c in selected_longs
            ]
            short_betas = [
                min(max(Decimal(str(c.beta)), beta_low), beta_high) for c in selected_shorts
            ]
            long_beta = sum(long_betas) / Decimal(len(long_betas))
            short_beta = sum(short_betas) / Decimal(len(short_betas))
            gross = Decimal(str(internal.target_gross))
            tilt = Decimal(str(internal.beta_tilt))
            raw = gross * short_beta / (long_beta + short_beta)
            long_notional = min(max(raw, half * (1 - tilt)), half * (1 + tilt))
            if internal.enforce_max_abs_net:
                max_dev = Decimal(str(internal.max_abs_net)) / Decimal(2)
                long_notional = min(max(long_notional, half - max_dev), half + max_dev)
            short_notional = gross - long_notional

        long_unit = (long_notional / Decimal(len(selected_longs))).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
        short_unit = (short_notional / Decimal(len(selected_shorts))).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
        max_name = Decimal(str(official.max_abs_name_weight))
        if long_unit > max_name:
            long_unit = max_name
            result.relaxations.append(
                Relaxation(
                    step=1,
                    description="reduced desired gross to satisfy the single-name cap (long side)",
                    estimated_cost="lower gross exposure than requested",
                )
            )
            result.status = SolverStatus.RELAXED
        if short_unit > max_name:
            short_unit = max_name
            result.relaxations.append(
                Relaxation(
                    step=1,
                    description="reduced desired gross to satisfy the single-name cap (short side)",
                    estimated_cost="lower gross exposure than requested",
                )
            )
            result.status = SolverStatus.RELAXED

        positions = [
            TargetPosition(security_id=c.security_id, target_weight=long_unit)
            for c in selected_longs
        ] + [
            TargetPosition(security_id=c.security_id, target_weight=-short_unit)
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
        max_unit = max(long_unit, short_unit)
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
                value=str(max_unit),
                status="pass" if max_unit <= max_name + tol else "fail",
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
            "long_unit": str(long_unit),
            "short_unit": str(short_unit),
            "long_notional": str(long_notional),
            "short_notional": str(short_notional),
            "gross": str(gross),
            "net": str(net),
            "relaxation_order": " -> ".join(RELAXATION_ORDER),
        }
        return result

    def _select_with_retention(
        self, ranked: Sequence[Candidate], k: int, side: Optional[str] = None
    ) -> list[Candidate]:
        if k <= 0:
            return []
        zone_size = (
            int(math.ceil(k * (1.0 + self._internal.retention_band)))
            if self._internal.ceil_retention_zone
            else int(k * (1.0 + self._internal.retention_band))
        )

        def _is_inc(c: Candidate) -> bool:
            if side == "long" and c.incumbent_long is not None:
                return c.incumbent_long
            if side == "short" and c.incumbent_short is not None:
                return c.incumbent_short
            return c.incumbent

        incumbents = [c for c in ranked[:zone_size] if _is_inc(c)]
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


def _quota(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder split of total names proportional to sector counts."""
    if not counts or total <= 0:
        return {}
    base_sum = sum(counts.values())
    quota = {s: int(n / base_sum * total) for s, n in counts.items()}
    remainder = sorted(
        ((s, (n / base_sum * total) - quota[s]) for s, n in counts.items()),
        key=lambda x: (-x[1], x[0]),
    )
    left = total - sum(quota.values())
    for i in range(max(0, left)):
        quota[remainder[i % len(remainder)][0]] += 1
    return quota
