"""Compliance on actual holdings (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 16.5, 26.5).

Checked against actual holdings, not targets: name count, gross, net,
single-name cap, and short-eligibility market-cap floor. Official breaches
fail closed with a typed error; internal breaches are recorded warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from pydantic import Field

from ..allocation.construction import OfficialLimits
from ..contracts.base import StrictModel
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse

ZERO = Decimal("0")


@dataclass(frozen=True)
class ActualHolding:
    security_id: str
    weight: Decimal
    market_cap_usd: Decimal
    formation_market_cap_usd: Decimal

    def __post_init__(self) -> None:
        if not self.security_id:
            raise ValueError("security_id must be non-empty")


class ComplianceCheck(StrictModel):
    name: str
    threshold: str
    value: str
    status: str = Field(pattern=r"^(pass|fail|warn)$")
    scope: str = Field(pattern=r"^(official|internal)$")


@dataclass
class ComplianceReport:
    checks: list[ComplianceCheck] = field(default_factory=list)
    incidents: list[str] = field(default_factory=list)

    @property
    def official_pass(self) -> bool:
        return all(c.status != "fail" for c in self.checks if c.scope == "official")

    def as_dict(self) -> dict:
        return {
            "official_pass": self.official_pass,
            "checks": [c.model_dump() for c in self.checks],
            "incidents": list(self.incidents),
        }


class ComplianceEngine:
    def __init__(self, official: OfficialLimits):
        self._official = official

    def check(
        self,
        holdings: Sequence[ActualHolding],
        *,
        max_abs_internal_net: Decimal = Decimal("0.10"),
    ) -> ComplianceReport:
        report = ComplianceReport()
        names = [h.security_id for h in holdings]
        if len(set(names)) != len(names):
            raise PodPlatformError(
                ToolResponse.failure(
                    summary="duplicate security in actual holdings",
                    code=ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    stop_condition="Actual holdings must be unique per security.",
                )
            )
        gross = sum((abs(h.weight) for h in holdings), ZERO)
        net = sum((h.weight for h in holdings), ZERO)
        max_name = max((abs(h.weight) for h in holdings), default=ZERO)
        shorts = [h for h in holdings if h.weight < ZERO]
        min_short_cap = min((h.formation_market_cap_usd for h in shorts), default=None)

        report.checks.append(
            ComplianceCheck(
                name="position_count",
                threshold=f"{self._official.min_positions}..{self._official.max_positions}",
                value=str(len(holdings)),
                status="pass"
                if self._official.min_positions <= len(holdings) <= self._official.max_positions
                else "fail",
                scope="official",
            )
        )
        report.checks.append(
            ComplianceCheck(
                name="gross",
                threshold=f"<= {self._official.max_gross}",
                value=str(gross),
                status="pass" if gross <= Decimal(str(self._official.max_gross)) else "fail",
                scope="official",
            )
        )
        report.checks.append(
            ComplianceCheck(
                name="net",
                threshold=f"[{self._official.min_net}, {self._official.max_net}]",
                value=str(net),
                status="pass"
                if Decimal(str(self._official.min_net)) <= net <= Decimal(str(self._official.max_net))
                else "fail",
                scope="official",
            )
        )
        report.checks.append(
            ComplianceCheck(
                name="single_name",
                threshold=f"<= {self._official.max_abs_name_weight}",
                value=str(max_name),
                status="pass"
                if max_name <= Decimal(str(self._official.max_abs_name_weight))
                else "fail",
                scope="official",
            )
        )
        short_ok = min_short_cap is None or min_short_cap >= Decimal(
            str(self._official.min_short_market_cap_usd)
        )
        report.checks.append(
            ComplianceCheck(
                name="short_eligibility",
                threshold=f">= {self._official.min_short_market_cap_usd}",
                value=str(min_short_cap) if min_short_cap is not None else "no shorts",
                status="pass" if short_ok else "fail",
                scope="official",
            )
        )
        report.checks.append(
            ComplianceCheck(
                name="abs_net_internal",
                threshold=f"<= {max_abs_internal_net}",
                value=str(abs(net)),
                status="pass" if abs(net) <= max_abs_internal_net else "warn",
                scope="internal",
            )
        )
        for check in report.checks:
            if check.status == "fail":
                report.incidents.append(f"compliance_fail:{check.name}")
        return report

    def enforce(self, holdings: Sequence[ActualHolding]) -> ComplianceReport:
        report = self.check(holdings)
        if not report.official_pass:
            failing = [c.name for c in report.checks if c.status == "fail" and c.scope == "official"]
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"actual holdings breach official limits: {', '.join(failing)}",
                    code=ErrorCode.PORTFOLIO_INFEASIBLE,
                    stop_condition="Official limits are checked on actual holdings and never silently relaxed.",
                )
            )
        return report
