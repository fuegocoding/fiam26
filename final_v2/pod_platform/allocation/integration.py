"""Family-level forecast integration (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 16.3 reference baseline).

Frozen reference policy (decision C2/C6):
- Group forecasts by economic mechanism family.
- Within a family, mean of eligible calibrated estimates.
- Across F admitted families, fixed equal weights summing to one.
- A missing family contributes zero active alpha with its missingness mask
  retained and its uncertainty inflated. Never raw-data zero imputation and
  never per-stock renormalization by coverage count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class FamilyEstimate:
    family_id: str
    security_id: str
    expected_excess_return: float
    uncertainty: float
    available: bool = True


@dataclass(frozen=True)
class IntegratedForecast:
    security_id: str
    alpha: float
    uncertainty: float
    families_total: int
    families_available: int
    missingness_mask: tuple[str, ...] = field(default_factory=tuple)

    @property
    def coverage_fraction(self) -> float:
        return (
            self.families_available / self.families_total
            if self.families_total
            else 0.0
        )


class FamilyIntegrator:
    def __init__(
        self,
        admitted_families: Sequence[str],
        *,
        neutral_uncertainty: float = 1.0,
        missing_uncertainty_multiplier: float = 2.0,
    ):
        if not admitted_families:
            raise ValueError("at least one admitted family is required")
        if len(set(admitted_families)) != len(admitted_families):
            raise ValueError("admitted families must be unique")
        if neutral_uncertainty <= 0:
            raise ValueError("neutral_uncertainty must be positive")
        if missing_uncertainty_multiplier < 1.0:
            raise ValueError("missing families cannot reduce uncertainty")
        self._families = tuple(admitted_families)
        self._neutral_uncertainty_value = neutral_uncertainty
        self._missing_multiplier = missing_uncertainty_multiplier

    @property
    def families(self) -> tuple[str, ...]:
        return self._families

    def integrate(
        self, estimates: Sequence[FamilyEstimate]
    ) -> IntegratedForecast:
        security_ids = {e.security_id for e in estimates}
        if len(security_ids) != 1:
            raise ValueError("estimates must belong to a single security")
        security_id = next(iter(security_ids))
        by_family: dict[str, FamilyEstimate] = {}
        for estimate in estimates:
            if estimate.family_id not in self._families:
                raise ValueError(
                    f"family {estimate.family_id!r} is not admitted"
                )
            if estimate.family_id in by_family:
                raise ValueError(
                    f"duplicate estimate for family {estimate.family_id!r}"
                )
            by_family[estimate.family_id] = estimate

        total = len(self._families)
        alpha_sum = 0.0
        variance_sum = 0.0
        missing: list[str] = []
        available = 0
        for family_id in self._families:
            estimate = by_family.get(family_id)
            if estimate is None or not estimate.available:
                missing.append(family_id)
                variance_sum += (
                    self._missing_multiplier * self._neutral_uncertainty_value
                ) ** 2
                continue
            available += 1
            alpha_sum += estimate.expected_excess_return
            variance_sum += estimate.uncertainty**2

        alpha = alpha_sum / total
        uncertainty = math.sqrt(variance_sum) / total
        return IntegratedForecast(
            security_id=security_id,
            alpha=alpha,
            uncertainty=uncertainty,
            families_total=total,
            families_available=available,
            missingness_mask=tuple(missing),
        )
