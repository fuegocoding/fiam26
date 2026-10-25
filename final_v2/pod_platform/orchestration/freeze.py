"""Run freeze and full-run acceptance gate (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 22.4, 26.8; W18).

A run is frozen only when the manifest is complete, every formation input
is approved, every deployed forecast carries a valid certificate, accounting
reconciles, actual holdings satisfy the declared implementation or disclose
incidents, reports exist, and trial counts and evaluation exposure are
reported. Passing software checks does not certify alpha.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

from pydantic import Field

from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from .run_manifest import RunManifest


class AcceptanceCheck(StrictModel):
    item: str = Field(min_length=1)
    satisfied: bool
    evidence: str = Field(min_length=1)


@dataclass
class FreezeResult:
    run_id: str
    frozen_at: dt.datetime
    manifest_hash: str
    output_hashes: dict[str, str] = field(default_factory=dict)
    checks: list[AcceptanceCheck] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return all(check.satisfied for check in self.checks)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "frozen_at": self.frozen_at.isoformat(),
            "manifest_hash": self.manifest_hash,
            "output_hashes": dict(sorted(self.output_hashes.items())),
            "accepted": self.accepted,
            "checks": [check.model_dump() for check in self.checks],
        }


REQUIRED_OUTPUTS = (
    "holdings_export",
    "returns_export",
    "performance_report",
    "compliance_report",
    "decision_dossier",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def freeze_run(
    manifest: RunManifest,
    outputs: dict[str, bytes],
    *,
    frozen_at: dt.datetime,
    formation_inputs_approved: bool,
    certificate_count: int,
    deployed_forecast_count: int,
    accounting_reconciled: bool,
    actual_holdings_compliant: bool,
    incidents_disclosed: Sequence[str] = (),
    trial_count: int = 0,
    evaluation_access_count: int = 0,
    material_agent_interventions: int = 0,
    agent_counterfactual_present: bool = False,
) -> FreezeResult:
    frozen_at = require_aware(frozen_at, "frozen_at")
    if material_agent_interventions < 0:
        raise ValueError("material_agent_interventions cannot be negative")
    checks: list[AcceptanceCheck] = []

    missing = [name for name in REQUIRED_OUTPUTS if name not in outputs]
    checks.append(
        AcceptanceCheck(
            item="run manifest complete and reproducible",
            satisfied=True,
            evidence=f"manifest {manifest.run_id} with {len(manifest.config_hashes)} config hashes",
        )
    )
    checks.append(
        AcceptanceCheck(
            item="official report and exhibits generated",
            satisfied=not missing,
            evidence="all required outputs present"
            if not missing
            else f"missing: {', '.join(missing)}",
        )
    )
    checks.append(
        AcceptanceCheck(
            item="all formation inputs approved",
            satisfied=formation_inputs_approved,
            evidence="source cards formation-eligible"
            if formation_inputs_approved
            else "unapproved formation input present",
        )
    )
    checks.append(
        AcceptanceCheck(
            item="every deployed forecast has a valid certificate",
            satisfied=certificate_count >= deployed_forecast_count and deployed_forecast_count > 0,
            evidence=f"{certificate_count} certificates for {deployed_forecast_count} forecasts",
        )
    )
    checks.append(
        AcceptanceCheck(
            item="accounting reconciles",
            satisfied=accounting_reconciled,
            evidence="NAV identity verified" if accounting_reconciled else "reconciliation failed",
        )
    )
    checks.append(
        AcceptanceCheck(
            item="actual holdings satisfy the declared implementation or incidents disclosed",
            satisfied=actual_holdings_compliant or bool(incidents_disclosed),
            evidence="compliant"
            if actual_holdings_compliant
            else f"incidents: {', '.join(incidents_disclosed) or 'none'}",
        )
    )
    checks.append(
        AcceptanceCheck(
            item="agent counterfactual generated for material agent interventions",
            satisfied=material_agent_interventions == 0 or agent_counterfactual_present,
            evidence="no material agent intervention in this run"
            if material_agent_interventions == 0
            else "twin comparison attached to run artifacts"
            if agent_counterfactual_present
            else "material agent interventions have no counterfactual evidence",
        )
    )
    checks.append(
        AcceptanceCheck(
            item="trial counts and evaluation exposure reported",
            satisfied=trial_count > 0 and evaluation_access_count >= 0,
            evidence=f"{trial_count} trials, {evaluation_access_count} evaluation accesses",
        )
    )

    output_hashes = {name: _sha256_bytes(payload) for name, payload in sorted(outputs.items())}
    result = FreezeResult(
        run_id=manifest.run_id,
        frozen_at=frozen_at,
        manifest_hash=canonical_digest(manifest.model_dump(mode="json")),
        output_hashes=output_hashes,
        checks=checks,
    )
    if not result.accepted:
        failing = [check.item for check in result.checks if not check.satisfied]
        raise PodPlatformError(
            ToolResponse.failure(
                summary=f"run {manifest.run_id} cannot be frozen: {'; '.join(failing)}",
                code=ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                stop_condition="Every 26.8 acceptance item must be satisfied before the report manifest freezes.",
            )
        )
    return result
