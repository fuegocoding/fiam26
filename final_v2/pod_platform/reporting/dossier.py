"""Decision dossier and holdings export (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.10, 28.4; W17)."""

from __future__ import annotations

import csv
import datetime as dt
import io
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..contracts.base import require_aware
from ..contracts.decisions import DecisionEvent
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse


@dataclass
class PositionDossier:
    security_id: str
    as_of: dt.datetime
    decision_events: list[DecisionEvent] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    forecast_ids: list[str] = field(default_factory=list)
    risk_review_ids: list[str] = field(default_factory=list)
    order_ids: list[str] = field(default_factory=list)
    fill_ids: list[str] = field(default_factory=list)

    def explanation(self) -> dict:
        return {
            "security_id": self.security_id,
            "as_of": self.as_of.isoformat(),
            "forecast_ids": list(self.forecast_ids),
            "risk_review_ids": list(self.risk_review_ids),
            "decision_events": [
                {
                    "event_id": event.event_id,
                    "actor_kind": event.actor_kind.value,
                    "chosen_action": event.chosen_action,
                    "rationale": event.rationale,
                    "authorizing_rule_or_certificate_id": event.authorizing_rule_or_certificate_id,
                    "resulting_artifact_id": event.resulting_artifact_id,
                }
                for event in self.decision_events
            ],
            "evidence_ids": list(self.evidence_ids),
            "order_ids": list(self.order_ids),
            "fill_ids": list(self.fill_ids),
        }


class DossierBuilder:
    def __init__(self) -> None:
        self._events: list[DecisionEvent] = []

    def record(self, event: DecisionEvent) -> None:
        self._events.append(event)

    def build(self, security_id: str, as_of: dt.datetime) -> PositionDossier:
        as_of = require_aware(as_of, "as_of")
        relevant = [
            event
            for event in self._events
            if event.effective_market_at <= as_of
            and (
                security_id in event.input_artifact_ids
                or security_id in event.resulting_artifact_id
                or security_id in event.chosen_action
            )
        ]
        if not relevant:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"no stored decision evidence for {security_id} at {as_of.isoformat()}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="An arbitrary position/date must be explainable from stored evidence; if not, the run is incomplete.",
                )
            )
        dossier = PositionDossier(security_id=security_id, as_of=as_of)
        for event in sorted(relevant, key=lambda e: e.effective_market_at):
            dossier.decision_events.append(event)
            dossier.evidence_ids.extend(event.input_artifact_ids)
            dossier.evidence_ids.append(event.resulting_artifact_id)
            if "forecast" in event.chosen_action or "forecast" in event.resulting_artifact_id:
                dossier.forecast_ids.append(event.resulting_artifact_id)
            if "review" in event.resulting_artifact_id:
                dossier.risk_review_ids.append(event.resulting_artifact_id)
            if "order" in event.resulting_artifact_id:
                dossier.order_ids.append(event.resulting_artifact_id)
            if "fill" in event.resulting_artifact_id:
                dossier.fill_ids.append(event.resulting_artifact_id)
        return dossier


@dataclass(frozen=True)
class HoldingsRow:
    security_id: str
    ticker: str
    name: str
    label_source: str
    label_reference_date: str
    quantity: float
    weight: float
    side: str


def holdings_csv(rows: Sequence[HoldingsRow]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "security_id",
            "ticker",
            "name",
            "label_source",
            "label_reference_date",
            "quantity",
            "weight",
            "side",
        ]
    )
    for row in rows:
        if not row.label_source or not row.label_reference_date:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"holding {row.security_id} lacks identity provenance",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="Every exported holding carries its label source and reference date.",
                )
            )
        writer.writerow(
            [
                row.security_id,
                row.ticker,
                row.name,
                row.label_source,
                row.label_reference_date,
                row.quantity,
                row.weight,
                row.side,
            ]
        )
    return buffer.getvalue()


def returns_csv(rows: Sequence[tuple[str, float]], label: str = "period") -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([label, "net_return"])
    for period, value in rows:
        writer.writerow([period, value])
    return buffer.getvalue()


def verify_deck_numbers(
    reported: dict[str, float], artifacts: dict[str, float], tolerance: float = 1e-9
) -> None:
    for key, value in reported.items():
        if key not in artifacts:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"deck number {key!r} has no backing artifact",
                    code=ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    stop_condition="Every slide metric references an artifact.",
                )
            )
        if abs(artifacts[key] - value) > tolerance:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"deck number {key!r} disagrees with artifact",
                    code=ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    stop_condition="Reporting reconciliation must pass before the report manifest freezes.",
                )
            )
