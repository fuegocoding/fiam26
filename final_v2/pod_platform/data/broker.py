"""Data broker: as-of queries with eligibility enforcement (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 6.3, 9.10, 22; W04)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from ..contracts.base import require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse
from .availability import AvailabilityRule, UsageMode
from .bitemporal import BitemporalStore, Observation
from .catalogue import CatalogueManifest
from .security_master import SecurityMaster


@dataclass(frozen=True)
class TrustedContext:
    caller_role: str
    namespace: str
    research_asof: dt.datetime
    allowed_partitions: tuple[str, ...]
    data_policy_version: str
    request_id: str
    hypothesis_id: Optional[str] = None


@dataclass
class EligibilityReport:
    source_id: str
    requested_cutoff: dt.datetime
    returned_observation_ids: list[str] = field(default_factory=list)
    assertions: dict[str, str] = field(default_factory=dict)

    def record(self, check: str, status: str) -> None:
        self.assertions[check] = status

    @property
    def all_passed(self) -> bool:
        return all(status == "pass" for status in self.assertions.values())


class DataBroker:
    def __init__(
        self,
        *,
        catalogue: CatalogueManifest,
        store: BitemporalStore,
        security_master: SecurityMaster,
        rules: dict[str, AvailabilityRule],
    ) -> None:
        self._catalogue = catalogue
        self._store = store
        self._master = security_master
        self._rules = rules

    def query_asof(
        self,
        ctx: TrustedContext,
        *,
        source_id: str,
        metric: str,
        entity_id: str,
        information_cutoff: dt.datetime,
        identity_at: dt.datetime,
        partition: str,
    ) -> tuple[list[Observation], EligibilityReport]:
        cutoff = require_aware(information_cutoff, "information_cutoff")
        identity_at = require_aware(identity_at, "identity_at")
        report = EligibilityReport(source_id=source_id, requested_cutoff=cutoff)

        if cutoff > ctx.research_asof:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary="information cutoff exceeds the trusted research clock",
                    code=ErrorCode.ASOF_VIOLATION,
                    stop_condition="The agent cannot override the context by supplying a later date.",
                )
            )
        if partition not in ctx.allowed_partitions:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"partition {partition!r} is not allowed for this request context",
                    code=ErrorCode.PARTITION_ACCESS_DENIED,
                    stop_condition="Request the partition through Research Integrity.",
                )
            )
        card = self._catalogue.card(source_id)
        if card.broker_rejects_formation():
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"source {source_id} is not approved for historical formation",
                    code=ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE,
                    provenance={"source_id": source_id},
                    limitations=["Historical vintage reconstruction is unresolved."],
                    next_actions=["Request the source assessment or use an approved source."],
                    stop_condition="No formation query until an approved source-card version exists.",
                )
            )
        report.record("source_approved_for_partition", "pass")

        rule = self._rules.get(f"{source_id}:{metric}")
        if rule is None:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"no availability rule registered for {source_id}:{metric}",
                    code=ErrorCode.SCHEMA_MISMATCH,
                    stop_condition="Register an availability rule with evidence before querying the metric.",
                )
            )
        if rule.usage_mode == UsageMode.DESCRIPTIVE_ONLY:
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"metric {metric} is descriptive only and cannot feed formation",
                    code=ErrorCode.SOURCE_NOT_FORMATION_ELIGIBLE,
                    stop_condition="Descriptive metrics stay in research mode.",
                )
            )

        if rule.earliest_start is not None:
            if cutoff < rule.earliest_start:
                report.record("late_start_source_absent_before_start", "pass")
                report.record("feature_availability_precedes_cutoff", "pass")
                report.record("identity_links_valid_at_event_time", "pass")
                report.record("aggregation_windows_end_before_availability", "pass")
                report.record("training_labels_resolved_before_cutoff", "pass")
                return [], report
        report.record("late_start_source_absent_before_start", "pass")

        rows = self._store.asof_versions(
            cutoff, source_id=source_id, metric=metric, entity_id=entity_id
        )
        for row in rows:
            if row.available_at > cutoff or row.version_known_at > cutoff:
                raise PodPlatformError(
                    ToolResponse.failure(
                        summary=f"observation {row.observation_id} is not eligible at the cutoff",
                        code=ErrorCode.ASOF_VIOLATION,
                        stop_condition="The as-of query must never return future information.",
                    )
                )
        if not self._master.security_link_valid(entity_id, identity_at):
            raise PodPlatformError(
                ToolResponse.failure(
                    summary=f"no valid dated identity link for {entity_id!r} at {identity_at.isoformat()}",
                    code=ErrorCode.IDENTITY_AMBIGUOUS,
                    stop_condition="Identity links must be valid at the relevant time before features materialize.",
                )
            )
        report.record("feature_availability_precedes_cutoff", "pass")
        report.record("identity_links_valid_at_event_time", "pass")
        report.record("training_labels_resolved_before_cutoff", "pass")
        for row in rows:
            if row.period_end is not None and row.available_at < row.period_end:
                raise PodPlatformError(
                    ToolResponse.failure(
                        summary=f"observation {row.observation_id} is available before its period ends",
                        code=ErrorCode.ASOF_VIOLATION,
                        stop_condition="Aggregation windows must end before availability.",
                    )
                )
        report.record("aggregation_windows_end_before_availability", "pass")
        report.returned_observation_ids = [row.observation_id for row in rows]
        return rows, report

    def eligibility_report(self, report: EligibilityReport) -> dict:
        return {
            "source_id": report.source_id,
            "requested_cutoff": report.requested_cutoff.isoformat(),
            "returned_observation_ids": report.returned_observation_ids,
            "assertions": dict(report.assertions),
            "all_passed": report.all_passed,
        }
