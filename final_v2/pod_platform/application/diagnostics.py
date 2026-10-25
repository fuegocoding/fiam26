"""Research failure taxonomy and feedback queue (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 12.2-12.7, 23.4).

Diagnoses are *hypotheses about why* a measured result failed, triggered only by
supplied measured quantities; missing inputs yield an explicit "unclassified"
diagnosis, never an invented cause. Diagnoses and queue items persist as private
notes in the owning pod's append-only ``PodMemory`` (the same store the queue
worker and MCP use), so a later agent can retrieve prior failures and not
rediscover a failed line under different wording. No model is called here.
"""

from __future__ import annotations

import datetime as dt
import enum
import math
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import Field, field_validator

from ..contracts.base import StrictModel, require_aware
from ..research.memory import MemoryEntry, PodMemory
from .common import digest, now_utc


class FailureKind(str, enum.Enum):
    TARGET_FORECAST_WRONG = "target_forecast_wrong"
    TARGET_RIGHT_RETURNS_ABSENT = "target_right_returns_absent"
    GROSS_WORKS_NET_FAILS = "gross_works_net_fails"
    STANDALONE_ADDS_LITTLE = "standalone_adds_little"
    COVERAGE_DEPENDENT = "coverage_dependent"
    PNL_BEFORE_PERMISSIBLE_ENTRY = "pnl_before_permissible_entry"
    SHARED_STRESS_LOSS = "shared_stress_loss"
    UNCLASSIFIED = "unclassified"


# spec 12.2 table: observed problem -> research question
RESEARCH_QUESTIONS: dict[FailureKind, str] = {
    FailureKind.TARGET_FORECAST_WRONG: "Is a measurement missing, is the relationship wrong, or is the method inadequate?",
    FailureKind.TARGET_RIGHT_RETURNS_ABSENT: "Is the forecast already priced, is the horizon wrong, or is the return bridge wrong?",
    FailureKind.GROSS_WORKS_NET_FAILS: "Is the failure turnover, timing, capacity, borrow, or the cost model?",
    FailureKind.STANDALONE_ADDS_LITTLE: "Is the information duplicated, or does the construction conflict with the existing book?",
    FailureKind.COVERAGE_DEPENDENT: "Is it mapping, selection, survivorship, or composition drift?",
    FailureKind.PNL_BEFORE_PERMISSIBLE_ENTRY: "Is it a timing defect, or a non-executable effect?",
    FailureKind.SHARED_STRESS_LOSS: "Is there hidden thematic exposure or crowding?",
    FailureKind.UNCLASSIFIED: "Which measured inputs are missing to classify the failure?",
}

SMALLEST_EXPERIMENT: dict[FailureKind, str] = {
    FailureKind.TARGET_FORECAST_WRONG: "Compare the naive baseline, each component alone and the combination on matured design labels.",
    FailureKind.TARGET_RIGHT_RETURNS_ABSENT: "Re-run the return bridge at two declared horizons and compare with announcement reaction alone.",
    FailureKind.GROSS_WORKS_NET_FAILS: "Re-run the same desired trades under the fixed cost and borrow sensitivity set; report turnover per name.",
    FailureKind.STANDALONE_ADDS_LITTLE: "Compute the incremental contribution versus the admitted baseline with the same construction and costs.",
    FailureKind.COVERAGE_DEPENDENT: "Split by covered/uncovered names with matched controls; test coverage alone as a predictor.",
    FailureKind.PNL_BEFORE_PERMISSIBLE_ENTRY: "Shift entry to the earliest permissible close (t+1) and recompute the payoff.",
    FailureKind.SHARED_STRESS_LOSS: "Decompose the stress loss by mechanism and source family; test the largest shared theme.",
    FailureKind.UNCLASSIFIED: "Measure the missing inputs listed in the diagnosis before proposing a redesign.",
}

ORIGINS = ("error", "paper", "data_discovery", "method", "team_decision")


class Diagnosis(StrictModel):
    kind: FailureKind
    research_question: str = Field(min_length=1)
    evidence: dict[str, float] = Field(default_factory=dict)
    missing_inputs: list[str] = Field(default_factory=list)
    is_hypothesis: bool = (
        True  # spec 12.2: diagnostic hypotheses, not automatic conclusions
    )
    note: str = "Diagnostic hypothesis from measured inputs; not a conclusion and not an automatic retrain trigger."

    @field_validator("evidence")
    @classmethod
    def _finite(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(v) for v in value.values()):
            raise ValueError("diagnostic evidence must be finite")
        return value


class QueueItem(StrictModel):
    """Research queue item (spec 12.3); every field is explicit."""

    item_id: str = Field(min_length=1)
    origin: str
    owning_strategy: str = Field(min_length=1)
    question: str = Field(min_length=1)
    existing_evidence: list[str] = Field(default_factory=list)
    smallest_discriminating_experiment: str = Field(min_length=1)
    required_sources_methods: list[str] = Field(default_factory=list)
    expected_resource_cost: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    decision_deadline: dt.datetime
    if_success: str = Field(min_length=1)
    if_failure: str = Field(min_length=1)
    diagnosis_kind: Optional[FailureKind] = None

    @field_validator("decision_deadline")
    @classmethod
    def _aware(cls, value: dt.datetime) -> dt.datetime:
        return require_aware(value, "decision_deadline")

    @field_validator("origin")
    @classmethod
    def _origin(cls, value: str) -> str:
        if value not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}")
        return value


def _num(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def classify_failure(
    metrics: Mapping[str, Any],
    *,
    coverage_gap_fraction: float = 0.5,
    pre_entry_share_limit: float = 0.25,
) -> list[Diagnosis]:
    """Apply the spec 12.2 taxonomy to *measured* inputs only.

    Recognised keys: ``target_ic_improvement``, ``bridge_ic``, ``gross_return``, ``net_return``,
    ``standalone_ic``, ``incremental_ic``, ``ic_covered``, ``ic_uncovered``, ``pre_entry_pnl_share``,
    ``stress_loss``, ``stress_limit``. Each rule fires only when every input it needs is present.
    """
    out: list[Diagnosis] = []
    seen_inputs: set[str] = set()

    def need(*keys: str) -> Optional[dict[str, float]]:
        values = {k: _num(metrics, k) for k in keys}
        seen_inputs.update(k for k, v in values.items() if v is not None)
        if any(v is None for v in values.values()):
            return None
        return {k: float(v) for k, v in values.items()}  # type: ignore[arg-type]

    def add(kind: FailureKind, evidence: dict[str, float]) -> None:
        out.append(
            Diagnosis(
                kind=kind,
                research_question=RESEARCH_QUESTIONS[kind],
                evidence=evidence,
            )
        )

    got = need("target_ic_improvement")
    if got is not None and got["target_ic_improvement"] <= 0:
        add(FailureKind.TARGET_FORECAST_WRONG, got)
    got_target = need("target_ic_improvement", "bridge_ic")
    if (
        got_target is not None
        and got_target["target_ic_improvement"] > 0
        and got_target["bridge_ic"] <= 0
    ):
        add(FailureKind.TARGET_RIGHT_RETURNS_ABSENT, got_target)
    got = need("gross_return", "net_return")
    if got is not None and got["gross_return"] > 0 and got["net_return"] <= 0:
        add(FailureKind.GROSS_WORKS_NET_FAILS, got)
    got = need("standalone_ic", "incremental_ic")
    if (
        got is not None
        and got["standalone_ic"] > 0
        and got["incremental_ic"] <= 0
    ):
        add(FailureKind.STANDALONE_ADDS_LITTLE, got)
    got = need("ic_covered", "ic_uncovered")
    if got is not None and abs(
        got["ic_covered"] - got["ic_uncovered"]
    ) >= coverage_gap_fraction * abs(got["ic_covered"]):
        add(FailureKind.COVERAGE_DEPENDENT, got)
    got = need("pre_entry_pnl_share")
    if got is not None and got["pre_entry_pnl_share"] >= pre_entry_share_limit:
        add(FailureKind.PNL_BEFORE_PERMISSIBLE_ENTRY, got)
    got = need("stress_loss", "stress_limit")
    if got is not None and got["stress_loss"] > got["stress_limit"]:
        add(FailureKind.SHARED_STRESS_LOSS, got)
    if not out:
        known = {
            "target_ic_improvement",
            "bridge_ic",
            "gross_return",
            "net_return",
            "standalone_ic",
            "incremental_ic",
            "ic_covered",
            "ic_uncovered",
            "pre_entry_pnl_share",
            "stress_loss",
            "stress_limit",
        }
        missing = sorted(known - seen_inputs)
        out.append(
            Diagnosis(
                kind=FailureKind.UNCLASSIFIED,
                research_question=RESEARCH_QUESTIONS[FailureKind.UNCLASSIFIED],
                missing_inputs=missing,
                note="No taxonomy rule fired on the supplied measured inputs; this is not evidence that the result is sound.",
            )
        )
    return out


class ResearchDiagnostics:
    """Typed facade over a pod's append-only memory (the same store the queue worker uses)."""

    def __init__(
        self, memory: PodMemory, clock: Callable[[], dt.datetime] = now_utc
    ):
        self.memory = memory
        self.clock = clock

    @staticmethod
    def classify(metrics: Mapping[str, Any], **kwargs: Any) -> list[Diagnosis]:
        return classify_failure(metrics, **kwargs)

    def record(
        self,
        subject_id: str,
        diagnoses: Sequence[Diagnosis],
        *,
        attempted: str,
        outcome: str,
        retry_justified: bool,
        retry_differs_by: str = "",
    ) -> MemoryEntry:
        """Persist what was tested, what happened, what remains unresolved (spec 12.5)."""
        content = {
            "type": "failure_diagnostic",
            "subject_id": subject_id,
            "attempted": attempted,
            "outcome": outcome,
            "kinds": [d.kind.value for d in diagnoses],
            "diagnoses": [d.model_dump(mode="json") for d in diagnoses],
            "unresolved": sorted(
                {m for d in diagnoses for m in d.missing_inputs}
            ),
            "retry_justified": bool(retry_justified),
            "retry_differs_by": retry_differs_by,
            "evaluated": False,
        }
        return self.memory.append(
            "note",
            content,
            idempotency_key=f"diag:{subject_id}:{digest(content)[:16]}",
        )

    def queue_item_for(
        self,
        diagnosis: Diagnosis,
        *,
        subject_id: str,
        decision_deadline: dt.datetime,
        existing_evidence: Sequence[str] = (),
        origin: str = "error",
        required_sources_methods: Sequence[str] = (),
        dependencies: Sequence[str] = (),
        expected_resource_cost: Optional[Mapping[str, str]] = None,
    ) -> QueueItem:
        return QueueItem(
            item_id=f"rq_{digest({'s': subject_id, 'k': diagnosis.kind.value, 'e': diagnosis.evidence})[:16]}",
            origin=origin,
            owning_strategy=self.memory.namespace,
            question=f"{subject_id}: {diagnosis.research_question}",
            existing_evidence=list(existing_evidence),
            smallest_discriminating_experiment=SMALLEST_EXPERIMENT[
                diagnosis.kind
            ],
            required_sources_methods=list(required_sources_methods),
            expected_resource_cost=dict(expected_resource_cost or {}),
            dependencies=list(dependencies),
            decision_deadline=decision_deadline,
            if_success="Register the amended specification as a new variant linked to its parent and re-run the registered comparison.",
            if_failure="Retain the record, pause this line under the stopping rules and do not retry under different wording.",
            diagnosis_kind=diagnosis.kind,
        )

    def enqueue(self, item: QueueItem) -> MemoryEntry:
        content = {
            "type": "research_queue_item",
            "state": "open",
            "item": item.model_dump(mode="json"),
            "evaluated": False,
        }
        return self.memory.append(
            "note", content, idempotency_key=f"rq:{item.item_id}"
        )

    def pause_line(self, line_id: str, reason: str) -> MemoryEntry:
        """Stopping rule (spec 12.7): keep the record so the line is not rediscovered."""
        content = {
            "type": "research_line_paused",
            "line_id": line_id,
            "reason": reason,
            "evaluated": False,
        }
        return self.memory.append(
            "note", content, idempotency_key=f"pause:{line_id}"
        )

    def list_queue(self, *, include_paused: bool = False) -> list[QueueItem]:
        paused = {
            e.content["line_id"]
            for e in self.memory.list("note", limit=1000)
            if e.content.get("type") == "research_line_paused"
        }
        items: list[QueueItem] = []
        for entry in self.memory.list("note", limit=1000):
            if entry.content.get("type") != "research_queue_item":
                continue
            item = QueueItem.model_validate(entry.content["item"])
            if not include_paused and any(
                item.item_id == p or item.question.startswith(f"{p}:")
                for p in paused
            ):
                continue
            items.append(item)
        return sorted(items, key=lambda i: (i.decision_deadline, i.item_id))

    def prior_attempts(
        self, subject_id: str, kind: Optional[FailureKind] = None
    ) -> list[dict[str, Any]]:
        """Own-namespace retrieval of earlier failures for a subject (spec 12.5 redundancy check)."""
        out: list[dict[str, Any]] = []
        for entry in self.memory.list("note", limit=1000):
            c = entry.content
            if (
                c.get("type") != "failure_diagnostic"
                or c.get("subject_id") != subject_id
            ):
                continue
            if kind is not None and kind.value not in c.get("kinds", []):
                continue
            out.append(c)
        return out
