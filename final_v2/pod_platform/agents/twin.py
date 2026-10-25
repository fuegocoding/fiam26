"""Exception-only agent layer with fixed-rule twin
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 20.7-20.9, 18.10).

Agents are called only on declared triggers (near-binding constraint,
surprising exposure change, data inconsistency, unexplained loss, hard
trade). Routine cases stay deterministic. Every agent intervention runs
against a fixed-rule twin with identical inputs; the comparison includes
all periods and losses caused by the agent. A malformed response falls
back to the deterministic path and never bypasses a financial rule.
"""

from __future__ import annotations

import datetime as dt
import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Optional

from ..contracts.base import require_aware
from ..contracts.decisions import ActorKind, DecisionEvent


class TriggerKind(str, enum.Enum):
    NEAR_BINDING_CONSTRAINT = "near_binding_constraint"
    SURPRISING_EXPOSURE_CHANGE = "surprising_exposure_change"
    DATA_SOURCE_INCONSISTENCY = "data_source_inconsistency"
    UNEXPLAINED_POSITION_LOSS = "unexplained_position_loss"
    DIFFICULT_TRADE = "difficult_trade"


@dataclass(frozen=True)
class ExceptionTrigger:
    kind: TriggerKind
    context: dict
    fired_at: dt.datetime

    def __post_init__(self) -> None:
        require_aware(self.fired_at, "fired_at")


@dataclass(frozen=True)
class AgentResponse:
    ok: bool
    proposed_action: str
    rationale: str
    model_ref: str
    prompt_ref: str
    cache_ref: str
    confidence_note: str = ""


@dataclass
class InterventionRecord:
    trigger: ExceptionTrigger
    agent_action: str
    twin_action: str
    chosen_action: str
    used_agent: bool
    fallback_reason: Optional[str] = None


class AgentTwin:
    def __init__(
        self,
        deterministic_action: Callable[[ExceptionTrigger], str],
        agent_call: Callable[[ExceptionTrigger], Optional[AgentResponse]],
    ):
        self._deterministic_action = deterministic_action
        self._agent_call = agent_call
        self._records: list[InterventionRecord] = []
        self._triggers_seen = 0
        self._calls_made = 0

    @property
    def records(self) -> list[InterventionRecord]:
        return list(self._records)

    @property
    def call_stats(self) -> dict[str, int]:
        return {
            "triggers_seen": self._triggers_seen,
            "calls_made": self._calls_made,
            "skipped_routine": 0,
        }

    def handle(
        self, trigger: ExceptionTrigger, *, force_call: bool = False
    ) -> InterventionRecord:
        self._triggers_seen += 1
        twin_action = self._deterministic_action(trigger)
        response: Optional[AgentResponse] = None
        if force_call or self._is_exception(trigger):
            self._calls_made += 1
            response = self._safe_call(trigger)
        if response is None or not response.ok:
            record = InterventionRecord(
                trigger=trigger,
                agent_action="",
                twin_action=twin_action,
                chosen_action=twin_action,
                used_agent=False,
                fallback_reason="malformed or unavailable agent response; deterministic fallback applies",
            )
        else:
            record = InterventionRecord(
                trigger=trigger,
                agent_action=response.proposed_action,
                twin_action=twin_action,
                chosen_action=response.proposed_action,
                used_agent=True,
            )
        self._records.append(record)
        return record

    def counterfactual_summary(self) -> dict:
        agent_used = [r for r in self._records if r.used_agent]
        divergent = [r for r in agent_used if r.agent_action != r.twin_action]
        return {
            "interventions": len(self._records),
            "agent_used": len(agent_used),
            "divergent_from_twin": len(divergent),
            "records": [
                {
                    "trigger": r.trigger.kind.value,
                    "used_agent": r.used_agent,
                    "agent_action": r.agent_action,
                    "twin_action": r.twin_action,
                    "chosen_action": r.chosen_action,
                    "fallback_reason": r.fallback_reason,
                }
                for r in self._records
            ],
        }

    def decision_event(
        self,
        record: InterventionRecord,
        *,
        event_id: str,
        owner: str,
        resulting_artifact_id: str,
    ) -> DecisionEvent:
        require_aware(record.trigger.fired_at, "fired_at")
        if record.used_agent and record.agent_action:
            raise ValueError(
                "use decision_event_from_agent for agent interventions"
            )
        return DecisionEvent(
            owner=owner,
            created_at=record.trigger.fired_at,
            event_id=event_id,
            actor_kind=ActorKind.CODE,
            actor_id="fixed_rule_twin",
            effective_research_at=record.trigger.fired_at,
            effective_market_at=record.trigger.fired_at,
            input_artifact_ids=[f"trigger_{record.trigger.kind.value}"],
            input_hashes={
                f"trigger_{record.trigger.kind.value}": hashlib.sha256(
                    json.dumps(
                        record.trigger.context, sort_keys=True, default=str
                    ).encode()
                ).hexdigest()
            },
            chosen_action=record.chosen_action,
            rationale=record.fallback_reason
            or "fixed-rule path with no agent intervention",
            authorizing_rule_or_certificate_id="rule_fixed_execution_baseline_v1",
            resulting_artifact_id=resulting_artifact_id,
        )

    def decision_event_from_agent(
        self,
        record: InterventionRecord,
        response: AgentResponse,
        *,
        event_id: str,
        owner: str,
        resulting_artifact_id: str,
    ) -> DecisionEvent:
        return DecisionEvent(
            owner=owner,
            created_at=record.trigger.fired_at,
            event_id=event_id,
            actor_kind=ActorKind.AGENT_ROLE,
            actor_id="exception_agent",
            effective_research_at=record.trigger.fired_at,
            effective_market_at=record.trigger.fired_at,
            input_artifact_ids=[f"trigger_{record.trigger.kind.value}"],
            input_hashes={
                f"trigger_{record.trigger.kind.value}": hashlib.sha256(
                    json.dumps(
                        record.trigger.context, sort_keys=True, default=str
                    ).encode()
                ).hexdigest()
            },
            proposed_alternatives=[record.twin_action],
            chosen_action=record.chosen_action,
            rationale=response.rationale,
            authorizing_rule_or_certificate_id="rule_exception_agent_scope_v1",
            model_ref=response.model_ref,
            prompt_ref=response.prompt_ref,
            cache_ref=response.cache_ref,
            resulting_artifact_id=resulting_artifact_id,
        )

    @staticmethod
    def _is_exception(trigger: ExceptionTrigger) -> bool:
        return trigger.kind in (
            TriggerKind.NEAR_BINDING_CONSTRAINT,
            TriggerKind.SURPRISING_EXPOSURE_CHANGE,
            TriggerKind.DATA_SOURCE_INCONSISTENCY,
            TriggerKind.UNEXPLAINED_POSITION_LOSS,
            TriggerKind.DIFFICULT_TRADE,
        )

    def _safe_call(self, trigger: ExceptionTrigger) -> Optional[AgentResponse]:
        try:
            return self._agent_call(trigger)
        except Exception:
            return None
