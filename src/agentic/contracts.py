"""Small, dependency-free contracts shared by the agentic pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

EVENT_CATEGORIES = (
    "earnings_results",
    "guidance_update",
    "material_agreement",
    "financing_credit",
    "capital_structure_debt",
    "share_issuance_buyback",
    "dividend",
    "mna",
    "asset_disposal",
    "restructuring_impairment",
    "management_change",
    "auditor_change",
    "accounting_nonreliance",
    "litigation_regulatory",
    "delisting_compliance",
    "shareholder_vote",
    "product_operations",
    "contract_partnership",
    "investor_conference",
    "other",
    "unreadable",
)

ROLE_NAMES = (
    "event_classifier",
    "materiality_analyst",
    "devils_advocate",
    "evidence_auditor",
)

VERDICT_KEYS = frozenset(
    {
        "event_category",
        "direction",
        "materiality",
        "surprise",
        "forward_looking",
        "distress_signal",
        "cash_flow_impact",
        "confidence",
        "contradiction_score",
        "evidence_quality",
        "abstain",
        "evidence_count",
        "summary",
    }
)

SIGNED_FIELDS = ("direction", "cash_flow_impact")
UNIT_FIELDS = (
    "materiality",
    "surprise",
    "forward_looking",
    "distress_signal",
    "confidence",
    "contradiction_score",
    "evidence_quality",
)


@dataclass(frozen=True)
class RedactedFiling:
    """Local filing envelope whose prompt-facing fields are already sanitized."""

    document_id: str
    opaque_id: str
    permno: int | None
    filing_date: str
    eom: str
    text: str
    visible_metadata: Mapping[str, Any]
    redaction_version: str = "redaction-v1"

    def prompt_payload(self) -> dict[str, Any]:
        """Return only data permitted to cross the identity firewall."""
        return {
            "metadata": dict(self.visible_metadata),
            "filing_text": self.text,
        }


@dataclass(frozen=True)
class AgentVerdict:
    role: str
    event_category: str
    direction: float
    materiality: float
    surprise: float
    forward_looking: float
    distress_signal: float
    cash_flow_impact: float
    confidence: float
    contradiction_score: float
    evidence_quality: float
    abstain: int
    evidence_count: int
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "event_category": self.event_category,
            "direction": self.direction,
            "materiality": self.materiality,
            "surprise": self.surprise,
            "forward_looking": self.forward_looking,
            "distress_signal": self.distress_signal,
            "cash_flow_impact": self.cash_flow_impact,
            "confidence": self.confidence,
            "contradiction_score": self.contradiction_score,
            "evidence_quality": self.evidence_quality,
            "abstain": self.abstain,
            "evidence_count": self.evidence_count,
            "summary": self.summary,
        }


def verdict_schema_text() -> str:
    return (
        "Return one JSON object with exactly these keys: "
        "event_category, direction, materiality, surprise, forward_looking, "
        "distress_signal, cash_flow_impact, confidence, contradiction_score, "
        "evidence_quality, abstain, evidence_count, summary. "
        "direction and cash_flow_impact are in [-1.0, 1.0]; "
        "materiality, surprise, forward_looking, distress_signal, confidence, "
        "contradiction_score, and evidence_quality are in [0.0, 1.0]; "
        "abstain is 0 or 1 and evidence_count is a non-negative integer."
    )


def _decode_json(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("verdict must be a JSON object")
    return obj


def _number(obj: Mapping[str, Any], key: str, low: float, high: float) -> float:
    value = obj[key]
    if isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not low <= value <= high:
        raise ValueError(f"{key} outside [{low}, {high}]: {value}")
    return value


def validate_verdict(raw: str | Mapping[str, Any], role: str = "unknown") -> AgentVerdict:
    obj = _decode_json(raw)
    if set(obj) != VERDICT_KEYS:
        missing = sorted(VERDICT_KEYS - set(obj))
        extra = sorted(set(obj) - VERDICT_KEYS)
        raise ValueError(f"verdict keys mismatch; missing={missing}, extra={extra}")
    if role not in ROLE_NAMES and role != "unknown":
        raise ValueError(f"unknown role: {role}")
    event_category = str(obj["event_category"])
    if event_category not in EVENT_CATEGORIES:
        raise ValueError(f"unknown event category: {event_category}")
    abstain = obj["abstain"]
    if isinstance(abstain, bool) or abstain not in (0, 1):
        raise ValueError("abstain must be 0 or 1")
    evidence_count = obj["evidence_count"]
    if isinstance(evidence_count, bool) or not isinstance(evidence_count, int):
        raise ValueError("evidence_count must be a non-negative integer")
    if evidence_count < 0:
        raise ValueError("evidence_count must be a non-negative integer")
    summary = obj["summary"]
    if not isinstance(summary, str) or len(summary) > 240:
        raise ValueError("summary must be a string of at most 240 characters")
    values: dict[str, Any] = {
        "role": role,
        "event_category": event_category,
        "direction": _number(obj, "direction", -1.0, 1.0),
        "materiality": _number(obj, "materiality", 0.0, 1.0),
        "surprise": _number(obj, "surprise", 0.0, 1.0),
        "forward_looking": _number(obj, "forward_looking", 0.0, 1.0),
        "distress_signal": _number(obj, "distress_signal", 0.0, 1.0),
        "cash_flow_impact": _number(obj, "cash_flow_impact", -1.0, 1.0),
        "confidence": _number(obj, "confidence", 0.0, 1.0),
        "contradiction_score": _number(obj, "contradiction_score", 0.0, 1.0),
        "evidence_quality": _number(obj, "evidence_quality", 0.0, 1.0),
        "abstain": int(abstain),
        "evidence_count": evidence_count,
        "summary": summary,
    }
    return AgentVerdict(**values)
