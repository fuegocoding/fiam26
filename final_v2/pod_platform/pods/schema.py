"""Evidence normalisation and the structured research-proposal schema
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 12.3, 20.2, 20.5).

Evidence cards are read as *generic dicts* (no dependency on the papers
package). Their text is untrusted data: it is truncated, scanned for
instruction-like content (flagged, never obeyed) and handed to the model only
inside a JSON field named ``untrusted_evidence``.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any, Iterable, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from ..agents.provider import ResearchMode
from ..contracts.base import StrictModel, canonical_digest, require_aware

MAX_VARIANTS = 8
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:\-]{0,127}$"
_ID_RE = re.compile(ID_PATTERN)
_IDENT = r"^[a-z][a-z0-9_]{0,63}$"

ID_KEYS = ("evidence_id", "card_id", "paper_id", "source_card_id", "source_id", "dataset_id", "id")
TITLE_KEYS = ("title", "name")
TEXT_KEYS = ("summary", "abstract", "description", "key_claims", "claims", "mechanism", "notes", "limitations")
DATE_KEYS = ("available_at", "first_public_at", "first_public_date", "published_at", "published", "as_of", "created_at")
# A later revision of the text is not the text that was public on the original date:
# the card is only available once its latest revision is.
REVISION_KEYS = ("updated_at", "updated", "revised_at", "last_revised_at", "version_date")

_INJECTION_PATTERNS = {
    "ignore_instructions": re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)", re.I),
    "disregard": re.compile(r"disregard\s+(the\s+|all\s+)?(system|previous|instructions|rules)", re.I),
    "role_override": re.compile(r"(you are now|act as|new instructions|system prompt|developer message)", re.I),
    "execute_code": re.compile(r"(execute|run|eval)\s+(the\s+following\s+)?(shell|bash|python|command|code|script)", re.I),
    "chat_markup": re.compile(r"(<\|im_start\|>|<\|system\|>|\[INST\]|###\s*system)", re.I),
    "citation_forgery": re.compile(r"(cite|reference)\s+(only\s+)?[A-Za-z0-9_\-]+\s+(as|instead)", re.I),
    "admission_attempt": re.compile(r"(admit|approve|deploy)\s+(this|the)?\s*(strategy|model|signal|portfolio)", re.I),
}


class EvidenceError(ValueError):
    """Evidence failed provenance or temporal checks (fail closed)."""


class CitationError(ValueError):
    """Proposal cites evidence or executors that were not provided."""


class EvidenceItem(StrictModel):
    evidence_id: str = Field(pattern=ID_PATTERN)
    kind: Literal["paper_card", "dataset_card", "source_card"]
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(max_length=20000)
    available_at: Optional[dt.datetime] = None
    availability_verified: bool = False
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    injection_flags: list[str] = Field(default_factory=list)

    @field_validator("available_at")
    @classmethod
    def _aware(cls, value: Optional[dt.datetime]) -> Optional[dt.datetime]:
        return None if value is None else require_aware(value, "available_at")

    def prompt_view(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "available_at": self.available_at.isoformat() if self.available_at else None,
            "scanner_flags": list(self.injection_flags),
        }


def _first(card: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if card.get(key) not in (None, "", [], {}):
            return card[key]
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_stringify(v)}" for k, v in sorted(value.items()))
    return str(value)


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str, limit: int) -> str:
    return _CONTROL.sub(" ", text).strip()[:limit]


def _parse_available(value: Any, card_id: str) -> dt.datetime:
    if isinstance(value, dt.datetime):
        parsed = value
    elif isinstance(value, dt.date):
        parsed = dt.datetime.combine(value, dt.time(23, 59, 59, 999999), tzinfo=dt.timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                # date-only: available no earlier than the end of that UTC day
                parsed = dt.datetime.combine(
                    dt.date.fromisoformat(text), dt.time(23, 59, 59, 999999), tzinfo=dt.timezone.utc
                )
            else:
                parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            raise EvidenceError(f"evidence {card_id}: unparseable availability timestamp") from None
    else:
        raise EvidenceError(f"evidence {card_id}: availability must be a timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError(f"evidence {card_id}: availability timestamp must be timezone-aware")
    return parsed.astimezone(dt.timezone.utc)


def scan_injection(text: str) -> list[str]:
    return sorted(name for name, pattern in _INJECTION_PATTERNS.items() if pattern.search(text))


def normalise_evidence(
    cards: Iterable[dict[str, Any]],
    kind: Literal["paper_card", "dataset_card", "source_card"],
    *,
    max_text_chars: int = 3000,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            raise EvidenceError(f"{kind} #{index} is not an object")
        raw_id = _first(card, ID_KEYS)
        if raw_id is None or not _ID_RE.match(str(raw_id)):
            raise EvidenceError(f"{kind} #{index} has no valid stable identifier")
        card_id = str(raw_id)
        title = _clean(_stringify(_first(card, TITLE_KEYS) or card_id), 300) or card_id
        parts = [f"{key}: {_stringify(card[key])}" for key in TEXT_KEYS if card.get(key) not in (None, "", [], {})]
        full_text = "\n".join(parts)
        raw_date = _first(card, DATE_KEYS)
        available = _parse_available(raw_date, card_id) if raw_date is not None else None
        for key in REVISION_KEYS:
            revised = card.get(key)
            if revised not in (None, "", [], {}):
                parsed = _parse_available(revised, card_id)
                available = parsed if available is None else max(available, parsed)
        flags = scan_injection(title + "\n" + full_text)
        items.append(
            EvidenceItem(
                evidence_id=card_id,
                kind=kind,
                title=title,
                text=_clean(full_text, max_text_chars),
                available_at=available,
                availability_verified=available is not None,
                content_hash=canonical_digest({"kind": kind, "card": card}),
                injection_flags=flags,
            )
        )
    return items


def filter_eligible(
    items: list[EvidenceItem],
    *,
    mode: ResearchMode,
    research_asof: dt.datetime,
    now: dt.datetime,
    max_items: int = 20,
) -> list[EvidenceItem]:
    """Reject (not silently drop) evidence that violates the time boundary."""
    asof = require_aware(research_asof, "research_asof")
    now = require_aware(now, "now")
    seen: set[str] = set()
    for item in items:
        if item.evidence_id in seen:
            raise EvidenceError(f"duplicate evidence id {item.evidence_id}")
        seen.add(item.evidence_id)
    if not items:
        raise EvidenceError("at least one evidence card is required")
    if len(items) > max_items:
        raise EvidenceError(f"at most {max_items} evidence cards per proposal")
    bad: list[str] = []
    for item in items:
        if mode == ResearchMode.HISTORICAL:
            if item.available_at is None or item.available_at > asof:
                bad.append(item.evidence_id)
        elif item.available_at is not None and item.available_at > now:
            bad.append(item.evidence_id)
    if bad:
        raise EvidenceError(f"evidence not available at the research date (or undated in historical mode): {bad}")
    return items


# --------------------------------------------------------------------------
# Proposal schema

Scalar = Union[str, int, float, bool]


class FeatureDefinition(StrictModel):
    name: str = Field(pattern=_IDENT)
    definition: str = Field(min_length=10, max_length=600)
    timing: str = Field(min_length=5, max_length=400)  # when each input becomes known vs signal date
    source_ids: list[str] = Field(min_length=1, max_length=6)


class ExperimentVariant(StrictModel):
    name: str = Field(pattern=_IDENT)
    params: dict[str, Scalar] = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def _params(cls, value: dict[str, Scalar]) -> dict[str, Scalar]:
        if len(value) > 12:
            raise ValueError("at most 12 parameters per variant")
        for key, item in value.items():
            if not re.match(_IDENT, key):
                raise ValueError(f"invalid parameter name {key!r}")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("parameters must be finite")
            if isinstance(item, str) and len(item) > 100:
                raise ValueError("string parameter too long")
        return value


class ExperimentPlan(StrictModel):
    executor_id: str = Field(pattern=_IDENT)  # a registered executor, never code
    primary_metric: str = Field(min_length=2, max_length=100)
    data_partition: Literal["design"]  # proposals never request check/final data
    variants: list[ExperimentVariant] = Field(min_length=1, max_length=MAX_VARIANTS)
    description: str = Field(min_length=10, max_length=800)

    @model_validator(mode="after")
    def _unique(self) -> "ExperimentPlan":
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        return self


class ResearchProposal(StrictModel):
    title: str = Field(min_length=5, max_length=200)
    mechanism: str = Field(min_length=20, max_length=1500)
    falsifiable_target: str = Field(min_length=20, max_length=800)
    baseline: str = Field(min_length=5, max_length=600)
    features: list[FeatureDefinition] = Field(min_length=1, max_length=12)
    evidence_refs: list[str] = Field(min_length=1, max_length=12)
    experiment_plan: ExperimentPlan
    competing_explanation: str = Field(min_length=10, max_length=800)
    limitations: list[str] = Field(min_length=1, max_length=10)
    missing_information: list[str] = Field(default_factory=list, max_length=10)
    evidence_quality: Literal["weak", "moderate", "strong"]

    @field_validator("limitations")
    @classmethod
    def _limits(cls, value: list[str]) -> list[str]:
        if any(len(v.strip()) < 5 or len(v) > 600 for v in value):
            raise ValueError("each limitation must be 5-600 characters")
        return value

    @model_validator(mode="after")
    def _unique(self) -> "ResearchProposal":
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("duplicate evidence_refs")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("duplicate feature names")
        return self


def validate_citations(
    proposal: ResearchProposal, allowed_ids: Iterable[str], allowed_executors: Iterable[str]
) -> None:
    """Every cited source id and the executor must be one we supplied."""
    allowed = set(allowed_ids)
    cited = set(proposal.evidence_refs)
    for feature in proposal.features:
        cited.update(feature.source_ids)
    unknown = sorted(cited - allowed)
    if unknown:
        raise CitationError(f"proposal cites evidence that was not provided: {unknown}")
    executors = set(allowed_executors)
    if proposal.experiment_plan.executor_id not in executors:
        raise CitationError(
            f"executor {proposal.experiment_plan.executor_id!r} is not an allowed registered executor"
        )
