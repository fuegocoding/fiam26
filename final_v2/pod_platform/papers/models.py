"""Data models, errors, identifier and time helpers for the papers service.

Time rules (fail closed):
* every datetime is timezone aware and normalised to UTC;
* provider dates that carry no time of day (``2020-05-01``) are read as the END
  of that day, the latest possible public moment (conservative for as-of);
* an ``as_of`` that is naive, unparseable, or later than the service clock raises
  ``TemporalError``; a date-only ``as_of`` is read as the START of that day.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PapersError(Exception):
    """Base class for all papers-service errors."""


class PaperValidationError(PapersError):
    pass


class TemporalError(PapersError):
    pass


class OfflineError(PapersError):
    """Network needed but the service is offline or the cache lacks the item."""


class UrlNotAllowed(PapersError):
    pass


class ProviderError(PapersError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class AuthError(ProviderError):
    pass


class RateLimitError(ProviderError):
    pass


class ResponseTooLarge(ProviderError):
    pass


class NotFound(PapersError):
    pass


class ClaimError(PapersError):
    pass


class AccessStatus(str, Enum):
    METADATA_ONLY = "metadata_only"
    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    APPENDIX = "appendix"
    CODE = "code"


class ReplicationLabel(str, Enum):
    EXACT_REPLICATION = "EXACT_REPLICATION"
    CLOSE_REPLICATION = "CLOSE_REPLICATION"
    MECHANISM_TRANSFER = "MECHANISM_TRANSFER"
    METHOD_TRANSFER = "METHOD_TRANSFER"
    INCOMPLETE = "INCOMPLETE"


CLAIM_KINDS = (
    "economic_claim", "mechanism", "universe", "period", "target", "horizon",
    "variable_definition", "timing", "statistical_design", "baseline",
    "fixed_parameter", "estimated_parameter", "replication", "contrary_evidence",
    "correction", "limitation", "other",
)

# ---------------------------------------------------------------- identifiers
_NEW_ARXIV = r"\d{4}\.\d{4,5}"
_OLD_ARXIV = r"[a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?/\d{7}"
ARXIV_ID_RE = re.compile(rf"^(?:{_NEW_ARXIV}|{_OLD_ARXIV})$")
ARXIV_VERSIONED_RE = re.compile(rf"^({_NEW_ARXIV}|{_OLD_ARXIV})v(\d{{1,3}})$")
DOI_RE = re.compile(r"^10\.\d{4,9}/[^\s]{1,200}$")
OPENALEX_RE = re.compile(r"^W\d{1,15}$")
LOCAL_RE = re.compile(r"^[0-9a-f]{16}$")


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip()
    v = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", v, flags=re.I).strip().lower()
    if not DOI_RE.match(v):
        raise PaperValidationError("invalid DOI")
    return v


def split_arxiv(value: str) -> tuple[str, Optional[int]]:
    """Return (base id, version or None); accepts ``arXiv:`` and abs URLs."""
    v = value.strip()
    v = re.sub(r"^(?:arxiv:|https?://(?:export\.)?arxiv\.org/abs/)", "", v, flags=re.I)
    m = ARXIV_VERSIONED_RE.match(v)
    if m:
        return m.group(1), int(m.group(2))
    if ARXIV_ID_RE.match(v):
        return v, None
    raise PaperValidationError("invalid arXiv identifier")


def canonical_paper_id(value: str) -> tuple[str, Optional[int]]:
    """Parse a user-supplied id into ``(paper_id, version)``.

    Forms: ``arxiv:<id>[vN]``, ``doi:<doi>``, ``openalex:W123``, ``local:<hex16>``.
    A bare arXiv id or DOI is also accepted. Result ids are never used as paths.
    """
    v = (value or "").strip()
    if not v or len(v) > 300:
        raise PaperValidationError("empty or oversized paper id")
    low = v.lower()
    if low.startswith("openalex:"):
        w = v.split(":", 1)[1].upper()
        if not OPENALEX_RE.match(w):
            raise PaperValidationError("invalid OpenAlex id")
        return f"openalex:{w}", None
    if low.startswith("local:"):
        h = low.split(":", 1)[1]
        if not LOCAL_RE.match(h):
            raise PaperValidationError("invalid local id")
        return f"local:{h}", None
    if low.startswith("doi:") or low.startswith("10.") or "doi.org/" in low:
        return f"doi:{normalize_doi(v)}", None
    base, ver = split_arxiv(v)
    return f"arxiv:{base}", ver


# ----------------------------------------------------------------------- time
def ensure_aware(dt: Any, name: str = "datetime") -> datetime:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise TemporalError(f"{name} must be a timezone-aware datetime")
    return dt.astimezone(timezone.utc)


def parse_instant(value: Any, *, name: str = "date") -> datetime:
    """Parse a provider timestamp. Date-only => end of that UTC day."""
    if isinstance(value, datetime):
        return ensure_aware(value, name)
    if not isinstance(value, str) or not value.strip():
        raise TemporalError(f"{name} missing")
    s = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            d = date.fromisoformat(s)
        except ValueError as exc:
            raise TemporalError(f"{name} invalid: {s!r}") from exc
        return datetime.combine(d, time.max, tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TemporalError(f"{name} invalid: {s!r}") from exc
    if dt.tzinfo is None:
        raise TemporalError(f"{name} has no timezone: {s!r}")
    return dt.astimezone(timezone.utc)


def parse_as_of(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        try:
            d = date.fromisoformat(value.strip())
        except ValueError as exc:
            raise TemporalError(f"as_of invalid: {value!r}") from exc
        return datetime.combine(d, time.min, tzinfo=timezone.utc)
    return parse_instant(value, name="as_of")


# -------------------------------------------------------------------- hashing
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


# ---------------------------------------------------------------------- cards
class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Claim(_Model):
    """A statement extracted from one source text, tied to an exact quote.

    ``quote == source_text[char_start:char_end]`` of the text whose hash is
    ``source_hash``; ``page``/``section`` are derived from that text, never typed in.
    """

    claim_id: str
    kind: str
    text: str = Field(max_length=2000)
    quote: str = Field(max_length=2000)
    source: str  # "abstract" | "full_text"
    source_hash: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    page: Optional[int] = None
    section: Optional[str] = None
    extraction: str = "manual"  # "manual" | "heuristic" (needs human review)

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        if v not in CLAIM_KINDS:
            raise ValueError(f"unknown claim kind {v!r}")
        return v

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        if v not in ("abstract", "full_text"):
            raise ValueError("source must be 'abstract' or 'full_text'")
        return v


class PaperCard(_Model):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    version: Optional[int] = None  # arXiv version number of this card, if known
    latest_version: Optional[int] = None
    version_key: str  # "v3", "pub" (published version of record) or "unversioned"
    openalex_id: Optional[str] = None
    source: str  # provider name: arxiv | openalex | local
    first_public_date: Optional[datetime] = None
    version_date: Optional[datetime] = None  # None => unknown => excluded as-of
    abstract: Optional[str] = None
    abstract_hash: Optional[str] = None
    access_status: AccessStatus = AccessStatus.METADATA_ONLY
    urls: dict[str, str] = Field(default_factory=dict)
    content_hash: Optional[str] = None  # sha256 of stored text
    original_hash: Optional[str] = None  # sha256 of raw bytes/text as received
    content_chars: int = 0
    content_truncated: bool = False
    text_status: str = "none"  # none | abstract_only | full_text | incomplete:<reason>
    claims: list[Claim] = Field(default_factory=list)
    retrieved_at: datetime
    card_hash: Optional[str] = None

    @field_validator("first_public_date", "version_date", "retrieved_at")
    @classmethod
    def _aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        return None if v is None else ensure_aware(v)

    def compute_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"card_hash"})
        return sha256_text(canonical_json(data))

    def sealed(self) -> "PaperCard":
        return self.model_copy(update={"card_hash": self.compute_hash()})


def utcnow_check(dt: datetime) -> datetime:
    return ensure_aware(dt, "clock()")


def days(n: int) -> timedelta:
    return timedelta(days=n)
