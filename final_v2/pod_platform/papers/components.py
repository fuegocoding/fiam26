"""Cached, quoted paper components for bounded research, without replication claims.

Quote selection declares relevance; this service verifies provenance, not an NLP
keyword guess about meaning. It never approves a source, dataset, method or model.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import ConfigDict, Field, ValidationError, field_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..pods.charters import pod_spec
from ..pods.schema import EvidenceItem, scan_injection
from .models import (
    CLAIM_KINDS,
    Claim,
    PapersError,
    canonical_paper_id,
    sha256_text,
)


class ComponentRejected(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


class ComponentRequest(StrictModel):
    namespace: str
    component_kind: Literal["mechanism", "feature", "method", "diagnostic"]
    component_name: str = Field(min_length=1, max_length=300)
    paper_id: str
    version_key: str = Field(min_length=1, max_length=32)
    claim_ids: list[str] = Field(min_length=1, max_length=8)
    source: Literal["abstract", "full_text"]
    as_of: dt.datetime | None = None
    unavailable_controls: list[str] = Field(default_factory=list, max_length=8)
    label: Literal["component_only"] = "component_only"

    @field_validator("as_of")
    @classmethod
    def _aware(cls, value):
        return None if value is None else require_aware(value, "as_of")

    @field_validator("paper_id")
    @classmethod
    def _id(cls, value):
        if not value.startswith(("arxiv:", "doi:", "openalex:", "local:")):
            raise ValueError("select a cached paper id, never a file or URL")
        canonical_paper_id(value)
        return value

    @field_validator("unavailable_controls")
    @classmethod
    def _controls(cls, values):
        if any(not 1 <= len(v.strip()) <= 600 for v in values):
            raise ValueError("control limitations must be 1..600 characters")
        return values


class ComponentEvidence(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    component_id: str
    namespace: str
    component_kind: Literal["mechanism", "feature", "method", "diagnostic"]
    component_name: str
    paper_id: str
    version_key: str
    source: Literal["abstract", "full_text"]
    source_hash: str
    version_date: dt.datetime
    retrieved_at: dt.datetime
    as_of: dt.datetime
    claims: tuple[Claim, ...]
    unavailable_controls: tuple[str, ...]
    limitations: tuple[str, ...]
    label: Literal["component_only"] = "component_only"
    replication_status: Literal["not_replication"] = "not_replication"
    evidence: EvidenceItem
    content_hash: str


class ComponentEvidenceService:
    def __init__(self, *, papers, namespace, clock):
        if pod_spec(namespace).namespace != namespace:
            raise ComponentRejected(
                "namespace_mismatch", "canonical pod namespace required"
            )
        self.papers, self.namespace, self.clock = papers, namespace, clock

    def quote(self, *, paper_id, version_key, source, kind, quote):
        """Record one exact cached excerpt; this is a claim selection, not approval."""
        pid, version = canonical_paper_id(paper_id)
        card = self.papers.store.get_row(
            self.papers.store.resolve(pid) or pid, version_key
        )
        if card is None or (version is not None and card.version != version):
            raise ComponentRejected(
                "version_mismatch", "select an existing cached version"
            )
        if (
            source not in ("abstract", "full_text")
            or kind not in CLAIM_KINDS
            or not 1 <= len(quote) <= 2000
        ):
            raise ComponentRejected(
                "invalid_quote",
                "known source, claim kind and 1..2000 character quote required",
            )
        body = self.papers.text_of(card, source)
        start = body.find(quote) if body else -1
        if start < 0:
            raise ComponentRejected(
                "ungrounded_claims",
                "quote is absent from the selected cached source",
            )
        claim = self.papers._make_claim(
            card, kind, body, start, start + len(quote), source, "manual"
        )
        # Component resolution additionally checks source hashes, dates and receipt.
        selected = card.model_copy(
            update={
                "claims": [
                    c for c in card.claims if c.claim_id != claim.claim_id
                ]
                + [claim]
            }
        )
        if self.papers.verify_claims(selected):
            raise ComponentRejected(
                "ungrounded_claims", "cached claim verification failed"
            )
        return self.papers.store.save_card(selected)

    def resolve(self, request: ComponentRequest | dict) -> ComponentEvidence:
        try:
            rv = ComponentRequest.model_validate(request)
        except (ValidationError, PapersError) as exc:
            raise ComponentRejected("invalid_component", str(exc)) from exc
        if rv.namespace != self.namespace:
            raise ComponentRejected(
                "namespace_mismatch", "component belongs to another pod"
            )
        now = require_aware(self.clock(), "clock")
        cutoff = rv.as_of or now
        if cutoff > now:
            raise ComponentRejected(
                "future_time", "as_of exceeds the service clock"
            )
        # Cache lookup directly avoids any implicit PaperService network fallback.
        pid, version = canonical_paper_id(rv.paper_id)
        pid = self.papers.store.resolve(pid) or pid
        try:
            card = self.papers.store.get_row(pid, rv.version_key)
        except PapersError as exc:
            raise ComponentRejected("ungrounded_claims", str(exc)) from exc
        if card is None or (version is not None and card.version != version):
            raise ComponentRejected(
                "version_mismatch", "requested cached version is absent"
            )
        if (
            card.version_date is None
            or card.version_date > cutoff
            or (
                card.first_public_date is not None
                and card.first_public_date > cutoff
            )
        ):
            raise ComponentRejected(
                "future_version", "version is undated or unavailable at as_of"
            )
        if card.retrieved_at > now:
            raise ComponentRejected(
                "future_receipt",
                "cached text receipt exceeds the service clock",
            )
        pinned = bool(
            card.arxiv_id
            and card.version
            and card.version_key == f"v{card.version}"
        )
        if rv.source == "abstract" and not pinned:
            raise ComponentRejected(
                "unpinned_abstract",
                "abstract evidence requires an immutable paper version",
            )
        if not pinned and card.retrieved_at > cutoff:
            raise ComponentRejected(
                "late_text", "unversioned text was received after as_of"
            )
        selected = {c.claim_id: c for c in card.claims}
        if len(set(rv.claim_ids)) != len(rv.claim_ids) or any(
            c not in selected for c in rv.claim_ids
        ):
            raise ComponentRejected(
                "ungrounded_claims", "select distinct cached claim ids"
            )
        claims = tuple(selected[c] for c in sorted(rv.claim_ids))
        if any(c.source != rv.source or not c.quote.strip() for c in claims):
            raise ComponentRejected(
                "ungrounded_claims", "each claim must quote the chosen source"
            )
        source_hash = (
            card.abstract_hash if rv.source == "abstract" else card.content_hash
        )
        try:
            body = self.papers.text_of(card, rv.source)
            problems = self.papers.verify_claims(
                card.model_copy(update={"claims": list(claims)})
            )
        except PapersError as exc:
            raise ComponentRejected("ungrounded_claims", str(exc)) from exc
        if not body or sha256_text(body) != source_hash or problems:
            raise ComponentRejected(
                "ungrounded_claims",
                "source hash or exact quote span failed verification",
            )
        limitations = [
            "Selected component only; not replication; no formation or admission validity established.",
            "Quote relevance is a declared research interpretation, not a verified paper result.",
        ]
        if rv.source == "abstract":
            limitations.append(
                "Abstract only: full text and paper controls have not been verified."
            )
        elif card.content_truncated:
            limitations.append(
                "Cached full text is truncated; only the selected excerpt is verified."
            )
        if card.retrieved_at > cutoff:
            limitations.append(
                "Retrieved after as_of; evidence is bound to the earlier immutable public version."
            )
        limitations += [
            "Unavailable control: " + c for c in rv.unavailable_controls
        ]
        payload = dict(
            namespace=rv.namespace,
            component_kind=rv.component_kind,
            component_name=rv.component_name,
            paper_id=card.paper_id,
            version_key=card.version_key,
            source=rv.source,
            source_hash=source_hash,
            version_date=card.version_date.isoformat(),
            retrieved_at=card.retrieved_at.isoformat(),
            as_of=cutoff.isoformat(),
            claims=[c.model_dump(mode="json") for c in claims],
            unavailable_controls=rv.unavailable_controls,
            limitations=limitations,
            label="component_only",
            replication_status="not_replication",
        )
        component_id = "component_" + canonical_digest(payload)[:24]
        text = (
            f"component_only; not_replication. Selected {rv.component_kind}: {rv.component_name}\n"
            f"Source {card.paper_id}@{card.version_key}; {rv.source}; source hash {source_hash}; "
            f"version date {card.version_date.isoformat()}; receipt {card.retrieved_at.isoformat()}.\n"
            + "\n".join(
                f"Exact quote [{c.claim_id}] ({c.char_start}:{c.char_end}): {c.quote}"
                for c in claims
            )
            + "\nLimitations: "
            + " ".join(limitations)
        )
        evidence = EvidenceItem(
            evidence_id=component_id,
            kind="paper_card",
            title=card.title[:300],
            text=text,
            available_at=card.version_date,
            availability_verified=True,
            content_hash=canonical_digest({"component": payload, "text": text}),
            injection_flags=scan_injection(text),
        )
        record = {
            **payload,
            "component_id": component_id,
            "evidence": evidence.model_dump(mode="json"),
        }
        return ComponentEvidence.model_validate(
            {**record, "content_hash": canonical_digest(record)}
        )

    def verify(self, component: ComponentEvidence | dict) -> ComponentEvidence:
        try:
            supplied = ComponentEvidence.model_validate(component)
        except ValidationError as exc:
            raise ComponentRejected("invalid_component", str(exc)) from exc
        pid = supplied.paper_id
        if pid.startswith("arxiv:") and supplied.version_key.startswith("v"):
            pid += supplied.version_key
        actual = self.resolve(
            ComponentRequest(
                namespace=supplied.namespace,
                component_kind=supplied.component_kind,
                component_name=supplied.component_name,
                paper_id=pid,
                version_key=supplied.version_key,
                claim_ids=[c.claim_id for c in supplied.claims],
                source=supplied.source,
                as_of=supplied.as_of,
                unavailable_controls=list(supplied.unavailable_controls),
            )
        )
        if supplied.model_dump(mode="json") != actual.model_dump(mode="json"):
            raise ComponentRejected(
                "component_tamper",
                "component differs from cached, source-grounded evidence",
            )
        return actual
