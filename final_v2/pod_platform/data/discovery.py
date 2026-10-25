"""Bounded alternative-source discovery (spec 8.4; catalogue is documentation, never entitlement).

Three things live here: (1) search/describe over the configured ``CatalogueManifest`` cards plus host-approved
metadata, (2) provider discovery and bounded samples through *host-registered* connector objects under a finite,
durable per-(namespace, provider) query budget with a query cache, and (3) a pending-review queue of proposed
sources/connector work items. A discovered source or sample never establishes history, vintage, coverage or
formation eligibility: only an integrity-owned review of a ``SourceCard`` whose established screen items cite
recorded non-sample evidence is accepted, and even then the catalogue is not mutated (the host imports the card).

Connectors are trusted host code (they alone read named environment credentials); this service never receives a
URL, path, module name or secret from a caller, and passes connectors only query text, as-of and limits.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import threading
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, Sequence

from pydantic import Field, ValidationError

from ..application.common import fail, now_utc
from ..application.namespaces import RoleContext, RoleGateway
from ..contracts.base import StrictModel, canonical_digest
from ..contracts.errors import ErrorCode
from ..contracts.sources import AssessmentStatus, CardStatus, SourceCard
from ..research.jobs import open_sqlite
from .catalogue import CatalogueManifest

Availability = Literal["available", "unsupported", "entitlement_blocked", "unresolved_history", "unavailable_as_of"]
MAX_QUERY_LIMIT, MAX_SAMPLE_ROWS, MAX_SAMPLE_COLUMNS = 10, 20, 12
_IDENT = r"^[a-z][a-z0-9_]{0,63}$"
_QUERY_OK = re.compile(r"^[A-Za-z0-9 ,.\-_&%()']{2,200}$")  # no ':' '/' '\\' '@' '=': no URLs, paths or key=value
_QUERY_DENIED = re.compile(r"(www\.|api[_ -]?key|token|password|secret|bearer|credential)", re.I)
_STOP = frozenset("the and for with from that this into over data dataset source sources of on in to a an by is are".split())

_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_budgets (namespace TEXT NOT NULL, provider_id TEXT NOT NULL, total INTEGER NOT NULL,
    used INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (namespace, provider_id));
CREATE TABLE IF NOT EXISTS query_log (seq INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL, namespace TEXT NOT NULL,
    provider_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS query_cache (fingerprint TEXT PRIMARY KEY, namespace TEXT NOT NULL, provider_id TEXT NOT NULL,
    kind TEXT NOT NULL, result_json TEXT NOT NULL, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS proposals (proposal_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, provider_id TEXT NOT NULL,
    provider_ref TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL, candidate_json TEXT NOT NULL, rationale TEXT NOT NULL,
    query_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE (namespace, provider_id, provider_ref));
CREATE TABLE IF NOT EXISTS evidence (evidence_id TEXT PRIMARY KEY, proposal_id TEXT, kind TEXT NOT NULL, summary TEXT NOT NULL,
    digest TEXT NOT NULL, recorded_by TEXT NOT NULL, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (proposal_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reviewer TEXT NOT NULL, card_json TEXT,
    note TEXT NOT NULL, at TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS query_log_no_update BEFORE UPDATE ON query_log BEGIN SELECT RAISE(ABORT, 'query log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS query_log_no_delete BEFORE DELETE ON query_log BEGIN SELECT RAISE(ABORT, 'query log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS query_cache_no_update BEFORE UPDATE ON query_cache BEGIN SELECT RAISE(ABORT, 'query cache entries are immutable'); END;
CREATE TRIGGER IF NOT EXISTS reviews_no_update BEFORE UPDATE ON reviews BEGIN SELECT RAISE(ABORT, 'reviews are immutable'); END;
CREATE TRIGGER IF NOT EXISTS reviews_no_delete BEFORE DELETE ON reviews BEGIN SELECT RAISE(ABORT, 'reviews are immutable'); END;
CREATE TRIGGER IF NOT EXISTS evidence_no_update BEFORE UPDATE ON evidence BEGIN SELECT RAISE(ABORT, 'evidence is immutable'); END;
"""


class CatalogueMeta(StrictModel):
    """Host-approved search metadata for one catalogue card (the card itself has only topic and grain)."""

    source_id: str
    title: str = Field(min_length=1, max_length=200)
    concepts: list[str] = Field(default_factory=list, max_length=30)
    fields: list[str] = Field(default_factory=list, max_length=60)
    description: str = Field(default="", max_length=1000)


class ProviderCandidate(StrictModel):
    provider_ref: str = Field(pattern=_IDENT)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=600)
    concepts: list[str] = Field(default_factory=list, max_length=20)
    fields: list[str] = Field(default_factory=list, max_length=40)
    entity_grain: list[str] = Field(default_factory=list, max_length=6)
    dataset_hint: Optional[str] = Field(default=None, max_length=100)
    history_start: Optional[dt.date] = None
    availability: Availability


class SourceConnector(Protocol):
    """Host-registered, trusted provider adapter. Reads its own named env credentials internally; receives no secrets."""

    provider_id: str

    def discover(self, query: str, *, as_of: dt.datetime, limit: int) -> Sequence[Any]: ...

    def sample(self, provider_ref: str, *, as_of: dt.datetime, max_rows: int) -> Mapping[str, Any]: ...


class _Rejected(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


def _stem(token: str) -> str:
    return re.sub(r"(ing|ies|es|s|ed)$", "", token) if len(token) > 4 else token


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOP}


def _hit(query_stem: str, stems: set[str]) -> bool:
    return any(query_stem == s or (len(query_stem) >= 4 and len(s) >= 4 and (s.startswith(query_stem) or query_stem.startswith(s))) for s in stems)


def check_query(query: Any) -> str:
    if not isinstance(query, str) or not _QUERY_OK.match(query) or _QUERY_DENIED.search(query):
        fail(ErrorCode.SCHEMA_MISMATCH, "query must be 2-200 characters of plain words (no URLs, paths, credentials or key=value)",
             "Queries are economic concepts or field names; the service never fetches arbitrary locations.")
    return " ".join(query.split())


def card_availability(card: SourceCard) -> str:
    if card.status in (CardStatus.REJECTED, CardStatus.QUARANTINED):
        return "blocked"
    if not card.access_confirmed:
        return "entitlement_unconfirmed"
    if card.screen.history.status != AssessmentStatus.ESTABLISHED:
        return "unresolved_history"
    return "available" if card.formation_eligible else "available_not_formation_eligible"


class DiscoveryService:
    def __init__(self, path, *, catalogue: CatalogueManifest, metadata: Sequence[CatalogueMeta] = (),
                 gateway: Optional[RoleGateway] = None, clock: Callable[[], dt.datetime] = now_utc):
        self.path, self.catalogue, self.gateway, self._clock = path, catalogue, gateway or RoleGateway(), clock
        self._meta = {m.source_id: m for m in metadata}
        for source_id in self._meta:
            catalogue.card(source_id)  # metadata for an unknown card is a configuration error (KeyError)
        self._connectors: dict[str, SourceConnector] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        with open_sqlite(path) as conn:
            conn.executescript(_SCHEMA)

    def _at(self) -> str:
        return self._clock().isoformat()

    # -- host configuration (trusted code only; never exposed as an agent tool) ------------------------------
    def register_connector(self, connector: SourceConnector) -> None:
        provider_id = getattr(connector, "provider_id", None)
        if not isinstance(provider_id, str) or not re.match(_IDENT, provider_id):
            raise ValueError("connector.provider_id must be a plain lower-case identifier")
        if not callable(getattr(connector, "discover", None)) or not callable(getattr(connector, "sample", None)):
            raise ValueError("connector must implement discover() and sample()")
        if provider_id in self._connectors:
            raise ValueError(f"connector {provider_id} already registered")
        self._connectors[provider_id] = connector

    def set_provider_budget(self, namespace: str, provider_id: str, total_queries: int) -> int:
        """Finite provider-query budget; re-stating the same total is a no-op, changing it is refused."""
        if not re.match(_IDENT, namespace or "") or not re.match(_IDENT, provider_id or "") or int(total_queries) < 0:
            raise ValueError("invalid budget configuration")
        with open_sqlite(self.path, immediate=True) as conn:
            row = conn.execute("SELECT total FROM provider_budgets WHERE namespace=? AND provider_id=?", (namespace, provider_id)).fetchone()
            if row is None:
                conn.execute("INSERT INTO provider_budgets (namespace, provider_id, total) VALUES (?,?,?)", (namespace, provider_id, int(total_queries)))
            elif row[0] != int(total_queries):
                raise ValueError(f"budget for {namespace}/{provider_id} is already fixed at {row[0]}")
        return int(total_queries)

    def queries_remaining(self, namespace: str, provider_id: str) -> Optional[int]:
        with open_sqlite(self.path) as conn:
            row = conn.execute("SELECT total, used FROM provider_budgets WHERE namespace=? AND provider_id=?", (namespace, provider_id)).fetchone()
        return None if row is None else row[0] - row[1]

    # -- approved catalogue search/describe ---------------------------------------------------------------------
    def _view(self, card: SourceCard) -> dict:
        meta = self._meta.get(card.source_id)
        return {
            "source_id": card.source_id, "dataset_id": card.dataset_id, "topic": card.topic, "title": meta.title if meta else card.source_id,
            "concepts": list(meta.concepts) if meta else [], "fields": list(meta.fields) if meta else [],
            "entity_grain": list(card.entity_grain), "card_status": card.status.value, "availability": card_availability(card),
            "access_confirmed": card.access_confirmed, "formation_eligible": card.formation_eligible,
            "screen": {name: getattr(card.screen, name).status.value for name in ("history", "metric_lag", "vintage", "coverage")},
            "unresolved_questions": list(card.unresolved_questions),
        }

    def search_catalogue(self, ctx: RoleContext, query: str, *, limit: int = 5) -> list[dict]:
        self.gateway.authorize(ctx, "catalogue.search")
        wanted = _stems(check_query(query))
        hits = []
        for source_id in self._meta:  # only host-approved, configured metadata is searchable
            card, meta = self.catalogue.card(source_id), self._meta[source_id]
            parts = ((3, _stems(" ".join(meta.concepts))), (2, _stems(" ".join(meta.fields))), (2, _stems(card.topic)),
                     (1, _stems(meta.title)), (1, _stems(meta.description)), (1, _stems(" ".join(card.entity_grain))),
                     (1, _stems(f"{card.source_id} {card.dataset_id}".replace("_", " "))))
            score = sum(max((w for w, stems in parts if _hit(q, stems)), default=0) for q in wanted)
            if score:
                hits.append((-score, source_id, {**self._view(card), "score": score}))
        return [h[2] for h in sorted(hits, key=lambda h: h[:2])[: max(1, min(int(limit), 20))]]

    def describe_catalogue(self, ctx: RoleContext, source_id: str) -> dict:
        self.gateway.authorize(ctx, "catalogue.describe")
        if source_id not in self._meta:
            fail(ErrorCode.SCHEMA_MISMATCH, f"unknown catalogue source {source_id!r}", "Only approved configured catalogue entries are described.")
        view = self._view(self.catalogue.card(source_id))
        view["description"] = self._meta[source_id].description
        return view

    # -- provider discovery and bounded samples ---------------------------------------------------------------
    def discover(self, ctx: RoleContext, namespace: str, provider_id: str, query: str, *, limit: int = MAX_QUERY_LIMIT) -> dict:
        self.gateway.authorize(ctx, "source.discover", namespace)
        query = check_query(query)
        limit = max(1, min(int(limit), MAX_QUERY_LIMIT))
        as_of = ctx.research_asof
        if provider_id not in self._connectors:
            return self._explicit("unsupported", namespace, provider_id, "no connector is registered for this provider")
        fp = canonical_digest({"k": "discover", "ns": namespace, "p": provider_id, "q": query.lower(), "n": limit, "d": as_of.date().isoformat()})

        def post(raw: Any) -> dict:
            candidates, dropped, seen = [], 0, set()
            for item in list(raw)[: limit * 3]:
                try:
                    cand = item if isinstance(item, ProviderCandidate) else ProviderCandidate.model_validate(item)
                except (ValidationError, TypeError):
                    dropped += 1
                    continue
                if cand.provider_ref in seen:
                    dropped += 1
                    continue
                seen.add(cand.provider_ref)
                availability = cand.availability
                if availability == "available" and cand.history_start is None:
                    availability = "unresolved_history"
                elif availability == "available" and cand.history_start > as_of.date():
                    availability = "unavailable_as_of"
                row = cand.model_copy(update={"availability": availability}).model_dump(mode="json")
                row.update(historical_coverage_established=False, formation_eligible=False)
                candidates.append(row)
            return {"status": "ok", "candidates": candidates[:limit], "dropped_candidates": dropped + max(0, len(candidates) - limit)}

        result, cached = self._provider_call(namespace, provider_id, "discover", fp, lambda c: c.discover(query, as_of=as_of, limit=limit), post)
        return {**result, "cached": cached, "fingerprint": fp, "provider_id": provider_id, "namespace": namespace,
                "as_of": as_of.isoformat(), "queries_remaining": self.queries_remaining(namespace, provider_id)}

    def sample(self, ctx: RoleContext, namespace: str, query_fingerprint: str, provider_ref: str, *, max_rows: int = 5) -> dict:
        self.gateway.authorize(ctx, "source.sample", namespace)
        max_rows = max(1, min(int(max_rows), MAX_SAMPLE_ROWS))
        as_of = ctx.research_asof
        found = self._candidate(namespace, query_fingerprint, provider_ref)
        provider_id, cand = found
        if cand["availability"] != "available" or provider_id not in self._connectors:
            return self._explicit(cand["availability"] if cand["availability"] != "available" else "unsupported", namespace, provider_id,
                                  "sampling is only attempted for sources discovered as available")
        fp = canonical_digest({"k": "sample", "ns": namespace, "p": provider_id, "r": provider_ref, "n": max_rows, "d": as_of.date().isoformat()})

        def post(raw: Any) -> dict:
            columns = [str(c) for c in list(raw.get("columns", []))[:MAX_SAMPLE_COLUMNS]]
            rows = []
            for row in list(raw.get("rows", []))[:max_rows]:
                clean = {}
                for col in columns:
                    value = row.get(col)
                    if isinstance(value, str):
                        value = value[:80]
                        if re.match(r"^\d{4}-\d{2}-\d{2}", value) and dt.date.fromisoformat(value[:10]) > as_of.date():
                            raise _Rejected("rejected_asof", f"sample row dated {value[:10]} is after the research as-of")
                    elif value is not None and not isinstance(value, (int, float, bool)):
                        raise _Rejected("rejected_sample", "sample cells must be scalars")
                    clean[col] = value
                rows.append(clean)
            return {"status": "ok", "columns": columns, "rows": rows, "row_count": len(rows), "sample_digest": canonical_digest({"c": columns, "r": rows})}

        result, cached = self._provider_call(namespace, provider_id, "sample", fp, lambda c: c.sample(provider_ref, as_of=as_of, max_rows=max_rows), post)
        if result.get("status") == "ok":
            result = {**result, "evidence_id": f"sample:{result['sample_digest'][:24]}", "coverage": AssessmentStatus.SAMPLE_ONLY.value,
                      "historical_coverage_established": False}
            with open_sqlite(self.path, immediate=True) as conn:
                conn.execute("INSERT OR IGNORE INTO evidence VALUES (?,?,?,?,?,?,?)",
                             (result["evidence_id"], None, "sample", f"sample of {provider_id}/{provider_ref}", result["sample_digest"], namespace, self._at()))
        return {**result, "cached": cached, "fingerprint": fp, "provider_id": provider_id, "provider_ref": provider_ref}

    @staticmethod
    def _explicit(status: str, namespace: str, provider_id: str, reason: str) -> dict:
        return {"status": status, "reason": reason, "namespace": namespace, "provider_id": provider_id, "candidates": [], "rows": [],
                "queries_spent": 0, "historical_coverage_established": False}

    def _candidate(self, namespace: str, fingerprint: str, provider_ref: str) -> tuple[str, dict]:
        with open_sqlite(self.path) as conn:
            row = conn.execute("SELECT provider_id, result_json FROM query_cache WHERE fingerprint=? AND namespace=? AND kind='discover'",
                               (fingerprint, namespace)).fetchone()
        for cand in (json.loads(row[1])["candidates"] if row else []):
            if cand["provider_ref"] == provider_ref:
                return row[0], cand
        fail(ErrorCode.SCHEMA_MISMATCH, "candidate was not returned by a discovery in this namespace", "Sample and propose only discovered candidates.")

    def _lock_for(self, fp: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(fp, threading.Lock())

    def _provider_call(self, namespace, provider_id, kind, fp, call, post) -> tuple[dict, bool]:
        """Cache lookup, atomic budget reservation (BEFORE the callback), callback, then the cached/logged outcome."""
        with self._lock_for(fp):
            refusal = None
            with open_sqlite(self.path, immediate=True) as conn:
                hit = conn.execute("SELECT result_json FROM query_cache WHERE fingerprint=?", (fp,)).fetchone()
                if hit:
                    conn.execute("INSERT INTO query_log (fingerprint, namespace, provider_id, kind, status, detail, at) VALUES (?,?,?,?,?,?,?)",
                                 (fp, namespace, provider_id, kind, "cache_hit", "", self._at()))
                    return json.loads(hit[0]), True
                budget = conn.execute("SELECT total, used FROM provider_budgets WHERE namespace=? AND provider_id=?", (namespace, provider_id)).fetchone()
                if budget is None or budget[1] >= budget[0]:
                    refusal = "no_budget_configured" if budget is None else "budget_exhausted"
                    conn.execute("INSERT INTO query_log (fingerprint, namespace, provider_id, kind, status, detail, at) VALUES (?,?,?,?,?,?,?)",
                                 (fp, namespace, provider_id, kind, "refused", refusal, self._at()))
                else:
                    conn.execute("UPDATE provider_budgets SET used = used + 1 WHERE namespace=? AND provider_id=?", (namespace, provider_id))
                    conn.execute("INSERT INTO query_log (fingerprint, namespace, provider_id, kind, status, detail, at) VALUES (?,?,?,?,?,?,?)",
                                 (fp, namespace, provider_id, kind, "reserved", "", self._at()))
            if refusal:
                fail(ErrorCode.PROVIDER_BUDGET_EXCEEDED, f"provider query budget refused for {namespace}/{provider_id}: {refusal}",
                     "Provider queries are finite; a cached identical query is free, a new one needs remaining budget.")
            try:
                result = post(call(self._connectors[provider_id]))
            except _Rejected as exc:
                result = {"status": exc.status, "reason": exc.reason}
            except Exception as exc:  # connector failures are explicit results; the spent query stays on the log
                result = {"status": "connector_error", "error": type(exc).__name__}
            with open_sqlite(self.path, immediate=True) as conn:
                conn.execute("INSERT INTO query_log (fingerprint, namespace, provider_id, kind, status, detail, at) VALUES (?,?,?,?,?,?,?)",
                             (fp, namespace, provider_id, kind, result["status"], result.get("reason", result.get("error", "")), self._at()))
                if result["status"] == "ok":
                    try:
                        conn.execute("INSERT INTO query_cache VALUES (?,?,?,?,?,?)",
                                     (fp, namespace, provider_id, kind, json.dumps(result, sort_keys=True), self._at()))
                    except sqlite3.IntegrityError:  # another process cached the same fingerprint first: use its stored result
                        result = json.loads(conn.execute("SELECT result_json FROM query_cache WHERE fingerprint=?", (fp,)).fetchone()[0])
            return result, False

    def query_log(self, namespace: Optional[str] = None) -> list[dict]:
        with open_sqlite(self.path) as conn:
            rows = conn.execute("SELECT fingerprint, namespace, provider_id, kind, status, detail FROM query_log"
                                + (" WHERE namespace=?" if namespace else "") + " ORDER BY seq", (namespace,) if namespace else ()).fetchall()
        return [dict(zip(("fingerprint", "namespace", "provider_id", "kind", "status", "detail"), r)) for r in rows]

    # -- proposals (own namespace) and integrity review --------------------------------------------------------
    def propose_source(self, ctx: RoleContext, namespace: str, query_fingerprint: str, provider_ref: str, rationale: str) -> dict:
        self.gateway.authorize(ctx, "source.propose", namespace)
        if not isinstance(rationale, str) or not 10 <= len(rationale.strip()) <= 600:
            fail(ErrorCode.SCHEMA_MISMATCH, "rationale must be 10-600 characters", "State the economic reason for the source.")
        provider_id, cand = self._candidate(namespace, query_fingerprint, provider_ref)
        hint = cand.get("dataset_hint")
        known = [c for c in (self.catalogue.card(s) for s in self.catalogue.source_ids()) if hint and c.dataset_id == hint]
        if known:
            fail(ErrorCode.SCHEMA_MISMATCH, f"candidate duplicates catalogue source {known[0].source_id}", "Use the catalogue entry instead of proposing it again.")
        proposal_id = "prop_" + canonical_digest({"ns": namespace, "p": provider_id, "r": provider_ref})[:16]
        kind = "connector_work_item" if cand["availability"] in ("unsupported", "entitlement_blocked") else "new_source"
        with open_sqlite(self.path, immediate=True) as conn:
            if conn.execute("SELECT 1 FROM proposals WHERE proposal_id=?", (proposal_id,)).fetchone():
                return {**self.get_proposal(proposal_id), "duplicate": True}
            conn.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?,?,?)", (proposal_id, namespace, provider_id, provider_ref, kind, "pending_review",
                         json.dumps(cand, sort_keys=True), rationale.strip(), query_fingerprint, self._at()))
        return {**self.get_proposal(proposal_id), "duplicate": False}

    def get_proposal(self, proposal_id: str) -> Optional[dict]:
        with open_sqlite(self.path) as conn:
            row = conn.execute("SELECT proposal_id, namespace, provider_id, provider_ref, kind, status, candidate_json, rationale, query_fingerprint "
                               "FROM proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if row is None:
            return None
        out = dict(zip(("proposal_id", "namespace", "provider_id", "provider_ref", "kind", "status"), row[:6]))
        return {**out, "candidate": json.loads(row[6]), "rationale": row[7], "query_fingerprint": row[8], "formation_eligible": False}

    def list_proposals(self, ctx: RoleContext, namespace: Optional[str] = None) -> list[dict]:
        if ctx.caller_role == "strategy":
            self.gateway.authorize(ctx, "source.propose", namespace or ctx.namespace)
            namespace = ctx.namespace
        else:
            self.gateway.authorize(ctx, "source.review")
        with open_sqlite(self.path) as conn:
            ids = [r[0] for r in conn.execute("SELECT proposal_id FROM proposals" + (" WHERE namespace=?" if namespace else "") + " ORDER BY created_at, proposal_id",
                                              (namespace,) if namespace else ())]
        return [self.get_proposal(i) for i in ids]

    def record_evidence(self, ctx: RoleContext, proposal_id: str, evidence_id: str, summary: str, content_digest: str) -> dict:
        """Integrity records a non-sample attestation (e.g. a vendor history statement) a reviewed card may cite."""
        self.gateway.authorize(ctx, "source.review")
        if self.get_proposal(proposal_id) is None or not re.match(r"^[A-Za-z0-9_.:\-]{3,100}$", evidence_id or "") or evidence_id.startswith("sample:"):
            fail(ErrorCode.SCHEMA_MISMATCH, "unknown proposal or invalid evidence id", "Attestation ids are not sample ids.")
        if not re.match(r"^[0-9a-f]{64}$", content_digest or "") or not summary.strip():
            fail(ErrorCode.SCHEMA_MISMATCH, "evidence needs a summary and a sha256 content digest", "Evidence is hash-bound.")
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?,?)", (evidence_id, proposal_id, "attestation", summary[:500], content_digest, ctx.namespace, self._at()))
        return {"evidence_id": evidence_id, "proposal_id": proposal_id, "kind": "attestation"}

    def review_proposal(self, ctx: RoleContext, proposal_id: str, decision: Literal["accept", "reject"], note: str,
                        card: "SourceCard | dict | None" = None) -> dict:
        """Integrity-owned. ``accept`` needs a full ``SourceCard``; every ESTABLISHED screen item must cite recorded
        non-sample evidence for this proposal. The result is stored, never added to the catalogue or made eligible."""
        self.gateway.authorize(ctx, "source.review")
        proposal = self.get_proposal(proposal_id)
        if proposal is None or proposal["status"] != "pending_review" or decision not in ("accept", "reject") or not note.strip():
            fail(ErrorCode.SCHEMA_MISMATCH, "proposal is unknown, already reviewed, or the decision/note is invalid", "Reviews are immutable.")
        sealed_json = None
        if decision == "accept":
            try:
                card = card if isinstance(card, SourceCard) else SourceCard.model_validate(card)
            except (ValidationError, TypeError) as exc:
                fail(ErrorCode.SCHEMA_MISMATCH, f"acceptance requires a valid SourceCard: {str(exc)[:200]}", "Booleans do not approve sources.")
            self._check_card(card, proposal)
            sealed_json = json.dumps(card.sealed().model_dump(mode="json"), sort_keys=True)
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute("INSERT INTO reviews VALUES (?,?,?,?,?,?)", (proposal_id, decision, ctx.namespace, sealed_json, note.strip()[:500], self._at()))
            conn.execute("UPDATE proposals SET status=? WHERE proposal_id=?", ("accepted_pending_import" if decision == "accept" else "rejected", proposal_id))
        return self.get_proposal(proposal_id)

    def _check_card(self, card: SourceCard, proposal: dict) -> None:
        cand = proposal["candidate"]
        problems = []
        if card.source_id in self.catalogue.source_ids() or any(c.dataset_id == card.dataset_id for c in map(self.catalogue.card, self.catalogue.source_ids())):
            problems.append("card duplicates a catalogue source")
        if card.dataset_id not in {cand.get("dataset_hint"), proposal["provider_ref"]}:
            problems.append("card.dataset_id does not match the proposed source")
        if cand["availability"] == "entitlement_blocked" and card.access_confirmed:
            problems.append("access_confirmed contradicts the connector's entitlement block")
        if card.status == CardStatus.APPROVED and not (card.access_confirmed and card.screen.all_established):
            problems.append("an APPROVED card needs confirmed access and all four screens established")
        if card.formation_eligible and card.status != CardStatus.APPROVED:
            problems.append("a formation-eligible card must be APPROVED")
        with open_sqlite(self.path) as conn:
            for name in ("history", "metric_lag", "vintage", "coverage"):
                item = getattr(card.screen, name)
                if item.status != AssessmentStatus.ESTABLISHED:
                    continue
                row = conn.execute("SELECT kind, proposal_id FROM evidence WHERE evidence_id=?", (item.evidence_id,)).fetchone()
                if row is None or row[0] != "attestation" or row[1] != proposal["proposal_id"]:
                    problems.append(f"{name} is ESTABLISHED without recorded non-sample evidence for this proposal")
        if problems:
            fail(ErrorCode.SCHEMA_MISMATCH, "; ".join(problems), "A discovery result or sample cannot establish coverage; cite integrity evidence.")

    def reviewed_card(self, proposal_id: str) -> Optional[SourceCard]:
        with open_sqlite(self.path) as conn:
            row = conn.execute("SELECT card_json FROM reviews WHERE proposal_id=? AND decision='accept'", (proposal_id,)).fetchone()
        return SourceCard.model_validate(json.loads(row[0])) if row and row[0] else None
