"""PaperService: search / get / ingest / claims / replication specs over a persistent cache.

Network happens only when ``online=True`` AND a transport is available; otherwise
only the SQLite cache is used and a miss raises ``OfflineError``.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel

from . import providers as P
from .models import (
    CLAIM_KINDS,
    AccessStatus,
    Claim,
    ClaimError,
    NotFound,
    OfflineError,
    PaperCard,
    PapersError,
    PaperValidationError,
    ProviderError,
    ReplicationLabel,
    TemporalError,
    canonical_json,
    canonical_paper_id,
    ensure_aware,
    normalize_doi,
    parse_as_of,
    sha256_text,
)
from .store import PaperStore
from .text import (
    PdfUnavailable,
    extract_pdf_text,
    heuristic_spans,
    page_of,
    section_of,
)
from .transport import (
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT,
    Transport,
    check_status,
    validate_url,
)

REQUIRED_FOR = {
    ReplicationLabel.EXACT_REPLICATION: (
        "economic_claim",
        "universe",
        "period",
        "target",
        "horizon",
        "variable_definition",
        "timing",
        "statistical_design",
        "baseline",
    ),
    ReplicationLabel.MECHANISM_TRANSFER: ("economic_claim", "mechanism"),
    ReplicationLabel.METHOD_TRANSFER: ("statistical_design", "baseline"),
}
REQUIRED_FOR[ReplicationLabel.CLOSE_REPLICATION] = REQUIRED_FOR[
    ReplicationLabel.EXACT_REPLICATION
]


class SearchResult(BaseModel):
    query: str
    mode: str  # "contemporary" | "historical_replay"
    as_of: Optional[datetime]
    origin: str  # "network" | "query_cache" | "local_cache"
    cards: list[PaperCard]
    warnings: list[str]


class PaperService:
    def __init__(
        self,
        store,
        *,
        clock: Callable[[], datetime],
        transport: Optional[Transport] = None,
        online: bool = False,
        openalex_api_key: Optional[str] = None,
        openalex_mailto: Optional[str] = None,
        providers: Iterable[str] = ("arxiv", "openalex"),
        timeout: float = DEFAULT_TIMEOUT,
        max_body_bytes: int = DEFAULT_MAX_BYTES,
        max_text_chars: int = 2_000_000,
        max_version_lookups: int = 12,
    ):
        self.store = (
            store if isinstance(store, PaperStore) else PaperStore(store)
        )
        self.clock, self.transport, self.online = clock, transport, bool(online)
        self._oa_key, self._oa_mailto = openalex_api_key, openalex_mailto
        self.providers = tuple(providers)
        if not set(self.providers) <= {"arxiv", "openalex"}:
            raise PaperValidationError("unknown provider")
        self.timeout, self.max_body = float(timeout), int(max_body_bytes)
        self.max_text_chars, self.max_lookups = (
            int(max_text_chars),
            int(max_version_lookups),
        )
        if self.timeout <= 0 or self.max_body <= 0 or self.max_text_chars <= 0:
            raise PaperValidationError(
                "timeout and size limits must be positive"
            )
        self.warnings: list[str] = []

    # ------------------------------------------------------------ plumbing
    def now(self) -> datetime:
        return ensure_aware(self.clock(), "clock()")

    def _as_of(self, as_of: Any) -> Optional[datetime]:
        v = parse_as_of(as_of)
        if v is not None and v > self.now():
            raise TemporalError("as_of is later than the service clock")
        return v

    def _fetch(self, url: str, provider: str, accept: str) -> bytes:
        if not self.online or self.transport is None:
            raise OfflineError(
                "network access is disabled (pass online=True / --online) "
                "and the item is not in the cache"
            )
        validate_url(url)
        resp = self.transport.get(
            url,
            headers={"Accept": accept},
            timeout=self.timeout,
            max_bytes=self.max_body,
        )
        check_status(resp, url, provider)
        if len(resp.body) > self.max_body:
            raise ProviderError("response body exceeds the configured limit")
        return resp.body

    # ----------------------------------------------------------- selection
    @staticmethod
    def _pick(
        rows: list[PaperCard], as_of: Optional[datetime]
    ) -> Optional[PaperCard]:
        """Choose the row valid at ``as_of`` (fail closed: unknown dates never qualify)."""
        epoch = datetime.min.replace(tzinfo=timezone.utc)
        if as_of is None:
            return max(
                rows,
                key=lambda c: (
                    c.version_date or c.first_public_date or epoch,
                    c.version or 0,
                ),
                default=None,
            )
        cand = [
            c
            for c in rows
            if c.version_date is not None
            and c.version_date <= as_of
            and (c.first_public_date is None or c.first_public_date <= as_of)
        ]
        ax = [c for c in cand if c.version is not None]
        if ax:
            top = max(ax, key=lambda c: c.version)
            nxt_known = any(r.version == top.version + 1 for r in rows)
            latest_rows = [r for r in rows if r.latest_version]
            proven = nxt_known or any(
                r.latest_version == top.version and as_of <= r.retrieved_at
                for r in latest_rows
            )
            if not proven:  # a later version dated <= as_of might exist; cannot tell offline
                cand = [c for c in cand if c.version is None]
        best = max(
            cand, key=lambda c: (c.version_date, c.version or 0), default=None
        )
        return (
            None if best is None else PaperService._point_in_time(best, as_of)
        )

    @staticmethod
    def _point_in_time(card: PaperCard, as_of: datetime) -> PaperCard:
        """Drop abstract text that cannot be shown to match what was public at ``as_of``.

        Version-pinned arXiv rows keep their per-version abstract. An unversioned row
        (e.g. an OpenAlex work) carries whatever abstract the provider served when it
        was retrieved; if that was after ``as_of`` the old publication date proves
        nothing about the text, so only metadata survives (claims on it are removed).
        """
        if (
            card.version is not None
            or card.abstract is None
            or card.retrieved_at <= as_of
        ):
            return card
        return card.model_copy(
            update=dict(
                abstract=None,
                abstract_hash=None,
                access_status=card.access_status
                if card.content_hash
                else AccessStatus.METADATA_ONLY,
                text_status=card.text_status
                if card.content_hash
                else "incomplete:abstract_retrieved_after_as_of",
                claims=[c for c in card.claims if c.source != "abstract"],
            )
        )

    def _rows(self, pid: str) -> list[PaperCard]:
        return self.store.rows(self.store.resolve(pid) or pid)

    # --------------------------------------------------------------- arXiv
    def _arxiv_version(self, base: str, version: Optional[int]) -> PaperCard:
        """Cached or fetched card for one arXiv version (``None`` => latest)."""
        if version is not None:
            row = self.store.get_row(
                self.store.resolve(f"arxiv:{base}") or f"arxiv:{base}",
                f"v{version}",
            )
            if row is not None:
                return row
        body = self._fetch(
            P.arxiv_id_url(base, version), "arxiv", "application/atom+xml"
        )
        cards = [
            c
            for c in P.parse_arxiv_feed(body, self.now())
            if c.arxiv_id == base
        ]
        if len(cards) != 1 or (
            version is not None and cards[0].version != version
        ):
            raise NotFound(
                f"arXiv has no entry for {base}"
                + (f"v{version}" if version else "")
            )
        card = cards[0]
        if version is None:
            card = card.model_copy(update={"latest_version": card.version})
        return self.store.upsert_card(card)

    def _resolve_arxiv(
        self, base: str, as_of: Optional[datetime]
    ) -> Optional[PaperCard]:
        pid = self.store.resolve(f"arxiv:{base}") or f"arxiv:{base}"
        if as_of is None:
            rows = self.store.rows(pid)
            if rows or not self.online:
                got = self._pick(rows, None)
                if got is None:
                    raise OfflineError(f"{pid} is not in the cache")
                return got
            return self._arxiv_version(base, None)
        got = self._pick(self.store.rows(pid), as_of)
        if got is not None or not self.online:
            return got
        latest = self._arxiv_version(base, None)
        if latest.first_public_date and latest.first_public_date > as_of:
            return None
        if latest.version_date and latest.version_date <= as_of:
            return latest
        lo, hi, lookups = 1, latest.version or 1, 0
        while hi - lo > 1:  # version dates are monotone in version number
            if lookups >= self.max_lookups:
                self.warnings.append(
                    f"{pid}: version lookup budget exhausted; excluded"
                )
                return None
            mid = (lo + hi) // 2
            lookups += 1
            if self._arxiv_version(base, mid).version_date <= as_of:
                lo = mid
            else:
                hi = mid
        chosen = self._arxiv_version(base, lo)
        return (
            chosen
            if chosen.version_date and chosen.version_date <= as_of
            else None
        )

    # ---------------------------------------------------------------- search
    def search(
        self, query: str, *, as_of: Any = None, limit: int = 10
    ) -> SearchResult:
        """Search providers (online) or the cache (offline). ``as_of`` => historical replay."""
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            raise PaperValidationError("limit must be an integer in 1..50")
        cut = self._as_of(as_of)
        toks = P.clean_query(query)
        mode = "contemporary" if cut is None else "historical_replay"
        qkey = sha256_text(
            canonical_json(
                [
                    toks,
                    cut.isoformat() if cut else None,
                    limit,
                    list(self.providers),
                ]
            )
        )
        self.warnings = []
        if self.online and self.transport is not None:
            cards, errors = self._search_online(query, cut, limit)
            if not cards and errors:
                raise errors[0]
            self.store.put_query(
                qkey,
                [{"p": c.paper_id, "k": c.version_key} for c in cards],
                self.now(),
            )
            return SearchResult(
                query=query,
                mode=mode,
                as_of=cut,
                origin="network",
                cards=cards,
                warnings=self.warnings,
            )
        cached = self.store.get_query(qkey)
        if cached is not None:
            picked, seen = [], set()
            for r in cached:
                if r["p"] in seen:
                    continue
                seen.add(r["p"])
                c = self._pick(self.store.rows(r["p"]), cut)
                if c is not None:
                    picked.append(c)
            return SearchResult(
                query=query,
                mode=mode,
                as_of=cut,
                origin="query_cache",
                cards=picked[:limit],
                warnings=[],
            )
        return SearchResult(
            query=query,
            mode=mode,
            as_of=cut,
            origin="local_cache",
            cards=self._search_local(toks, cut, limit),
            warnings=["offline: results come from the local cache only"],
        )

    def _search_online(self, query, cut, limit):
        per_provider, errors = [], []
        for name in self.providers:
            try:
                if name == "arxiv":
                    url = P.arxiv_search_url(query, limit, cut)
                    cards = P.parse_arxiv_feed(
                        self._fetch(url, "arxiv", "application/atom+xml"),
                        self.now(),
                    )
                    cards = [
                        c.model_copy(update={"latest_version": c.version})
                        for c in cards
                    ]
                else:
                    url = P.openalex_search_url(
                        query,
                        limit,
                        cut,
                        api_key=self._oa_key,
                        mailto=self._oa_mailto,
                    )
                    cards = P.parse_openalex_works(
                        self._fetch(url, "openalex", "application/json"),
                        self.now(),
                    )
                per_provider.append([self.store.upsert_card(c) for c in cards])
            except (PapersError, ValueError) as exc:
                errors.append(exc)
                self.warnings.append(f"{name}: {exc}")
        out, seen = [], set()
        longest = max((len(p) for p in per_provider), default=0)
        for i in range(longest):  # round-robin so no provider starves the other
            for lst in per_provider:
                if i >= len(lst) or lst[i].paper_id in seen:
                    continue
                seen.add(lst[i].paper_id)
                card = self._select_for_search(lst[i], cut)
                if card is not None:
                    out.append(card)
        return out[:limit], errors

    def _select_for_search(
        self, card: PaperCard, cut: Optional[datetime]
    ) -> Optional[PaperCard]:
        if cut is None:
            return self._pick(self._rows(card.paper_id), None)
        try:
            if card.arxiv_id:
                return self._resolve_arxiv(card.arxiv_id, cut)
        except (ProviderError, NotFound, OfflineError) as exc:
            self.warnings.append(f"{card.paper_id}: excluded ({exc})")
            return None
        return self._pick(self._rows(card.paper_id), cut)

    def _search_local(self, toks, cut, limit):
        scored = []
        for pid in {c.paper_id for c in self.store.all_cards()}:
            c = self._pick(self.store.rows(pid), cut)
            if c is None:
                continue
            hay = " ".join(
                [c.title, c.abstract or "", " ".join(c.authors)]
            ).lower()
            score = sum(t.lower() in hay for t in toks)
            if score:
                scored.append((-score, c.paper_id, c))
        return [c for _, _, c in sorted(scored, key=lambda x: x[:2])[:limit]]

    # ------------------------------------------------------------------- get
    def get(
        self, paper_id: str, *, as_of: Any = None, refresh: bool = False
    ) -> PaperCard:
        """Card for ``paper_id`` (``arxiv:ID[vN]``, ``doi:``, ``openalex:``, ``local:``).

        With ``as_of`` the returned VERSION is the latest one dated on/before the cutoff;
        unknown or later version dates raise ``NotFound``. Cache-first unless ``refresh``.
        """
        cut = self._as_of(as_of)
        pid, ver = canonical_paper_id(paper_id)
        canon = self.store.resolve(pid) or pid
        if pid.startswith("arxiv:") or canon.startswith("arxiv:"):
            base = (pid if pid.startswith("arxiv:") else canon)[6:]
            if ver is not None:
                card = self._arxiv_version(base, ver)
                if cut is not None and (
                    card.version_date is None or card.version_date > cut
                ):
                    raise NotFound(f"{pid}v{ver} is dated after as_of")
                return card
            if refresh and self.online:
                self._arxiv_version(base, None)
            card = self._resolve_arxiv(base, cut)
            if card is None:
                raise NotFound(
                    f"no version of {pid} is known to be public by as_of"
                )
            return card
        rows = self.store.rows(canon)
        if (not rows or refresh) and self.online:
            kind, val = pid.split(":", 1)
            if kind in ("doi", "openalex"):
                url = P.openalex_lookup_url(
                    kind,
                    val if kind == "doi" else val,
                    api_key=self._oa_key,
                    mailto=self._oa_mailto,
                )
                for c in P.parse_openalex_works(
                    self._fetch(url, "openalex", "application/json"), self.now()
                ):
                    self.store.upsert_card(c)
                canon = self.store.resolve(pid) or pid
                rows = self.store.rows(canon)
        if not rows:
            raise (OfflineError if not self.online else NotFound)(
                f"{pid} not found in cache/provider"
            )
        card = self._pick(rows, cut)
        if card is None:
            raise NotFound(
                f"{pid} has no version with a known date on or before as_of"
            )
        return card

    # ---------------------------------------------------------------- content
    def _attach(
        self, card: PaperCard, text: str, original_hash: str
    ) -> PaperCard:
        truncated = len(text) > self.max_text_chars
        stored = text[: self.max_text_chars]
        h = sha256_text(stored)
        self.store.put_content(stored, h, original_hash, truncated)
        upd = card.model_copy(
            update=dict(
                content_hash=h,
                original_hash=original_hash,
                content_chars=len(stored),
                content_truncated=truncated,
                text_status="full_text",
                access_status=AccessStatus.FULL_TEXT,
                claims=[],
            )
        )
        return self.store.save_card(upd)

    def ingest_full_text(
        self, paper_id: str, *, as_of: Any = None
    ) -> PaperCard:
        """Fetch & store this version's full text where permitted; else explicit incomplete status.

        Only arXiv PDFs (version-specific, allowlisted URL built from the validated id)
        are downloaded. Metadata is never turned into content.
        """
        card = self.get(paper_id, as_of=as_of)
        if (
            card.content_hash
            and self.store.get_content(card.content_hash) is not None
        ):
            return card
        if not (card.arxiv_id and card.version):
            # persist onto the stored row: ``card`` may be an as-of view with its abstract stripped
            stored = self.store.get_row(card.paper_id, card.version_key) or card
            saved = self.store.save_card(
                stored.model_copy(
                    update={
                        "text_status": "incomplete:no_permitted_full_text_source"
                    }
                )
            )
            cut = self._as_of(as_of)
            return saved if cut is None else self._point_in_time(saved, cut)
        data = self._fetch(
            P.arxiv_pdf_url(card.arxiv_id, card.version),
            "arxiv",
            "application/pdf",
        )
        orig = hashlib.sha256(data).hexdigest()
        try:
            text = extract_pdf_text(data)
        except PdfUnavailable as exc:
            return self.store.save_card(
                card.model_copy(
                    update={
                        "text_status": f"incomplete:{exc}",
                        "original_hash": orig,
                    }
                )
            )
        return self._attach(card, text, orig)

    def ingest_local(
        self,
        path,
        *,
        title: str,
        first_public_date: datetime,
        version_date: Optional[datetime] = None,
        authors: Iterable[str] = (),
        doi: Optional[str] = None,
        provenance: str = "team-supplied",
    ) -> PaperCard:
        """Register a team-supplied .txt/.md/.pdf file (local paper library backend)."""
        if not title.strip():
            raise PaperValidationError("title required")
        first = ensure_aware(first_public_date, "first_public_date")
        ver = (
            ensure_aware(version_date, "version_date")
            if version_date is not None
            else None
        )
        p = Path(path)
        if p.is_symlink() or not p.is_file():
            raise PaperValidationError(
                "path must be a regular, non-symlink file"
            )
        size = p.stat().st_size
        if size == 0 or size > self.max_body:
            raise PaperValidationError(
                "file is empty or exceeds the size limit"
            )
        raw = p.read_bytes()
        orig = hashlib.sha256(raw).hexdigest()
        text, incomplete = None, None
        if raw.startswith(b"%PDF-"):
            try:
                text = extract_pdf_text(raw)
            except PdfUnavailable as exc:
                incomplete = str(exc)
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PaperValidationError(
                    "local file is neither PDF nor UTF-8 text"
                ) from exc
        card = PaperCard(
            paper_id=f"local:{orig[:16]}",
            title=" ".join(title.split()),
            authors=list(authors),
            doi=normalize_doi(doi) if doi else None,
            version_key="unversioned",
            source="local",
            first_public_date=first,
            version_date=ver,
            urls={},
            retrieved_at=self.now(),
        )
        card = self.store.upsert_card(card)
        if card.content_hash:  # already ingested: keep existing claims
            return card
        if incomplete is not None:
            return self.store.save_card(
                card.model_copy(
                    update={
                        "text_status": f"incomplete:{incomplete}",
                        "original_hash": orig,
                    }
                )
            )
        return self._attach(card, text, orig)

    def text_of(
        self, card: PaperCard, source: str = "full_text"
    ) -> Optional[str]:
        if source == "abstract":
            return card.abstract
        return (
            self.store.get_content(card.content_hash)
            if card.content_hash
            else None
        )

    # ----------------------------------------------------------------- claims
    def _make_claim(
        self, card, kind, text, start, end, source, extraction, claim_text=None
    ) -> Claim:
        src_hash = (
            card.abstract_hash if source == "abstract" else card.content_hash
        )
        quote = text[start:end]
        return Claim(
            claim_id=sha256_text(f"{src_hash}|{kind}|{start}|{end}")[:16],
            kind=kind,
            text=claim_text or quote,
            quote=quote,
            source=source,
            source_hash=src_hash,
            char_start=start,
            char_end=end,
            page=page_of(text, start) if source == "full_text" else None,
            section=section_of(text, start)
            if source == "full_text"
            else "Abstract",
            extraction=extraction,
        )

    def add_claim(
        self,
        paper_id: str,
        *,
        kind: str,
        quote: str,
        text: Optional[str] = None,
        source: str = "full_text",
        as_of: Any = None,
    ) -> PaperCard:
        """Attach a claim grounded in an exact ``quote`` found in the stored source text."""
        card = self.get(paper_id, as_of=as_of)
        body = self.text_of(card, source)
        if not body:
            raise ClaimError(
                f"no stored {source} text for {card.paper_id}; cannot ground a claim"
            )
        if (
            not quote
            or len(quote) > 2000
            or kind not in CLAIM_KINDS
            or (text is not None and len(text) > 2000)
        ):
            raise ClaimError(
                "claim needs a known kind and a quote of 1..2000 characters"
            )
        start = body.find(quote)
        if start < 0:
            raise ClaimError("quote not found verbatim in the source text")
        claim = self._make_claim(
            card, kind, body, start, start + len(quote), source, "manual", text
        )
        claims = [c for c in card.claims if c.claim_id != claim.claim_id] + [
            claim
        ]
        return self.store.save_card(card.model_copy(update={"claims": claims}))

    def extract_claims(self, paper_id: str, *, as_of: Any = None) -> PaperCard:
        """Add heuristic extractive claims (exact sentences; needs human review)."""
        card = self.get(paper_id, as_of=as_of)
        source = "full_text" if self.text_of(card) else "abstract"
        body = self.text_of(card, source)
        if not body:
            raise ClaimError(
                "no text available; metadata alone is never used to create claims"
            )
        new = [
            self._make_claim(card, k, body, s, e, source, "heuristic")
            for k, s, e in heuristic_spans(body)
        ]
        keep = [
            c
            for c in card.claims
            if c.claim_id not in {n.claim_id for n in new}
        ]
        return self.store.save_card(
            card.model_copy(update={"claims": keep + new})
        )

    def verify_claims(self, card: PaperCard) -> list[str]:
        """Return problems; empty list means every claim re-derives from the stored text."""
        problems = []
        for c in card.claims:
            try:
                body = self.text_of(card, c.source)
            except PapersError:  # stored text failed its content hash
                body = None
            want = (
                card.abstract_hash
                if c.source == "abstract"
                else card.content_hash
            )
            if (
                body is None
                or c.source_hash != want
                or sha256_text(body) != want
            ):
                problems.append(f"{c.claim_id}: source text/hash mismatch")
            elif body[c.char_start : c.char_end] != c.quote:
                problems.append(
                    f"{c.claim_id}: quote does not match text offsets"
                )
        return problems

    # -------------------------------------------------- replication / methods
    def create_replication_spec(
        self,
        paper_id: str,
        *,
        requested_label: str,
        as_of: Any = None,
        deviations: Iterable[str] = (),
        comparison_plan: Optional[str] = None,
    ) -> dict:
        """Draft a replication/adaptation spec from grounded claims. No LLM is involved.

        Falls back to INCOMPLETE (never fills gaps) when full text, claims, or evidence are missing.
        """
        label = ReplicationLabel(requested_label)
        devs = [d.strip() for d in deviations if d.strip()]
        card = self.get(paper_id, as_of=as_of)
        reasons: list[str] = []
        if label is ReplicationLabel.EXACT_REPLICATION and devs:
            raise PaperValidationError(
                "EXACT_REPLICATION cannot declare deviations; use CLOSE_REPLICATION"
            )
        if label is ReplicationLabel.CLOSE_REPLICATION and not (
            devs and comparison_plan
        ):
            raise PaperValidationError(
                "CLOSE_REPLICATION requires deviations and a comparison_plan"
            )
        need = REQUIRED_FOR.get(label, ())
        if label is not ReplicationLabel.INCOMPLETE:
            if card.text_status != "full_text" or card.content_truncated:
                reasons.append(
                    f"full text unavailable or truncated (text_status={card.text_status})"
                )
            bad = self.verify_claims(card)
            if bad:
                reasons.append("claim verification failed: " + "; ".join(bad))
        have = {c.kind for c in card.claims}
        missing = [k for k in need if k not in have]
        if missing:
            reasons.append(
                "no source-grounded claim for: " + ", ".join(missing)
            )
        final = (
            label
            if not reasons and label is not ReplicationLabel.INCOMPLETE
            else ReplicationLabel.INCOMPLETE
        )
        evidence = {
            k: [
                {
                    "claim_id": c.claim_id,
                    "page": c.page,
                    "section": c.section,
                    "quote": c.quote,
                    "extraction": c.extraction,
                }
                for c in card.claims
                if c.kind == k
            ]
            for k in sorted(have)
        }
        body = {
            "paper_id": card.paper_id,
            "version_key": card.version_key,
            "card_hash": card.card_hash,
            "content_hash": card.content_hash,
            "version_date": card.version_date.isoformat()
            if card.version_date
            else None,
            "research_mode": "contemporary"
            if as_of is None
            else "historical_replay",
            "as_of": self._as_of(as_of).isoformat()
            if as_of is not None
            else None,
            "requested_label": label.value,
            "label": final.value,
            "missing_claim_kinds": missing,
            "reasons": reasons,
            "deviations": devs,
            "comparison_plan": comparison_plan,
            "evidence": evidence,
            "status": "draft_requires_human_review",
            "llm_called": False,
        }
        spec_id = sha256_text(canonical_json(body))[:16]
        body = {
            "spec_id": spec_id,
            **body,
            "created_at": self.now().isoformat(),
        }
        self.store.put_artifact("replication_spec", spec_id, body, self.now())
        return body

    def create_method_card(
        self,
        method_id: str,
        version: str,
        *,
        paper_id: Optional[str] = None,
        supported_targets: Iterable[str] = (),
        min_history: Optional[str] = None,
        assumptions: Iterable[str] = (),
        baselines: Iterable[str] = (),
        failure_modes: Iterable[str] = (),
        reference_implementation: Optional[str] = None,
    ) -> dict:
        """Persist an immutable method card; the source paper is referenced by card hash."""
        if not re.fullmatch(
            r"[a-z][a-z0-9_\-]{1,63}", method_id
        ) or not re.fullmatch(r"[0-9A-Za-z.\-]{1,32}", version):
            raise PaperValidationError("invalid method_id/version")
        card = self.get(paper_id) if paper_id else None
        body = {
            "method_id": method_id,
            "version": version,
            "supported_targets": list(supported_targets),
            "min_history": min_history,
            "assumptions": list(assumptions),
            "baselines": list(baselines),
            "failure_modes": list(failure_modes),
            "reference_implementation": reference_implementation,
            "source": {
                "paper_id": card.paper_id,
                "card_hash": card.card_hash,
                "version_key": card.version_key,
            }
            if card
            else None,
            "status": "draft",
            "created_at": self.now().isoformat(),
        }
        key = f"{method_id}@{version}"
        old = self.store.get_artifact("method_card", key)
        if old is not None:
            raise PaperValidationError(
                f"method card {key} already exists (cards are immutable)"
            )
        self.store.put_artifact("method_card", key, body, self.now())
        return body


__all__ = ["PaperService", "SearchResult", "PapersError"]
