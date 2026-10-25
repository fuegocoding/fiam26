"""Provider backends: URL builders and strict parsers for arXiv (Atom) and OpenAlex (JSON).

Parsers are pure (bytes in, ``PaperCard`` out) so they are testable offline.
Nothing here performs I/O; ``PaperService`` owns transport, caching and policy.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Optional
from urllib.parse import quote, urlencode

from .models import (
    ARXIV_ID_RE,
    ARXIV_VERSIONED_RE,
    AccessStatus,
    PaperCard,
    PaperValidationError,
    ProviderError,
    TemporalError,
    ensure_aware,
    normalize_doi,
    parse_instant,
    sha256_text,
)

ARXIV_QUERY = "https://export.arxiv.org/api/query"
OPENALEX_WORKS = "https://api.openalex.org/works"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
MAX_QUERY_CHARS = 300
MAX_AUTHORS = 200
MAX_ABSTRACT_CHARS = 20000


def _ws(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def clean_query(query: str) -> list[str]:
    if (
        not isinstance(query, str)
        or not query.strip()
        or len(query) > MAX_QUERY_CHARS
    ):
        raise PaperValidationError(
            f"query must be 1..{MAX_QUERY_CHARS} characters"
        )
    toks = re.findall(r"[\w\-]+", query, flags=re.UNICODE)[:12]
    if not toks:
        raise PaperValidationError("query has no searchable terms")
    return toks


# ------------------------------------------------------------------- arXiv
def arxiv_search_url(
    query: str, limit: int, as_of: Optional[datetime] = None
) -> str:
    terms = " AND ".join(f"all:{t}" for t in clean_query(query))
    if (
        as_of is not None
    ):  # v1 submission must be on/before cutoff (necessary, not sufficient)
        terms += f" AND submittedDate:[190001010000 TO {as_of.strftime('%Y%m%d%H%M')}]"
    q = urlencode(
        {
            "search_query": terms,
            "start": 0,
            "max_results": int(limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        quote_via=quote,
    )
    return f"{ARXIV_QUERY}?{q}"


def arxiv_id_url(arxiv_id: str, version: Optional[int] = None) -> str:
    if not ARXIV_ID_RE.match(arxiv_id):
        raise PaperValidationError("invalid arXiv id")
    ident = f"{arxiv_id}v{int(version)}" if version else arxiv_id
    return f"{ARXIV_QUERY}?{urlencode({'id_list': ident, 'max_results': 1}, safe='/')}"


def arxiv_pdf_url(arxiv_id: str, version: int) -> str:
    if not ARXIV_ID_RE.match(arxiv_id) or int(version) < 1:
        raise PaperValidationError("invalid arXiv id/version")
    return f"https://arxiv.org/pdf/{arxiv_id}v{int(version)}"


def parse_arxiv_feed(body: bytes, retrieved_at: datetime) -> list[PaperCard]:
    if b"<!DOCTYPE" in body[:4096].upper() or b"<!ENTITY" in body.upper():
        raise ProviderError(
            "arXiv response contains a DTD/entity declaration; rejected"
        )
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ProviderError(f"arXiv response is not valid XML: {exc}") from exc
    if root.tag != f"{ATOM}feed":
        raise ProviderError("arXiv response is not an Atom feed")
    cards = []
    for e in root.findall(f"{ATOM}entry"):
        raw_id = _ws(e.findtext(f"{ATOM}id"))
        if "/api/errors" in raw_id:
            raise ProviderError(
                "arXiv API error: "
                + (_ws(e.findtext(f"{ATOM}summary")) or "unknown")
            )
        m = re.match(r"^https?://arxiv\.org/abs/(.+)$", raw_id)
        vm = ARXIV_VERSIONED_RE.match(m.group(1)) if m else None
        if not vm:
            raise ProviderError(f"unparseable arXiv entry id {raw_id[:80]!r}")
        base, ver = vm.group(1), int(vm.group(2))
        title = _ws(e.findtext(f"{ATOM}title"))
        if not title:
            raise ProviderError(f"arXiv entry {base} has no title")
        try:
            first = parse_instant(
                _ws(e.findtext(f"{ATOM}published")), name="published"
            )
            upd = parse_instant(
                _ws(e.findtext(f"{ATOM}updated")), name="updated"
            )
        except TemporalError as exc:
            raise ProviderError(f"arXiv entry {base}: {exc}") from exc
        if upd < first:
            raise ProviderError(
                f"arXiv entry {base}: updated precedes published"
            )
        authors = [
            _ws(a.findtext(f"{ATOM}name")) for a in e.findall(f"{ATOM}author")
        ][:MAX_AUTHORS]
        abstract = (
            _ws(e.findtext(f"{ATOM}summary"))[:MAX_ABSTRACT_CHARS] or None
        )
        doi = None
        raw_doi = _ws(e.findtext(f"{ARXIV_NS}doi"))
        if raw_doi:
            try:
                doi = normalize_doi(raw_doi)
            except PaperValidationError:
                doi = None
        cards.append(
            PaperCard(
                paper_id=f"arxiv:{base}",
                title=title,
                authors=[a for a in authors if a],
                doi=doi,
                arxiv_id=base,
                version=ver,
                latest_version=None,
                version_key=f"v{ver}",
                source="arxiv",
                first_public_date=first,
                version_date=upd,
                abstract=abstract,
                abstract_hash=sha256_text(abstract) if abstract else None,
                access_status=AccessStatus.ABSTRACT
                if abstract
                else AccessStatus.METADATA_ONLY,
                urls={
                    "abstract": f"https://arxiv.org/abs/{base}v{ver}",
                    "pdf": arxiv_pdf_url(base, ver),
                },
                text_status="abstract_only" if abstract else "none",
                retrieved_at=retrieved_at,
            )
        )
    return cards


# ---------------------------------------------------------------- OpenAlex
def openalex_search_url(
    query: str,
    limit: int,
    as_of: Optional[datetime] = None,
    *,
    api_key: Optional[str] = None,
    mailto: Optional[str] = None,
) -> str:
    params: dict[str, Any] = {
        "search": " ".join(clean_query(query)),
        "per-page": int(limit),
    }
    if as_of is not None:
        params["filter"] = f"to_publication_date:{as_of.date().isoformat()}"
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key
    return f"{OPENALEX_WORKS}?{urlencode(params, quote_via=quote)}"


def openalex_lookup_url(
    kind: str,
    value: str,
    *,
    api_key: Optional[str] = None,
    mailto: Optional[str] = None,
) -> str:
    """Single-work lookup through the filter query (identifier is a query value, not a path)."""
    if kind == "doi":
        flt = f"doi:https://doi.org/{normalize_doi(value)}"
    elif kind == "openalex":
        if not re.fullmatch(r"W\d{1,15}", value):
            raise PaperValidationError("invalid OpenAlex id")
        flt = f"openalex_id:{value}"
    else:
        raise PaperValidationError(f"unsupported lookup kind {kind!r}")
    params: dict[str, Any] = {"filter": flt, "per-page": 1}
    if mailto:
        params["mailto"] = mailto
    if api_key:
        params["api_key"] = api_key
    return f"{OPENALEX_WORKS}?{urlencode(params, quote_via=quote)}"


def _abstract_from_index(idx: Any) -> Optional[str]:
    if not isinstance(idx, dict) or not idx:
        return None
    slots: dict[int, str] = {}
    for word, positions in list(idx.items())[:5000]:
        if not isinstance(word, str) or not isinstance(positions, list):
            return None
        for p in positions[:200]:
            if not isinstance(p, int) or p < 0 or p > 20000:
                return None
            slots[p] = word
    text = " ".join(slots[i] for i in sorted(slots))
    return text[:MAX_ABSTRACT_CHARS] or None


def _safe_url(u: Any) -> Optional[str]:
    if (
        isinstance(u, str)
        and u.startswith(("https://", "http://"))
        and len(u) <= 500
        and not re.search(r"[\s\x00-\x1f]", u)
    ):
        return u
    return None


def parse_openalex_works(
    body: bytes, retrieved_at: datetime
) -> list[PaperCard]:
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderError(
            f"OpenAlex response is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ProviderError("OpenAlex response lacks a 'results' list")
    cards = []
    for w in data["results"]:
        if not isinstance(w, dict):
            raise ProviderError("OpenAlex result is not an object")
        m = re.match(
            r"^https://openalex\.org/(W\d{1,15})$", str(w.get("id", ""))
        )
        if not m:
            raise ProviderError("OpenAlex result has invalid id")
        wid = m.group(1)
        title = _ws(w.get("display_name") or w.get("title"))
        if not title:
            raise ProviderError(f"OpenAlex work {wid} has no title")
        doi = None
        if w.get("doi"):
            try:
                doi = normalize_doi(str(w["doi"]))
            except PaperValidationError:
                doi = None
        arxiv_id = None
        if doi and doi.startswith("10.48550/arxiv."):
            cand = doi[len("10.48550/arxiv.") :]
            arxiv_id = cand if ARXIV_ID_RE.match(cand) else None
        locs = [
            location
            for location in (w.get("locations") or [])
            if isinstance(location, dict)
        ]
        primary = (
            w.get("primary_location")
            if isinstance(w.get("primary_location"), dict)
            else {}
        )
        for loc in [primary] + locs:
            mm = re.match(
                r"^https?://arxiv\.org/abs/([^v?#]+)",
                str(loc.get("landing_page_url") or ""),
            )
            if mm and ARXIV_ID_RE.match(mm.group(1)) and arxiv_id is None:
                arxiv_id = mm.group(1)
        first = None
        if w.get("publication_date"):
            try:
                first = parse_instant(
                    str(w["publication_date"]), name="publication_date"
                )
            except TemporalError as exc:
                raise ProviderError(f"OpenAlex work {wid}: {exc}") from exc
        # Only a published version of record has a trustworthy, fixed version date.
        is_vor = primary.get("version") == "publishedVersion" and w.get(
            "type"
        ) not in ("preprint", "posted-content")
        abstract = _abstract_from_index(w.get("abstract_inverted_index"))
        authors = []
        for a in (w.get("authorships") or [])[:MAX_AUTHORS]:
            name = _ws(((a or {}).get("author") or {}).get("display_name"))
            if name:
                authors.append(name)
        urls = {}
        for key, val in (
            ("landing", primary.get("landing_page_url")),
            ("pdf_hint", primary.get("pdf_url")),
            ("openalex", f"https://openalex.org/{wid}"),
        ):
            if _safe_url(val):
                urls[key] = val
        if doi:
            urls["doi"] = f"https://doi.org/{doi}"
        pid = (
            f"arxiv:{arxiv_id}"
            if arxiv_id
            else (f"doi:{doi}" if doi else f"openalex:{wid}")
        )
        cards.append(
            PaperCard(
                paper_id=pid,
                title=title,
                authors=authors,
                doi=doi,
                arxiv_id=arxiv_id,
                version=None,
                version_key="pub" if is_vor else "unversioned",
                openalex_id=wid,
                source="openalex",
                first_public_date=first,
                version_date=first if is_vor else None,
                abstract=abstract,
                abstract_hash=sha256_text(abstract) if abstract else None,
                access_status=AccessStatus.ABSTRACT
                if abstract
                else AccessStatus.METADATA_ONLY,
                urls=urls,
                text_status="abstract_only" if abstract else "none",
                retrieved_at=ensure_aware(retrieved_at),
            )
        )
    return cards
