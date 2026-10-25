"""Academic paper service (spec section 10, with 20.3/20.5 boundaries).

Public API (all in ``pod_platform.papers``)::

    store = PaperStore("papers.sqlite")            # persistent metadata + content cache
    svc = PaperService(store, clock=lambda: datetime.now(timezone.utc),
                       transport=UrllibTransport(), online=True,   # online is opt-in
                       openalex_api_key=os.environ.get("OPENALEX_API_KEY"))

    svc.search("post earnings drift", as_of="2017-06-30", limit=10) -> SearchResult
    svc.get("arxiv:1706.03762", as_of=None) -> PaperCard        # also doi:/openalex:/local:
    svc.ingest_full_text("arxiv:1706.03762v1") -> PaperCard      # PDF via optional pypdf
    svc.ingest_local(path, title=..., first_public_date=...) -> PaperCard
    svc.add_claim(id, kind="universe", quote="exact text") -> PaperCard
    svc.extract_claims(id) -> PaperCard                          # heuristic, extractive only
    svc.create_replication_spec(id, requested_label="CLOSE_REPLICATION", ...) -> dict
    svc.create_method_card("my_method", "1", paper_id=id, ...) -> dict

Guarantees
* Modes: ``as_of=None`` is contemporary research; ``as_of`` is historical replay.
  A version qualifies only if its own version date is known and <= as_of; an old
  first-publication date never admits a later revision. Unknown dates are excluded.
  arXiv versions are located by bisection over versioned lookups (monotone dates);
  offline, a version is accepted only if the cache proves no newer one existed.
* Time: clocks and dates are timezone aware; naive/future ``as_of`` raises ``TemporalError``.
* Network: nothing is fetched unless ``online=True`` and a transport is supplied. URLs must
  be https on ``export.arxiv.org``, ``arxiv.org`` or ``api.openalex.org`` (no IPs, userinfo,
  odd ports); redirects are re-validated; bodies and timeouts are bounded.
* Content: full text is only attached from real extracted text (never from metadata); if
  pypdf or permission is missing the card says ``text_status="incomplete:<reason>"``.
  Claims carry an exact quote, char offsets, page/section and the source hash, and are
  re-verified (``verify_claims``). Replication specs fall back to ``INCOMPLETE``.
* Credentials: the OpenAlex key is only added to request URLs, never stored or logged.
* No LLM is called anywhere in this package.
"""

from .models import (AccessStatus, AuthError, Claim, ClaimError, NotFound, OfflineError,
                     PaperCard, PaperValidationError, PapersError, ProviderError, RateLimitError,
                     ReplicationLabel, ResponseTooLarge, TemporalError, UrlNotAllowed,
                     canonical_paper_id)
from .service import PaperService, SearchResult
from .store import PaperStore
from .transport import ALLOWED_HOSTS, Transport, TransportResponse, UrllibTransport, validate_url

__all__ = [
    "AccessStatus", "AuthError", "Claim", "ClaimError", "NotFound", "OfflineError", "PaperCard",
    "PaperService", "PaperStore", "PaperValidationError", "PapersError", "ProviderError",
    "RateLimitError", "ReplicationLabel", "ResponseTooLarge", "SearchResult", "TemporalError",
    "Transport", "TransportResponse", "UrlNotAllowed", "UrllibTransport", "ALLOWED_HOSTS",
    "canonical_paper_id", "validate_url",
]
