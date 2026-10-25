"""HTTP transport: allowlisted HTTPS hosts, finite timeouts, bounded bodies.

``Transport`` is the injection point: tests pass a fake with
``get(url, *, headers, timeout, max_bytes) -> TransportResponse``.
``validate_url`` is applied by the service before ANY transport call, so even a
fake transport can never be asked for a non-allowlisted URL.
"""

from __future__ import annotations

import ipaddress
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Protocol
from urllib.parse import urlsplit

from .models import (AuthError, ProviderError, RateLimitError, ResponseTooLarge,
                     UrlNotAllowed)

ALLOWED_HOSTS = frozenset({"export.arxiv.org", "arxiv.org", "api.openalex.org"})
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 25 * 1024 * 1024
MAX_REDIRECTS = 3


@dataclass(frozen=True)
class TransportResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def get(self, url: str, *, headers: Mapping[str, str], timeout: float,
            max_bytes: int) -> TransportResponse: ...


def validate_url(url: str, allowed_hosts: frozenset = ALLOWED_HOSTS) -> str:
    """Return ``url`` if it is https, on an allowlisted host, without userinfo/odd port."""
    if not isinstance(url, str) or len(url) > 2000 or any(c in url for c in "\r\n\t \x00"):
        raise UrlNotAllowed("malformed URL")
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise UrlNotAllowed("only https URLs are allowed")
    host = (parts.hostname or "").lower()
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UrlNotAllowed("userinfo in URL is not allowed")
    try:
        ipaddress.ip_address(host)
        raise UrlNotAllowed("IP-literal hosts are not allowed")
    except ValueError:
        pass
    if host not in allowed_hosts:
        raise UrlNotAllowed(f"host {host!r} is not an allowlisted provider host")
    try:
        port = parts.port
    except ValueError as exc:
        raise UrlNotAllowed("invalid port") from exc
    if port not in (None, 443):
        raise UrlNotAllowed("non-default port is not allowed")
    return url


def redact_url(url: str) -> str:
    """Drop the query string (may hold an API key) for messages and logs."""
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}{p.path}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # handled manually so each hop is validated
        return None


class UrllibTransport:
    """Real transport. Only constructed when the caller explicitly goes online."""

    def __init__(self, *, user_agent: str = "fiam-papers/1.0 (research; contact via repo)",
                 min_interval: float = 3.0, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic):
        self.user_agent = user_agent
        self.min_interval = min_interval
        self._sleep, self._mono = sleep, monotonic
        self._last: dict[str, float] = {}
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _throttle(self, host: str) -> None:
        last = self._last.get(host)
        if last is not None:
            wait = self.min_interval - (self._mono() - last)
            if wait > 0:
                self._sleep(wait)
        self._last[host] = self._mono()

    def get(self, url, *, headers, timeout, max_bytes):
        for _ in range(MAX_REDIRECTS + 1):
            validate_url(url)
            self._throttle(urlsplit(url).hostname or "")
            hdrs = {"User-Agent": self.user_agent, **dict(headers)}
            req = urllib.request.Request(url, headers=hdrs, method="GET")
            try:
                resp = self._opener.open(req, timeout=timeout)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                    from urllib.parse import urljoin
                    url = urljoin(url, e.headers["Location"])
                    continue
                return TransportResponse(e.code, b"", dict(e.headers or {}))
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise ProviderError(f"network error for {redact_url(url)}: {type(e).__name__}") from e
            with resp:
                chunks, total = [], 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ResponseTooLarge(f"response exceeds {max_bytes} bytes")
                    chunks.append(chunk)
                return TransportResponse(resp.status, b"".join(chunks), dict(resp.headers))
        raise ProviderError("too many redirects")


def check_status(resp: TransportResponse, url: str, provider: str) -> None:
    """Raise a precise error for non-200 responses."""
    if resp.status == 200:
        return
    where = redact_url(url)
    if resp.status in (401, 403):
        hint = (" Set OPENALEX_API_KEY (and optionally OPENALEX_MAILTO)." if provider == "openalex"
                else "")
        raise AuthError(f"{provider} rejected the request ({resp.status}) at {where}.{hint}", resp.status)
    if resp.status == 429:
        raise RateLimitError(f"{provider} rate limit hit at {where}; retry later", 429)
    raise ProviderError(f"{provider} returned HTTP {resp.status} at {where}", resp.status)
