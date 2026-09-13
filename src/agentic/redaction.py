"""Strict, stable redaction before any filing reaches a language model."""
from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from typing import Any

from .contracts import RedactedFiling

REDACTION_VERSION = "redaction-v1"

_URL_RE = re.compile(r"\bhttps?://[^\s)]+", re.IGNORECASE)
_ACCESSION_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
_DATE_RES = (
    re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2}\b"),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2}(?:,?\s+(?:19|20)\d{2})?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:19|20)\d{2}\b"),
)

_IDENTITY_FIELDS = (
    "company_name",
    "content_company_name",
    "historical_crsp_company_name",
    "provider_company_name",
    "ticker",
    "provider_ticker",
    "cik",
    "cusip",
    "cusip8",
    "isin",
    "figi",
    "composite_figi",
    "share_class_figi",
    "gvkey",
    "iid",
    "gvkey_iid",
    "permno",
    "permco",
    "document_id",
)

_HIDDEN_METADATA_FIELDS = {
    *_IDENTITY_FIELDS,
    "sector",
    "industry",
    "exchange",
    "country",
    "region",
    "product_name",
    "report_url",
    "accession_number",
    "filing_date",
    "filing_date_raw",
    "filing_date_utc_raw",
    "content_report_date",
    "content_report_date_raw",
    "content_report_date_basis",
    "metadata_json",
}


def stable_opaque_id(document_id: str, secret: str) -> str:
    """Return a stable opaque ID without exposing the underlying document ID."""
    if not secret:
        raise ValueError("a non-empty redaction secret is required")
    digest = hmac.new(
        secret.encode("utf-8"), document_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:20]
    return f"DOC_{digest.upper()}"


def _replacement_values(filing: Mapping[str, Any]) -> list[str]:
    values = []
    for key in _IDENTITY_FIELDS:
        value = filing.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if len(text) >= 2:
            values.append(text)
    return sorted(set(values), key=lambda value: (-len(value), value.lower()))


def _mask_known_values(text: str, values: list[str]) -> str:
    for value in values:
        pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)
        text = pattern.sub("<IDENTIFIER>", text)
    return text


def _mask_patterns(text: str) -> str:
    text = _URL_RE.sub("<URL>", text)
    text = _ACCESSION_RE.sub("<ACCESSION>", text)
    for pattern in _DATE_RES:
        text = pattern.sub("<DATE>", text)
    return text


def _visible_metadata(filing: Mapping[str, Any]) -> dict[str, Any]:
    items = filing.get("items") or []
    if not isinstance(items, (list, tuple)):
        items = [str(items)]
    metadata: dict[str, Any] = {
        "form_type": str(filing.get("form_type") or "unknown"),
        "items": [str(item) for item in items],
        "is_amendment": bool(filing.get("is_amendment", False)),
    }
    text = str(filing.get("text") or "")
    metadata["text_chars_bucket"] = min(len(text) // 1000, 100)
    return metadata


def redact_filing(filing: Mapping[str, Any], secret: str) -> RedactedFiling:
    document_id = str(filing.get("document_id") or "")
    if not document_id:
        raise ValueError("filing requires document_id")
    text = str(filing.get("text") or "")
    text = _mask_known_values(text, _replacement_values(filing))
    text = _mask_patterns(text)
    filing_date = str(filing.get("filing_date") or "")
    eom = str(filing.get("eom") or "")
    if not eom and filing_date:
        eom = filing_date[:7]
    return RedactedFiling(
        document_id=document_id,
        opaque_id=stable_opaque_id(document_id, secret),
        permno=int(filing["permno"]) if filing.get("permno") is not None else None,
        filing_date=filing_date,
        eom=eom,
        text=text,
        visible_metadata=_visible_metadata(filing),
        redaction_version=REDACTION_VERSION,
    )
