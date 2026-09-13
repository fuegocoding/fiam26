"""Strict, stable redaction before any filing reaches a language model."""
from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from typing import Any

from .contracts import RedactedFiling

REDACTION_VERSION = "redaction-v2"

_URL_RE = re.compile(r"\bhttps?://[^\s)]+", re.IGNORECASE)
_ACCESSION_RE = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
_EXCHANGE_TICKER_RE = re.compile(
    r"\b(?:NASDAQ|NYSE(?:\s+American)?|AMEX|OTC|TSX|LSE|ASX)\s*"
    r"(?::|/|\(|\b)\s*[A-Z][A-Z0-9.-]{1,9}\)?\b",
    re.IGNORECASE,
)
_LABELED_IDENTIFIER_RE = re.compile(
    r"\b(?:CIK|CUSIP|ISIN|FIGI|LEI|GVKEY|PERMNO|PERMCO)\s*"
    r"(?:No\.?|Number)?\s*(?:[:=#]\s*)?[A-Z0-9-]{5,}\b",
    re.IGNORECASE,
)
_DATE_RES = (
    re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b(?:19|20)\d{2}[.]\d{1,2}[.]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:19|20)\d{2}\b"),
    re.compile(r"\b(?:19|20)\d{6}\b"),
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
            if key in {"company_name", "content_company_name", "provider_company_name"}:
                tokens = re.findall(r"[A-Za-z0-9&'-]+", text)
                suffixes = {
                    "inc",
                    "incorporated",
                    "corp",
                    "corporation",
                    "co",
                    "company",
                    "limited",
                    "ltd",
                    "llc",
                    "lp",
                    "plc",
                    "sa",
                    "nv",
                }
                while len(tokens) > 1 and tokens[-1].lower().rstrip(".") in suffixes:
                    tokens.pop()
                if tokens and tokens[0].lower() not in {"the", "a", "an"}:
                    values.append(" ".join(tokens))
    return sorted(set(values), key=lambda value: (-len(value), value.lower()))


def _mask_known_values(text: str, values: list[str]) -> str:
    for value in values:
        pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)
        text = pattern.sub("<IDENTIFIER>", text)
    return text


def _mask_patterns(text: str) -> str:
    text = _URL_RE.sub("<URL>", text)
    text = _ACCESSION_RE.sub("<ACCESSION>", text)
    text = _EXCHANGE_TICKER_RE.sub("<TICKER>", text)
    text = _LABELED_IDENTIFIER_RE.sub("<IDENTIFIER>", text)
    for pattern in _DATE_RES:
        text = pattern.sub("<DATE>", text)
    return text


def _visible_metadata(filing: Mapping[str, Any]) -> dict[str, Any]:
    items = filing.get("items")
    if items is None:
        items = []
    if isinstance(items, (str, bytes)):
        items = [items]
    elif not isinstance(items, (list, tuple)):
        try:
            items = list(items)
        except TypeError:
            items = [items]
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
