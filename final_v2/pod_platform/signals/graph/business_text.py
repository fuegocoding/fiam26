"""Dated business descriptions and event text extracted from 8-K filings."""

import pathlib
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_BUSINESS_TERMS = (
    "provider",
    "provides",
    "platform",
    "service",
    "services",
    "product",
    "products",
    "manufacturer",
    "manufactures",
    "develops",
    "operates",
    "customers",
    "clients",
    "retail",
    "software",
    "technology",
    "healthcare",
    "financial",
    "energy",
    "market",
    "business",
    "solutions",
    "network",
    "brands",
    "stores",
    "distribution",
)
_SPECIFIC_TERMS = (
    "software",
    "platform",
    "network",
    "stores",
    "retail",
    "customers",
    "clients",
    "manufacturer",
    "manufactures",
    "healthcare",
    "payments",
    "cloud",
    "energy",
    "products",
    "services",
    "solutions",
    "distribution",
    "subscription",
)
_STOP_MARKERS = (
    "forward-looking statements",
    "forward looking statements",
    "safe harbor",
    "investor contact",
    "media contact",
    "contacts:",
    "item 9.01",
    "signature",
    "source:",
    "for more information",
    "###",
)
_BOILERPLATE = (
    "broad, non-exclusionary distribution",
    "deemed to be material information",
    "through various channels",
    "involve risks and uncertainties",
    "estimated quantities of crude oil",
    "actual results may differ materially",
)
_REJECTED_SENTENCES = (
    "securities and exchange commission",
    "forward-looking",
    "safe harbor",
    "signature",
    "exhibit",
    "registrant",
    "telephone number",
)
_REJECTED_HEADINGS = (
    "about the transaction",
    "about the merger",
    "about the offering",
    "about this",
    "about our",
    "about its",
    "about forward",
)
_GENERIC_NAME_WORDS = {
    "company",
    "corporation",
    "incorporated",
    "holdings",
    "limited",
}
_SPACE = re.compile(r"\s+")
_SUMMARY = re.compile(
    r"\b(?:is|are)\s+(?:a|an|the)\s+(?:leading|global|premier|major|provider|"
    r"manufacturer|operator|technology|financial|healthcare|specialty|"
    r"diversified)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def build_profiles(
    filings_path: pathlib.Path, output_path: pathlib.Path
) -> pathlib.Path:
    """Writes one dated text record per filing, available at its filing date.

    Args:
        filings_path: Parquet of identified 8-K filings with full text.
        output_path: Destination parquet; reused when it already exists.

    Returns:
        The output path.
    """
    if output_path.exists():
        return output_path
    schema = pa.schema(
        [
            ("permco", pa.int64()),
            ("document_id", pa.string()),
            ("available_at", pa.timestamp("us", tz="UTC")),
            ("business_description", pa.string()),
            ("event_text", pa.string()),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "permco",
        "document_id",
        "filing_date",
        "company_name",
        "title",
        "items",
        "text",
    ]
    end_of_day = pd.Timedelta(hours=23, minutes=59, seconds=59)
    with pq.ParquetWriter(output_path, schema, compression="zstd") as writer:
        source = pq.ParquetFile(filings_path)
        for batch in source.iter_batches(batch_size=2048, columns=columns):
            records = []
            for filing in batch.to_pylist():
                name = filing["company_name"]
                description = extract_business_description(filing["text"], name)
                records.append(
                    {
                        "permco": filing["permco"],
                        "document_id": filing["document_id"],
                        "available_at": (
                            pd.Timestamp(filing["filing_date"], tz="UTC")
                            + end_of_day
                        ),
                        "business_description": description
                        or extract_durable_business_language(
                            filing["text"], name
                        ),
                        "event_text": event_text(
                            filing["title"],
                            filing["items"],
                            filing["text"],
                            name,
                        ),
                    }
                )
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
    return output_path


def extract_business_description(
    text: str | None, company_name: str | None = None
) -> str | None:
    """Issuer profile, preferring the latest "About <issuer>" footer."""
    compact = _clean(text)
    if not compact:
        return None
    lower = compact.lower()
    end = len(compact)
    for _ in range(8):
        position = lower.rfind("about ", 0, end)
        if position < 0:
            break
        candidate = _truncate_section(compact[position:])
        heading = candidate[:120].lower()
        if not heading.startswith(_REJECTED_HEADINGS) and _looks_like_profile(
            candidate
        ):
            return _remove_issuer_name(candidate, company_name)
        end = position
    # Fall back to a summary sentence near the end that names the issuer.
    tail = compact[max(0, len(compact) - 18000) :]
    name_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z]{4,}", company_name or "")
        if token.lower()
        not in {"corporation", "company", "incorporated", "holdings"}
    ]
    for match in reversed(list(_SUMMARY.finditer(tail))):
        start = max(0, match.start() - 180)
        candidate = _truncate_section(tail[start:], limit=1200)
        prefix = candidate[:260].lower()
        names_issuer = not name_tokens or any(t in prefix for t in name_tokens)
        if names_issuer and _looks_like_profile(candidate):
            return _remove_issuer_name(candidate, company_name)
    return None


def extract_durable_business_language(
    text: str | None, company_name: str | None = None, limit: int = 1800
) -> str:
    """Highest-scoring business-model sentences, kept in document order."""
    compact = _clean(text)
    if not compact:
        return ""
    body = compact[min(900, len(compact)) :]
    candidates = []
    for position, sentence in enumerate(_SENTENCE.split(body)):
        sentence = sentence.strip()
        lower = sentence.lower()
        if not 80 <= len(sentence) <= 700 or any(
            term in lower for term in _REJECTED_SENTENCES
        ):
            continue
        base_hits = sum(term in lower for term in _BUSINESS_TERMS)
        specific_hits = sum(term in lower for term in _SPECIFIC_TERMS)
        score = base_hits + 2 * specific_hits
        if score >= 4:
            candidates.append((score, -position, sentence))
    selected = [sentence for _, _, sentence in sorted(candidates, reverse=True)]
    selected = selected[:6][::-1]
    return _remove_issuer_name(" ".join(selected)[:limit], company_name)


def event_text(
    title: str | None,
    items: list[str] | None,
    text: str | None = None,
    company_name: str | None = None,
) -> str:
    """Filing title, form items and an excerpt of the first item's text."""
    title_value = _clean(title)
    item_values = sorted(
        {str(item).strip() for item in (items or []) if str(item).strip()}
    )
    compact = _clean(text)
    lower = compact.lower()
    item_start = lower.find("item ", min(700, len(lower)))
    if item_start >= 0:
        excerpt = compact[item_start : item_start + 1400]
    else:
        excerpt = compact[900:2300]
    excerpt = _remove_issuer_name(excerpt, company_name)
    parts = [title_value, *[f"form item {item}" for item in item_values]]
    return " ".join([*parts, excerpt]).strip()


def _clean(value: str | None) -> str:
    return _SPACE.sub(" ", value or "").strip()


def _truncate_section(text: str, limit: int = 1400) -> str:
    section = text[:limit]
    lower = section.lower()
    cuts = [lower.find(marker) for marker in _STOP_MARKERS]
    cuts = [cut for cut in cuts if cut >= 160]
    if cuts:
        section = section[: min(cuts)]
    return section.strip(" :-–—|.;")


def _looks_like_profile(text: str) -> bool:
    lower = text.lower()
    if any(marker in lower for marker in _BOILERPLATE):
        return False
    hits = sum(term in lower for term in _BUSINESS_TERMS)
    return len(text) >= 160 and hits >= 3


def _remove_issuer_name(text: str, company_name: str | None) -> str:
    """Masks the issuer name so it cannot act as a similarity feature."""
    if not company_name:
        return text
    cleaned = re.sub(
        re.escape(_clean(company_name)), " issuer ", text, flags=re.IGNORECASE
    )
    meaningful = [
        token
        for token in re.findall(r"[A-Za-z]{4,}", company_name)
        if token.lower() not in _GENERIC_NAME_WORDS
    ]
    for token in meaningful:
        cleaned = re.sub(
            rf"\b{re.escape(token)}\b", " issuer ", cleaned, flags=re.IGNORECASE
        )
    return _clean(cleaned)
