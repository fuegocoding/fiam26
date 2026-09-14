"""Extract point-in-time business and event text evidence from historical 8-Ks."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_FILINGS = ROOT / "data/raw/8k_20150101_20260831_identified.parquet"
DEFAULT_OUTPUT = ROOT / "output/business_text_evidence.parquet"
PROFILE_VERSION = "8k-about-and-business-summary-v1"

_BUSINESS_TERMS = (
    "provider", "provides", "platform", "service", "services", "product", "products",
    "manufacturer", "manufactures", "develops", "operates", "customers", "clients",
    "retail", "software", "technology", "healthcare", "financial", "energy", "market",
    "business", "solutions", "network", "brands", "stores", "distribution",
)
_STOP_MARKERS = (
    "forward-looking statements", "forward looking statements", "safe harbor",
    "investor contact", "media contact", "contacts:", "item 9.01", "signature",
    "source:", "for more information", "###",
)
_SPACE = re.compile(r"\s+")
_SUMMARY = re.compile(
    r"\b(?:is|are)\s+(?:a|an|the)\s+(?:leading|global|premier|major|provider|"
    r"manufacturer|operator|technology|financial|healthcare|specialty|diversified)",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


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
    boilerplate = (
        "broad, non-exclusionary distribution", "deemed to be material information",
        "through various channels", "involve risks and uncertainties",
        "estimated quantities of crude oil", "actual results may differ materially",
    )
    if any(marker in lower for marker in boilerplate):
        return False
    hits = sum(term in lower for term in _BUSINESS_TERMS)
    return len(text) >= 160 and hits >= 3


def _remove_issuer_name(text: str, company_name: str | None) -> str:
    """Prevent issuer names from becoming artificial description features."""
    if not company_name:
        return text
    cleaned = re.sub(re.escape(_clean(company_name)), " issuer ", text, flags=re.IGNORECASE)
    meaningful = [
        token for token in re.findall(r"[A-Za-z]{4,}", company_name)
        if token.lower() not in {"company", "corporation", "incorporated", "holdings", "limited"}
    ]
    for token in meaningful:
        cleaned = re.sub(rf"\b{re.escape(token)}\b", " issuer ", cleaned, flags=re.IGNORECASE)
    return _clean(cleaned)


def extract_business_description(text: str | None, company_name: str | None = None) -> str | None:
    """Extract a compact issuer-description passage, favoring press-release footers."""
    compact = _clean(text)
    if not compact:
        return None
    lower = compact.lower()
    # Press releases commonly end in "About <issuer>". Search backward so the
    # latest footer wins, while rejecting incidental "about the transaction" prose.
    end = len(compact)
    for _ in range(8):
        pos = lower.rfind("about ", 0, end)
        if pos < 0:
            break
        candidate = _truncate_section(compact[pos:])
        heading = candidate[:120].lower()
        rejected = heading.startswith(("about the transaction", "about the merger", "about the offering",
                                       "about this", "about our", "about its", "about forward"))
        if not rejected and _looks_like_profile(candidate):
            return _remove_issuer_name(candidate, company_name)
        end = pos

    # Fallback to a business-summary sentence near the end of the filing. Requiring
    # the issuer's first meaningful name token nearby reduces counterparty captures.
    tail_start = max(0, len(compact) - 18000)
    tail = compact[tail_start:]
    name_tokens = [t.lower() for t in re.findall(r"[A-Za-z]{4,}", company_name or "")
                   if t.lower() not in {"corporation", "company", "incorporated", "holdings"}]
    for match in reversed(list(_SUMMARY.finditer(tail))):
        start = max(0, match.start() - 180)
        candidate = _truncate_section(tail[start:], limit=1200)
        prefix = candidate[:260].lower()
        if (not name_tokens or any(token in prefix for token in name_tokens)) and _looks_like_profile(candidate):
            return _remove_issuer_name(candidate, company_name)
    return None


def extract_durable_business_language(
    text: str | None, company_name: str | None = None, limit: int = 1800
) -> str:
    """Select business-model sentences from the historical filing body."""
    compact = _clean(text)
    if not compact:
        return ""
    # Skip the standardized cover page. Score rather than take the first matches
    # so operational language beats generic legal uses of "business".
    body = compact[min(900, len(compact)):]
    candidates = []
    reject = ("securities and exchange commission", "forward-looking", "safe harbor",
              "signature", "exhibit", "registrant", "telephone number")
    specific = ("software", "platform", "network", "stores", "retail", "customers", "clients",
                "manufacturer", "manufactures", "healthcare", "payments", "cloud", "energy",
                "products", "services", "solutions", "distribution", "subscription")
    for position, sentence in enumerate(_SENTENCE.split(body)):
        sentence = sentence.strip()
        lower = sentence.lower()
        if not 80 <= len(sentence) <= 700 or any(term in lower for term in reject):
            continue
        base_hits = sum(term in lower for term in _BUSINESS_TERMS)
        specific_hits = sum(term in lower for term in specific)
        score = base_hits + 2 * specific_hits
        if score >= 4:
            candidates.append((score, -position, sentence))
    selected = [sentence for _, _, sentence in sorted(candidates, reverse=True)[:6]]
    selected.reverse()  # retain approximate document order among chosen passages
    return _remove_issuer_name(" ".join(selected)[:limit], company_name)


def event_text(
    title: str | None, title_ai: str | None, items, text: str | None = None,
    company_name: str | None = None,
) -> str:
    # title_ai is accepted for backward-compatible callers but deliberately not
    # used by the dataset builder because its historical model availability is unknown.
    title_value = _clean(title)
    item_values = sorted({str(item).strip() for item in (items or []) if str(item).strip()})
    compact = _clean(text)
    lower = compact.lower()
    item_start = lower.find("item ", min(700, len(lower)))
    excerpt = compact[item_start:item_start + 1400] if item_start >= 0 else compact[900:2300]
    excerpt = _remove_issuer_name(excerpt, company_name)
    return " ".join([title_value, *[f"form item {item}" for item in item_values], excerpt]).strip()


def build_profiles(input_path: Path, output_path: Path) -> dict:
    schema = pa.schema([
        ("permco", pa.int64()), ("document_id", pa.string()),
        ("available_at", pa.timestamp("us", tz="UTC")),
        ("business_description", pa.string()), ("event_text", pa.string()),
        ("description_extracted", pa.bool_()),
        ("profile_version", pa.string()),
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, schema, compression="zstd")
    rows = descriptions = 0
    companies = set()
    try:
        source = pq.ParquetFile(input_path)
        columns = ["permco", "document_id", "filing_date", "company_name", "title", "title_ai",
                   "items", "text"]
        for batch in source.iter_batches(batch_size=2048, columns=columns):
            records = []
            for filing in batch.to_pylist():
                available = pd.Timestamp(filing["filing_date"], tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
                description = extract_business_description(filing["text"], filing["company_name"])
                if description:
                    descriptions += 1
                    companies.add(filing["permco"])
                durable_topics = description or extract_durable_business_language(
                    filing["text"], filing["company_name"]
                )
                records.append({
                    "permco": filing["permco"], "document_id": filing["document_id"],
                    "available_at": available, "business_description": durable_topics,
                    "event_text": event_text(
                        filing["title"], None, filing["items"], filing["text"], filing["company_name"]
                    ),
                    "description_extracted": description is not None,
                    "profile_version": PROFILE_VERSION,
                })
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
            rows += len(records)
    finally:
        writer.close()
    return {"rows": rows, "descriptions": descriptions, "companies_with_descriptions": len(companies)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_FILINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stats = build_profiles(args.input, args.output)
    print(f"wrote {stats['rows']:,} dated filing records to {args.output}")
    print(f"business descriptions: {stats['descriptions']:,} across {stats['companies_with_descriptions']:,} companies")


if __name__ == "__main__":
    main()
