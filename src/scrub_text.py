"""Deterministic 8-K text scrubber: replace identifying info with placeholders.

Reads data/raw/8k_20150101_20260831_identified.parquet in row-group batches and
writes data/scrubbed/8k_scrubbed.parquet with columns:
    document_id, text_scrubbed, n_replacements

Scrub layers (no LLM, no network, free):
  1. metadata identifiers: company names (all variants), tickers, CIK, CUSIP,
     GVKEY, FIGI, ISIN, accession numbers
  2. structural patterns: SEC commission file numbers, IRS employer IDs,
     phone numbers, emails, URLs
  3. optional spaCy NER (--ner): person/org/location names mapped to stable
     per-document placeholders (PERSON_A, PERSON_B, ...)

Run:
    python -m src.scrub_text                # full corpus, fast layers only
    python -m src.scrub_text --sample 500   # smoke test
    python -m src.scrub_text --ner          # adds spaCy NER layer (slow)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "8k_20150101_20260831_identified.parquet"
OUT_DIR = ROOT / "data" / "scrubbed"
OUT = OUT_DIR / "8k_scrubbed.parquet"
BATCH_ROWS = 2_000
SCRUB_VERSION = "v1"

COMPANY_PLACEHOLDER = "COMPANY_NAME"
PERSON_PLACEHOLDER_FMT = "PERSON_{}"

NAME_SUFFIXES = [
    r",?\s+incorporated\b\.?",
    r",?\s+inc\b\.?",
    r",?\s+corporation\b\.?",
    r",?\s+corp\b\.?",
    r",?\s+ltd\b\.?",
    r",?\s+limited\b\.?",
    r",?\s+llc\b\.?",
    r",?\s+l\.l\.c\b\.?",
    r",?\s+co\b\.?",
    r",?\s+company\b\.?",
    r",?\s+plc\b\.?",
    r",?\s+s\.?a\b\.?",
    r",?\s+ag\b\.?",
    r",?\s+n\.?v\b\.?",
    r",?\s+holdings?\b\.?",
    r",?\s+group\b\.?",
    r",?\s+international\b\.?",
]

META_NAME_COLS = [
    "company_name",
    "provider_company_name",
    "content_company_name",
    "historical_crsp_company_name",
    "compustat_header_company_name",
]
META_TICKER_COLS = ["ticker", "provider_ticker", "compustat_header_ticker"]
META_ID_COLS = ["cik", "cusip", "cusip8", "gvkey", "figi", "composite_figi",
                "share_class_figi", "isin", "accession_number"]

COMMISSION_FILE_RE = re.compile(r"\b\d{1,2}-\d{4,6}\b")
IRS_EIN_RE = re.compile(r"\b\d{2}-\d{7}\b")
PHONE_RE = re.compile(r"\(?\s*\d{3}\s*\)?\s*[-.]?\s*\d{3}\s*[-.]?\s*\d{4}")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")

READ_COLS = (
    ["document_id", "text"]
    + META_NAME_COLS
    + META_TICKER_COLS
    + META_ID_COLS
)


def unique(values) -> list[str]:
    out = []
    seen = set()
    for v in values:
        if v and isinstance(v, str) and v.strip() and v not in seen:
            seen.add(v)
            out.append(v.strip())
    return out


def name_variants(name: str) -> list[str]:
    variants = {name}
    base = re.sub(r"\s+", " ", name).strip()
    variants.add(base)
    for suffix in NAME_SUFFIXES:
        stripped = re.sub(suffix, "", base, flags=re.IGNORECASE).strip(" ,.")
        if len(stripped) >= 3:
            variants.add(stripped)
    for v in list(variants):
        variants.add(v.replace("&", " and "))
        variants.add(v.replace(" and ", " & "))
    return [v for v in variants if len(v) >= 3]


def build_company_patterns(names: list[str], tickers: list[str]) -> list[tuple[re.Pattern, str]]:
    patterns = []
    for name in sorted(unique(names), key=len, reverse=True):
        for variant in name_variants(name):
            patterns.append((
                re.compile(r"(?<![A-Za-z0-9])" + re.escape(variant) + r"(?![A-Za-z0-9])",
                           re.IGNORECASE),
                COMPANY_PLACEHOLDER,
            ))
    for t in sorted(unique(tickers)):
        if len(t) >= 2 and t.isalnum():
            patterns.append((
                re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])"),
                COMPANY_PLACEHOLDER,
            ))
    return patterns


def build_id_patterns(ids: list[str]) -> list[tuple[re.Pattern, str]]:
    patterns = []
    for v in unique(ids):
        if len(v) >= 4 and re.fullmatch(r"[\w.-]+", v):
            patterns.append((
                re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])"),
                "ID_NUMBER",
            ))
        if v.isdigit() and len(v) <= 10:
            padded = v.zfill(10)
            patterns.append((
                re.compile(r"(?<![0-9])" + padded + r"(?![0-9])"),
                "ID_NUMBER",
            ))
    return patterns


class Scrubber:
    def __init__(self, use_ner: bool = False):
        self.use_ner = use_ner
        self._ner = None
        if use_ner:
            import spacy
            self._ner = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    def scrub(self, text: str, company_patterns, id_patterns) -> tuple[str, Counter]:
        counts: Counter = Counter()
        if not text:
            return text, counts

        for pat, repl in company_patterns:
            text, n = pat.subn(repl, text)
            if n:
                counts["company"] += n
        for pat, repl in id_patterns:
            text, n = pat.subn(repl, text)
            if n:
                counts["ids"] += n

        for name, pat in [
            ("commission_file", COMMISSION_FILE_RE),
            ("ein", IRS_EIN_RE),
            ("phone", PHONE_RE),
            ("email", EMAIL_RE),
            ("url", URL_RE),
        ]:
            text, n = pat.subn(name.upper(), text)
            if n:
                counts[name] += n

        if self.use_ner and self._ner is not None:
            text = self._scrub_ner(text, counts)
        return text, counts

    def _scrub_ner(self, text: str, counts: Counter) -> str:
        doc = self._ner(text[:60000])
        replacements: dict[str, str] = {}
        person_idx = 0
        for ent in doc.ents:
            if ent.label_ not in ("PERSON", "ORG", "GPE", "LOC"):
                continue
            name = ent.text.strip()
            if len(name) < 3 or name in replacements:
                continue
            if ent.label_ == "PERSON":
                replacements[name] = PERSON_PLACEHOLDER_FMT.format(chr(65 + person_idx % 26))
                person_idx += 1
            else:
                replacements[name] = "ENTITY_NAME"
        for name, repl in sorted(replacements.items(), key=lambda kv: -len(kv[0])):
            pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])")
            text, n = pat.subn(repl, text)
            if n:
                counts["ner"] += n
        return text


def write_batch(writer: pq.ParquetWriter | None, records: list[dict]):
    if not records:
        return writer
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(OUT, table.schema)
    writer.write_table(table)
    return writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, help="scrub at most N filings")
    ap.add_argument("--ner", action="store_true", help="add spaCy NER layer")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    scrubber = Scrubber(use_ner=args.ner)
    pf = pq.ParquetFile(RAW)
    total = 0
    agg_counts: Counter = Counter()
    writer = None
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=READ_COLS):
        df = batch.to_pandas()
        if args.sample and total >= args.sample:
            break
        records = []
        for row in df.itertuples(index=False):
            if args.sample and total >= args.sample:
                break
            text = row.text or ""
            names = [getattr(row, c) for c in META_NAME_COLS]
            tickers = [getattr(row, c) for c in META_TICKER_COLS]
            ids = [getattr(row, c) for c in META_ID_COLS]
            company_patterns = build_company_patterns(names, tickers)
            id_patterns = build_id_patterns(ids)
            scrubbed, counts = scrubber.scrub(text, company_patterns, id_patterns)
            records.append({
                "document_id": row.document_id,
                "text_scrubbed": scrubbed,
                "n_replacements": sum(counts.values()),
            })
            agg_counts.update(counts)
            total += 1
        writer = write_batch(writer, records)
        print(f"scrubbed {total:,} ({time.time() - t0:,.0f}s)")

    if writer is not None:
        writer.close()

    print(f"done: {total:,} filings -> {OUT}")
    print("replacement counts:")
    for k, v in agg_counts.most_common():
        print(f"  {k}: {v:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
