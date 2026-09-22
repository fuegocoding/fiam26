"""CLI: score eligible 8-K filings with the redacted council and export features."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.agentic.aggregate import aggregate_verdicts
from src.agentic.contracts import ROLE_NAMES
from src.agentic.archive_llm_council.council import CouncilRunner
from src.agentic.provider import CompatibleLLM, LLMConfig
from src.agentic.prerank import select_uncertain_document_ids
from src.agentic.redaction import REDACTION_VERSION, redact_filing
from src.agentic.storage import CouncilStore
from src.config import llm_config

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "8k_20150101_20260831_identified.parquet"
STORE_PATH = ROOT / "data" / "scoring" / "council_scores.db"
OUTPUT = ROOT / "output" / "council_features.parquet"
FINBERT_SCORES = ROOT / "output" / "finbert_scores.parquet"
PROMPT_VERSION = "council-v3"
AVAILABILITY_POLICY = (
    "All filings dated in calendar month t are aggregated at month-end t and are used only "
    "for the next-month target t+1; no intramonth trading claim is made."
)

REDACTION_COLUMNS = [
    "document_id",
    "permno",
    "filing_date",
    "form_type",
    "items",
    "text",
    "ticker",
    "company_name",
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
    "sector",
    "industry",
    "report_url",
    "is_amendment",
    "content_report_date",
]


def iter_filings(months: pd.PeriodIndex | None, limit: int | None):
    pf = pq.ParquetFile(RAW)
    yielded = 0
    for batch in pf.iter_batches(batch_size=2000, columns=REDACTION_COLUMNS):
        frame = batch.to_pandas()
        frame["filing_date"] = pd.to_datetime(frame["filing_date"])
        if months is not None:
            frame = frame[frame["filing_date"].dt.to_period("M").isin(months)]
        for record in frame.to_dict("records"):
            yield record
            yielded += 1
            if limit and yielded >= limit:
                return


def filing_month_eom(filing_date: str) -> str:
    return str(pd.Timestamp(filing_date) + pd.offsets.MonthEnd(0))[:10]


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--month", help="YYYY-MM")
    group.add_argument("--year", help="YYYY")
    parser.add_argument("--sample", type=int)
    parser.add_argument(
        "--top-uncertain",
        type=int,
        help="pre-rank FinBERT filing scores and analyze only the top N uncertain filings",
    )
    parser.add_argument(
        "--finbert-scores",
        default=str(FINBERT_SCORES),
        help="Parquet produced by src.finbert used for the cheap pre-ranking stage",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry roles previously cached as failures",
    )
    parser.add_argument("--data", default=None, help="reserved for a future alternate filing source")
    args = parser.parse_args()
    if args.data:
        raise ValueError("--data is not supported yet; use the supplied filing Parquet")

    cfg = llm_config()
    use_tools = cfg["tool_calling"]
    if use_tools:
        print("bounded local filing tool-calling: enabled")
    secret = os.getenv("REDACTION_HMAC_KEY", "").strip()
    if not secret:
        raise RuntimeError("REDACTION_HMAC_KEY must be set for stable strict pseudonymization")
    if args.workers:
        cfg["max_concurrency"] = args.workers
    llm = CompatibleLLM(
        LLMConfig(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            timeout_s=cfg["timeout_s"],
            temperature=cfg["temperature"],
            max_output_tokens=cfg["max_output_tokens"],
            response_format=cfg["response_format"],
        )
    )
    model_key = f"{cfg['base_url']}::{cfg['model']}"
    months = None
    if args.month:
        months = pd.PeriodIndex([pd.Period(args.month, freq="M")])
    elif args.year:
        months = pd.period_range(f"{args.year}-01", f"{args.year}-12", freq="M")

    selected_ids = None
    if args.top_uncertain:
        selected_ids = set(select_uncertain_document_ids(args.finbert_scores, args.top_uncertain))
        print(f"pre-ranked uncertain filings: {len(selected_ids):,}")

    store = CouncilStore(STORE_PATH)
    processed = 0
    iterator_limit = args.sample if selected_ids is None else None
    for record in iter_filings(months, iterator_limit):
        if selected_ids is not None and record["document_id"] not in selected_ids:
            continue
        if args.sample and processed >= args.sample:
            break
        record["eom"] = filing_month_eom(str(record["filing_date"]))
        filing = redact_filing(record, secret)
        store.save_identity(filing.document_id, filing.opaque_id, filing.permno, filing.filing_date, filing.eom)
        cached = store.cached_roles(
            filing.document_id,
            model_key,
            PROMPT_VERSION,
            filing.redaction_version,
            retry_failed=args.retry_failed,
        )
        missing_roles = tuple(role for role in ROLE_NAMES if role not in cached)
        if missing_roles:
            result = CouncilRunner(llm, roles=missing_roles, use_tools=use_tools).score_filing(filing)
            for verdict in result.verdicts:
                store.save_verdict(
                    filing.document_id,
                    filing.permno,
                    filing.eom,
                    verdict.role,
                    model_key,
                    PROMPT_VERSION,
                    filing.redaction_version,
                    verdict.as_dict(),
                )
            for failure in result.failures:
                store.save_failure(
                    filing.document_id,
                    failure.role,
                    model_key,
                    PROMPT_VERSION,
                    filing.redaction_version,
                    failure.error,
                )
        processed += 1
        if processed % 100 == 0:
            print(f"processed filings: {processed:,}")

    features = aggregate_verdicts(
        store.iter_verdicts(model_key, PROMPT_VERSION, REDACTION_VERSION)
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT, index=False)
    store.close()
    print(f"saved {len(features):,} stock-month rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
