"""CLI: score 8-K filings with Jev System One and export stock-month features.

Full competition scope by default: every filing in
``data/raw/8k_20150101_20260831_identified.parquet`` is scored once and cached
in ``data/scoring/jev_scores.db``; reruns are idempotent. Use ``--month`` /
``--year`` / ``--sample`` for bounded pilots.

Availability: filings dated in calendar month t aggregate at month-end t and
predict the next-month target t+1 only. No intramonth execution claim.

Example:
    python -m src.agentic.run_jev --sample 20
    python -m src.agentic.run_jev --month 2021-01
    python -m src.agentic.run_jev --workers 8
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.agentic.filings import filing_month_eom, iter_filings
from src.agentic.jev_aggregate import aggregate_jev_rows
from src.agentic.jev_encode import encode_filing_answers
from src.agentic.jev_provider import JevClient, JevConfig
from src.agentic.jev_questions import JEV_PROMPT_VERSION, build_jev_questions, build_state
from src.agentic.jev_storage import JevStore
from src.agentic.redaction import REDACTION_VERSION, redact_filing
from src.config import jev_config

ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / "data" / "scoring" / "jev_scores.db"
OUTPUT = ROOT / "output" / "jev_features.parquet"


def _score_one(client: JevClient, questions: dict, state: dict) -> tuple[dict, dict]:
    response = client.system_one(state, questions)
    encoded = encode_filing_answers(response.answers, questions)
    return response.answers, encoded, {"model": response.model, **response.usage}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--month", help="YYYY-MM (pilot slice)")
    group.add_argument("--year", help="YYYY (pilot slice)")
    parser.add_argument("--sample", type=int, help="score at most N filings (pilot)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry filings previously cached as failures",
    )
    args = parser.parse_args()

    cfg = jev_config()
    if args.workers:
        cfg["max_concurrency"] = args.workers
    secret = os.getenv("REDACTION_HMAC_KEY", "").strip()
    if not secret:
        raise RuntimeError("REDACTION_HMAC_KEY must be set for stable strict pseudonymization")

    months = None
    if args.month:
        months = pd.PeriodIndex([pd.Period(args.month, freq="M")])
    elif args.year:
        months = pd.period_range(f"{args.year}-01", f"{args.year}-12", freq="M")

    client = JevClient(
        JevConfig(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
            model=cfg["model"],
            timeout_s=cfg["timeout_s"],
        )
    )
    model_key = f"{cfg['base_url']}::{cfg['model']}"
    questions = build_jev_questions()
    store = JevStore(STORE_PATH)

    # Phase 1: scan document IDs and resolve the pending set against the cache.
    # The cache check runs BEFORE redaction: a full resume then costs one cheap
    # parquet scan plus indexed selects, not 373k regex redactions.
    pending: set[str] = set()
    scanned = 0
    for record in iter_filings(months, args.sample):
        scanned += 1
        document_id = str(record.get("document_id") or "")
        if not document_id:
            continue
        if args.retry_failed or not store.is_cached(
            document_id, model_key, JEV_PROMPT_VERSION, REDACTION_VERSION
        ):
            pending.add(document_id)
    print(
        f"model={cfg['model']} prompt={JEV_PROMPT_VERSION} "
        f"scanned {scanned:,} filings, {len(pending):,} to score "
        f"({scanned - len(pending):,} cached)"
    )

    t0 = time.time()
    done = failed = 0
    input_tokens = 0
    total_cost = 0.0

    def work(filing):  # type: ignore[no-untyped-def]
        # Workers only score. All sqlite writes happen in the main thread:
        # a connection created in one thread cannot be used in another.
        answers, encoded, usage = _score_one(client, questions, build_state(filing))
        return filing, answers, encoded, usage

    def drain(pool, batch):  # type: ignore[no-untyped-def]
        nonlocal done, failed, input_tokens, total_cost
        futures = {pool.submit(work, filing): filing for filing in batch}
        for future in as_completed(futures):
            filing = futures[future]
            try:
                _, answers, encoded, usage = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded, pipeline continues
                failed += 1
                store.save_failure(
                    filing.document_id,
                    model_key,
                    JEV_PROMPT_VERSION,
                    filing.redaction_version,
                    f"{type(exc).__name__}: {exc}",
                )
                if failed <= 5:
                    print(f"  failure {failed}: {filing.document_id} -> {exc}"[:200])
            else:
                store.save_answer(
                    filing.document_id,
                    filing.permno,
                    filing.eom,
                    model_key,
                    JEV_PROMPT_VERSION,
                    filing.redaction_version,
                    answers,
                    encoded,
                    usage,
                )
                done += 1
                input_tokens += int(usage.get("input_tokens") or 0)
                if isinstance(usage.get("cost"), (int, float)):
                    total_cost += float(usage["cost"])
            if (done + failed) % 500 == 0:
                print(
                    f"progress: {done + failed:,}/{len(pending):,} "
                    f"({time.time() - t0:,.0f}s)"
                )

    # Phase 2: stream in bounded windows so a 373k-file backfill never
    # materializes all redacted texts in memory at once.
    window = max(cfg["max_concurrency"] * 4, 16)
    with ThreadPoolExecutor(max_workers=cfg["max_concurrency"]) as pool:
        batch: list = []
        for record in iter_filings(months, args.sample):
            document_id = str(record.get("document_id") or "")
            if document_id not in pending:
                continue
            record["eom"] = filing_month_eom(str(record["filing_date"]))
            filing = redact_filing(record, secret)
            store.save_identity(
                filing.document_id,
                filing.opaque_id,
                filing.permno,
                filing.filing_date,
                filing.eom,
            )
            batch.append(filing)
            if len(batch) >= window:
                drain(pool, batch)
                batch = []
        if batch:
            drain(pool, batch)

    print(
        f"done: {done:,} scored, {failed} failed, {input_tokens:,} input tokens, "
        f"~${total_cost:.4f} reported cost in {time.time() - t0:,.0f}s"
    )

    features = aggregate_jev_rows(
        store.iter_answers(model_key, JEV_PROMPT_VERSION, REDACTION_VERSION)
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT, index=False)
    store.close()
    print(f"saved {len(features):,} stock-month rows -> {OUTPUT}")


if __name__ == "__main__":
    main()
