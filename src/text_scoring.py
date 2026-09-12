"""8-K text scoring engine: filings -> LLM scores (OpenRouter, z-ai/glm-5.3-flash).

Idempotent: every scored document_id is cached in sqlite; reruns skip cached rows.
Run only on request, e.g.:
    python -m src.text_scoring --month 2021-01
    python -m src.text_scoring --year 2021
    python -m src.text_scoring --sample 20
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from openai import OpenAI

from src.config import llm_config
from src.prompts import EVENT_CATEGORIES, SYSTEM_PROMPT, build_user_message, validate_response

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "8k_20150101_20260831_identified.parquet"
SCORING_DIR = ROOT / "data" / "scoring"
DB_PATH = SCORING_DIR / "text_scores.db"
CHAR_BUDGET = 12_000
MAX_OUTPUT_TOKENS = 1_600
BATCH_ROWS = 2_000
RETRY_BACKOFF = [2, 5, 12]


def open_db():
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            document_id TEXT PRIMARY KEY,
            permno INTEGER,
            filing_date TEXT,
            model TEXT,
            payload_json TEXT,
            parsed INTEGER,
            error TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            created_at TEXT
        )
        """
    )
    con.commit()
    return con


def cached_ids(con) -> set[str]:
    rows = con.execute("SELECT document_id FROM scores WHERE parsed = 1").fetchall()
    return {r[0] for r in rows}


def _parse_or_repair(client, cfg, messages, resp):
    raw = resp.choices[0].message.content or ""
    try:
        return validate_response(raw), raw
    except ValueError as first_err:
        repair = [
            *messages,
            {"role": "assistant", "content": raw[:2_000]},
            {
                "role": "user",
                "content": (
                    "Your previous output was not valid. Return ONLY the JSON object again, "
                    "correcting this problem: "
                    + str(first_err)
                    + ". event_category must be spelled exactly as one of: "
                    + ", ".join(EVENT_CATEGORIES)
                ),
            },
        ]
        r2 = client.chat.completions.create(
            model=cfg["model"],
            messages=repair,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.0,
            extra_body={"reasoning": {"effort": "low", "exclude": True}},
            timeout=cfg["timeout_s"],
        )
        raw2 = r2.choices[0].message.content or ""
        return validate_response(raw2), raw2


def score_one(client, cfg, filing: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(
                ticker=filing["ticker"],
                company_name=filing["company_name"],
                filing_date=str(filing["filing_date"]),
                event_date_hint=filing.get("event_date_hint"),
                form_type=filing["form_type"],
                items=list(filing["items"]) if filing["items"] is not None else [],
                text=filing["text"],
                char_budget=CHAR_BUDGET,
            ),
        },
    ]
    last_err = None
    for attempt, delay in enumerate([0] + RETRY_BACKOFF):
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,
                extra_body={"reasoning": {"effort": "low", "exclude": True}},
                timeout=cfg["timeout_s"],
            )
            payload, raw = _parse_or_repair(client, cfg, messages, resp)
            usage = resp.usage
            return {
                "payload": payload,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "error": None,
                "parsed": 1,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return {
        "payload": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": last_err,
        "parsed": 0,
    }


def store(con, cfg, filing: dict, result: dict):
    con.execute(
        "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            filing["document_id"],
            filing["permno"],
            str(filing["filing_date"]),
            cfg["model"],
            json.dumps(result["payload"], ensure_ascii=False) if result["payload"] else None,
            result["parsed"],
            result["error"],
            result["prompt_tokens"],
            result["completion_tokens"],
            time.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )
    con.commit()


def iter_filing_batches(months: pd.PeriodIndex | None, max_filings: int | None):
    pf = pq.ParquetFile(RAW)
    done = 0
    for batch in pf.iter_batches(
        batch_size=BATCH_ROWS,
        columns=[
            "document_id",
            "permno",
            "filing_date",
            "form_type",
            "items",
            "text",
            "ticker",
            "company_name",
            "content_report_date",
        ],
    ):
        df = batch.to_pandas()
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        if months is not None:
            df = df[df["filing_date"].dt.to_period("M").isin(months)]
        if len(df):
            for record in df.to_dict("records"):
                yield record
                done += 1
                if max_filings and done >= max_filings:
                    return


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--month", help="YYYY-MM")
    g.add_argument("--year", help="YYYY")
    ap.add_argument("--sample", type=int, help="score at most N filings")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = llm_config()
    if args.workers:
        cfg["max_concurrency"] = args.workers

    months = None
    if args.month:
        months = pd.PeriodIndex([pd.Period(args.month, freq="M")])
    elif args.year:
        months = pd.period_range(f"{args.year}-01", f"{args.year}-12", freq="M")

    con = open_db()
    done = cached_ids(con)
    print(f"provider={cfg['provider']} model={cfg['model']} cached={len(done):,}")

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    todo = []
    for record in iter_filing_batches(months, args.sample):
        if record["document_id"] in done:
            continue
        todo.append(record)
        if args.sample and len(todo) >= args.sample:
            break
    print(f"filings to score: {len(todo):,}")

    t0 = time.time()
    ptok = ctok = 0
    fails = 0
    window = cfg["max_concurrency"] * 4
    with ThreadPoolExecutor(max_workers=cfg["max_concurrency"]) as pool:
        for start in range(0, len(todo), window):
            chunk = todo[start : start + window]
            futures = {pool.submit(score_one, client, cfg, f): f for f in chunk}
            for fut in as_completed(futures):
                filing = futures[fut]
                result = fut.result()
                store(con, cfg, filing, result)
                if result["parsed"]:
                    ptok += result["prompt_tokens"] or 0
                    ctok += result["completion_tokens"] or 0
                else:
                    fails += 1
                    print(f"  failure {fails}: {filing['document_id']} -> {(result['error'] or '')[:120]}")
            print(
                f"progress: {min(start + window, len(todo)):,}/{len(todo):,} "
                f"({time.time() - t0:,.0f}s)"
            )

    n = len(todo)
    print(
        f"done: {n:,} filings, {n - fails:,} parsed, {fails} failed, "
        f"{ptok + ctok:,} tokens in {time.time() - t0:,.0f}s"
    )
    con.close()


if __name__ == "__main__":
    sys.exit(main())
