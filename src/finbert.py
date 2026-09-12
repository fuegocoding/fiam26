"""8-K sentiment scoring with FinBERT (ProsuAI/finbert, 3-class: positive/neutral/negative).

Local GPU inference — no API, no cost, fully reproducible. Produces a coarse
sentiment signal per filing; the LLM engine (text_scoring.py) remains the source
for materiality/surprise/category etc.

Requires: torch, transformers, pandas, pyarrow (pip install torch transformers)

Idempotent: every scored document_id is cached in sqlite; reruns skip cached rows.
Run:
    python -m src.finbert --month 2021-01
    python -m src.finbert --year 2021
    python -m src.finbert --sample 20
    python -m src.finbert --export            # dump scores to parquet
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "8k_20150101_20260831_identified.parquet"
SCORING_DIR = ROOT / "data" / "scoring"
DB_PATH = SCORING_DIR / "finbert_scores.db"
OUT_PATH = ROOT / "output" / "finbert_scores.parquet"

MODEL_NAME = "ProsusAI/finbert"
CHAR_BUDGET = 12_000
BATCH_ROWS = 2_000
LABELS = ("positive", "negative", "neutral")


def open_db():
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            document_id TEXT PRIMARY KEY,
            permno INTEGER,
            filing_date TEXT,
            n_chunks INTEGER,
            sentiment REAL,
            prob_positive REAL,
            prob_negative REAL,
            prob_neutral REAL,
            sentiment_min_chunk REAL,
            sentiment_max_chunk REAL,
            created_at TEXT
        )
        """
    )
    con.commit()
    return con


def cached_ids(con) -> set[str]:
    rows = con.execute("SELECT document_id FROM scores").fetchall()
    return {r[0] for r in rows}


def iter_filing_batches(months: pd.PeriodIndex | None, max_filings: int | None):
    pf = pq.ParquetFile(RAW)
    done = 0
    for batch in pf.iter_batches(
        batch_size=BATCH_ROWS,
        columns=["document_id", "permno", "filing_date", "text"],
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


def load_pipeline(device: str, batch_size: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()
    # resolve label->index from config (don't assume 0/1/2 order)
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    idx = {lab: i for i, lab in id2label.items()}
    missing = [l for l in LABELS if l not in idx]
    if missing:
        raise RuntimeError(f"unexpected FinBERT labels: {id2label}")
    return tok, model, device, batch_size, idx


def aggregate(chunk_scores: list[dict]) -> dict | None:
    """chunk_scores: list of {positive, negative, neutral} prob dicts."""
    if not chunk_scores:
        return None
    per_chunk = [c["positive"] - c["negative"] for c in chunk_scores]
    n = len(chunk_scores)
    return {
        "n_chunks": n,
        "sentiment": sum(per_chunk) / n,
        "prob_positive": sum(c["positive"] for c in chunk_scores) / n,
        "prob_negative": sum(c["negative"] for c in chunk_scores) / n,
        "prob_neutral": sum(c["neutral"] for c in chunk_scores) / n,
        "sentiment_min_chunk": min(per_chunk),
        "sentiment_max_chunk": max(per_chunk),
    }


def store(con, filing: dict, agg: dict | None):
    con.execute(
        "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            filing["document_id"],
            filing["permno"],
            str(filing["filing_date"]),
            agg["n_chunks"] if agg else 0,
            agg["sentiment"] if agg else None,
            agg["prob_positive"] if agg else None,
            agg["prob_negative"] if agg else None,
            agg["prob_neutral"] if agg else None,
            agg["sentiment_min_chunk"] if agg else None,
            agg["sentiment_max_chunk"] if agg else None,
            time.strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )


def score_batch(tok, model, device, batch_size: int, texts: list[str], idx: dict) -> list[list[dict]]:
    """Tokenize all filings into 512-token chunks, run inference, group by filing."""
    import torch

    # build flat chunk list with back-pointers
    chunks: list[str] = []
    owner: list[int] = []
    for i, text in enumerate(texts):
        text = text[:CHAR_BUDGET] if isinstance(text, str) else ""
        if not text.strip():
            continue
        enc = tok(
            text,
            add_special_tokens=False,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=512,
            stride=64,
            return_tensors=None,
        )
        for ids in enc["input_ids"]:
            chunks.append(tok.decode(ids))
            owner.append(i)

    results: list[list[dict]] = [[] for _ in texts]
    if not chunks:
        return results

    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            part = chunks[start : start + batch_size]
            enc = tok(
                part,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            for j, p in enumerate(probs):
                results[owner[start + j]].append(
                    {
                        "positive": p[idx["positive"]],
                        "negative": p[idx["negative"]],
                        "neutral": p[idx["neutral"]],
                    }
                )
    return results


def export_parquet(con):
    df = pd.read_sql_query("SELECT * FROM scores", con)
    df.to_parquet(OUT_PATH, index=False)
    print(f"exported {len(df):,} rows -> {OUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--month", help="YYYY-MM")
    g.add_argument("--year", help="YYYY")
    ap.add_argument("--sample", type=int, help="score at most N filings")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--batch-size", type=int, default=64, help="chunks per forward pass")
    ap.add_argument("--export", action="store_true", help="export db to parquet and exit")
    args = ap.parse_args()

    con = open_db()
    if args.export:
        export_parquet(con)
        return 0

    try:
        tok, model, device, bs, idx = load_pipeline(args.device, args.batch_size)
    except ImportError as e:
        print(f"missing dependency: {e}\npip install torch transformers")
        return 1
    print(f"model={MODEL_NAME} device={device} fp16={device == 'cuda'}")

    months = None
    if args.month:
        months = pd.PeriodIndex([pd.Period(args.month, freq="M")])
    elif args.year:
        months = pd.period_range(f"{args.year}-01", f"{args.year}-12", freq="M")

    done = cached_ids(con)
    todo = []
    for record in iter_filing_batches(months, args.sample):
        if record["document_id"] in done:
            continue
        todo.append(record)
    print(f"filings to score: {len(todo):,} (cached: {len(done):,})")

    t0 = time.time()
    scored = empty = 0
    chunk_total = 0
    for start in range(0, len(todo), BATCH_ROWS):
        part = todo[start : start + BATCH_ROWS]
        texts = [r["text"] for r in part]
        grouped = score_batch(tok, model, device, bs, texts, idx)
        for filing, chunks in zip(part, grouped):
            agg = aggregate(chunks)
            if agg:
                chunk_total += agg["n_chunks"]
                scored += 1
            else:
                empty += 1
            store(con, filing, agg)
        con.commit()
        rate = scored / max(time.time() - t0, 1e-9)
        print(
            f"progress: {min(start + BATCH_ROWS, len(todo)):,}/{len(todo):,} "
            f"({time.time() - t0:,.0f}s, {rate:,.0f} filings/s)"
        )

    dt = time.time() - t0
    print(
        f"done: {scored:,} scored, {empty:,} empty, {chunk_total:,} chunks "
        f"in {dt:,.0f}s ({chunk_total / max(dt, 1e-9):,.0f} chunks/s)"
    )
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
