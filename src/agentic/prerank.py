"""Cheap, deterministic pre-ranking for cost-controlled filing analysis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "document_id",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
    "sentiment_min_chunk",
    "sentiment_max_chunk",
}


def with_uncertainty_score(scores: pd.DataFrame) -> pd.DataFrame:
    """Add a reproducible [0, 1] uncertainty score to FinBERT filing scores.

    Normalized three-class entropy is the primary signal. Chunk-level polarity
    disagreement is a small secondary signal so filings whose sections disagree
    are not lost when the aggregate probabilities are overconfident.
    """
    missing = sorted(REQUIRED_COLUMNS - set(scores.columns))
    if missing:
        raise ValueError(f"missing pre-ranking columns: {missing}")
    if scores["document_id"].duplicated().any():
        raise ValueError("document_id must be unique for pre-ranking")

    out = scores.copy()
    probabilities = out[["prob_positive", "prob_negative", "prob_neutral"]].astype(float)
    probabilities = probabilities.clip(lower=1e-12, upper=1.0)
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1) / np.log(3.0)
    disagreement = (
        out["sentiment_max_chunk"].astype(float)
        - out["sentiment_min_chunk"].astype(float)
    ).abs().clip(lower=0.0, upper=2.0) / 2.0
    out["prerank_uncertainty"] = (0.8 * entropy + 0.2 * disagreement).clip(0.0, 1.0)
    return out


def select_uncertain_document_ids(
    scores: pd.DataFrame | str | Path,
    limit: int,
) -> list[str]:
    """Return the top uncertain document IDs with deterministic tie-breaking."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    frame = pd.read_parquet(scores) if isinstance(scores, (str, Path)) else scores
    ranked = with_uncertainty_score(frame).sort_values(
        ["prerank_uncertainty", "document_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked.head(limit)["document_id"].astype(str).tolist()
