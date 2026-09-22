"""Deterministic aggregation of Jev filing rows into stock-month features.

Mirrors the archived council's aggregation semantics (median for levels,
MAD for disagreement, rates for abstention/contradiction, counts for
categories) with a ``jev_`` prefix so Jev and council features can coexist in
one ablation table.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .contracts import EVENT_CATEGORIES

#: Median-aggregated filing fields and their stock-month output names.
MEDIAN_FIELDS = (
    ("signed_direction", "jev_direction_median"),
    ("materiality", "jev_materiality_median"),
    ("surprise", "jev_surprise_median"),
    ("forward_looking", "jev_forward_looking_median"),
    ("distress_p", "jev_distress_p_median"),
    ("signed_cashflow", "jev_cashflow_median"),
    ("conf_event", "jev_confidence_median"),
    ("materiality_spread", "jev_materiality_spread_median"),
    ("event_entropy", "jev_event_entropy_median"),
)

JEV_AGGREGATE_COLUMNS = [
    "permno",
    "eom",
    "jev_n_filings",
    "jev_role_coverage",
    "jev_direction_median",
    "jev_direction_disagreement",
    "jev_materiality_median",
    "jev_surprise_median",
    "jev_forward_looking_median",
    "jev_distress_p_median",
    "jev_cashflow_median",
    "jev_confidence_median",
    "jev_materiality_spread_median",
    "jev_event_entropy_median",
    "jev_contradiction_rate",
    "jev_abstain_rate",
    "jev_n_material",
    "jev_distress_any",
    *[f"jev_event_{category}" for category in EVENT_CATEGORIES],
]


def _mad(values: pd.Series) -> float:
    median = float(values.median())
    return float((values - median).abs().median())


def aggregate_jev_rows(encoded_rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = list(encoded_rows)
    if not rows:
        return pd.DataFrame(columns=JEV_AGGREGATE_COLUMNS)
    frame = pd.DataFrame(rows)
    required = {"document_id", "permno", "eom", "event_category", "abstain", *{f for f, _ in MEDIAN_FIELDS}}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"jev rows missing columns: {sorted(missing)}")

    filing_rows: list[dict[str, Any]] = []
    for (permno, eom, document_id), group in frame.groupby(
        ["permno", "eom", "document_id"], dropna=False, sort=True
    ):
        result: dict[str, Any] = {
            "permno": permno,
            "eom": eom,
            "document_id": document_id,
            "abstain_rate": float(group["abstain"].mean()),
            # Low-confidence share stands in for the council's contradiction rate:
            # mass spread thinly across categories rather than concentrated.
            "contradiction_rate": float((1.0 - group["conf_event"]).mean()),
        }
        for field, _ in MEDIAN_FIELDS:
            result[field] = float(group[field].median())
        counts = group["event_category"].value_counts()
        result["primary_category"] = min(
            (c for c, n in counts.items() if n == counts.max()),
            key=EVENT_CATEGORIES.index,
        )
        result["n_material"] = int((group["materiality"] >= 0.5).sum())
        result["distress_any"] = int((group["distress_p"] >= 0.5).any())
        filing_rows.append(result)
    filings = pd.DataFrame(filing_rows)

    output_rows: list[dict[str, Any]] = []
    for (permno, eom), month in filings.groupby(["permno", "eom"], dropna=False, sort=True):
        out: dict[str, Any] = {
            "permno": permno,
            "eom": eom,
            "jev_n_filings": int(month["document_id"].nunique()),
            "jev_role_coverage": 1.0,
            # Cross-filing dispersion within the month: one Jev call per filing,
            # so disagreement is measured across filings, not within them.
            "jev_direction_disagreement": _mad(month["signed_direction"]),
            "jev_contradiction_rate": float(month["contradiction_rate"].mean()),
            "jev_abstain_rate": float(month["abstain_rate"].mean()),
            "jev_n_material": int(month["n_material"].sum()),
            "jev_distress_any": int(month["distress_any"].max()),
        }
        for field, column in MEDIAN_FIELDS:
            out[column] = float(month[field].median())
        for category in EVENT_CATEGORIES:
            out[f"jev_event_{category}"] = int((month["primary_category"] == category).sum())
        output_rows.append(out)
    result = pd.DataFrame(output_rows)
    return result.reindex(columns=JEV_AGGREGATE_COLUMNS, fill_value=0)
