"""Deterministic role and filing aggregation into stock-month features."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .contracts import EVENT_CATEGORIES, ROLE_NAMES

NUMERIC_FIELDS = (
    "direction",
    "materiality",
    "surprise",
    "forward_looking",
    "distress_signal",
    "cash_flow_impact",
    "confidence",
    "contradiction_score",
    "evidence_quality",
)

AGGREGATE_COLUMNS = [
    "permno",
    "eom",
    "council_n_filings",
    "council_role_coverage",
    "council_direction_median",
    "council_direction_disagreement",
    "council_materiality_median",
    "council_surprise_median",
    "council_forward_looking_median",
    "council_distress_signal_median",
    "council_cash_flow_impact_median",
    "council_confidence_median",
    "council_contradiction_rate",
    "council_evidence_quality_median",
    "council_abstain_rate",
    "council_evidence_count",
    "council_n_material",
    "council_distress_any",
    *[f"council_event_{category}" for category in EVENT_CATEGORIES],
]


def _mad(values: pd.Series) -> float:
    median = float(values.median())
    return float((values - median).abs().median())


def aggregate_verdicts(verdict_rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = list(verdict_rows)
    if not rows:
        return pd.DataFrame(columns=AGGREGATE_COLUMNS)
    frame = pd.DataFrame(rows)
    required = {"document_id", "permno", "eom", "role", "event_category", *NUMERIC_FIELDS, "abstain", "evidence_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"verdict rows missing columns: {sorted(missing)}")

    filing_rows: list[dict[str, Any]] = []
    filing_groups = frame.groupby(["permno", "eom", "document_id"], dropna=False, sort=True)
    for (permno, eom, document_id), group in filing_groups:
        result: dict[str, Any] = {
            "permno": permno,
            "eom": eom,
            "document_id": document_id,
            "role_coverage": group["role"].nunique() / len(ROLE_NAMES),
            "direction_disagreement": _mad(group["direction"]),
            "abstain_rate": float(group["abstain"].mean()),
            "contradiction_rate": float(group["contradiction_score"].mean()),
            "evidence_count": float(group["evidence_count"].sum()),
        }
        for field in NUMERIC_FIELDS:
            result[field] = float(group[field].median())
        filing_rows.append(result)
    filings = pd.DataFrame(filing_rows)

    output_rows: list[dict[str, Any]] = []
    month_groups = filings.groupby(["permno", "eom"], dropna=False, sort=True)
    for (permno, eom), month in month_groups:
        out: dict[str, Any] = {
            "permno": permno,
            "eom": eom,
            "council_n_filings": int(month["document_id"].nunique()),
            "council_role_coverage": float(month["role_coverage"].mean()),
            "council_direction_disagreement": float(month["direction_disagreement"].mean()),
            "council_contradiction_rate": float(month["contradiction_rate"].mean()),
            "council_abstain_rate": float(month["abstain_rate"].mean()),
            "council_evidence_count": float(month["evidence_count"].sum()),
            "council_n_material": int((month["materiality"] >= 0.5).sum()),
            "council_distress_any": int((month["distress_signal"] >= 0.5).any()),
        }
        for field in NUMERIC_FIELDS:
            out[f"council_{field}_median"] = float(month[field].median())
        for category in EVENT_CATEGORIES:
            out[f"council_event_{category}"] = int(
                frame.loc[
                    frame["document_id"].isin(month["document_id"])
                    & (frame["permno"] == permno)
                    & (frame["eom"] == eom)
                    & (frame["event_category"] == category)
                ].shape[0]
            )
        output_rows.append(out)
    result = pd.DataFrame(output_rows)
    return result.reindex(columns=AGGREGATE_COLUMNS, fill_value=0)
