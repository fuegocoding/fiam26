"""Local, versioned text and event measurements from dated supplied 8-Ks."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from ..contracts.base import StrictModel
from .filing_access import as_of_timestamp

TEXT_FEATURES = (
    "filings_text_risk_90d",
    "filings_text_uncertainty_90d",
    "filings_credit_event_90d",
    "filings_management_event_90d",
    "filings_earnings_event_90d",
)
# Transparent measurement dictionaries, not a trained sentiment classifier.
RISK = re.compile(
    r"\b(?:default|defaults|impairment|restatement|bankruptcy|covenant|litigation|breach)\b",
    re.I,
)
UNCERTAINTY = re.compile(
    r"\b(?:uncertain|uncertainty|contingent|contingencies|may|might)\b", re.I
)
VERSION = "literal_8k_measurements_v1"


def text_measurements(store, cache):
    metadata = store.metadata()
    if cache.exists():
        result = pd.read_parquet(cache)
        if result.document_id.tolist() == metadata.document_id.tolist():
            return result
    parquet, parts = pq.ParquetFile(store.path), []
    for group, records in metadata.groupby("_row_group"):
        columns = [
            x for x in ("text", "items") if x in parquet.schema_arrow.names
        ]
        page = (
            parquet.read_row_group(int(group), columns=columns)
            .take(pa.array(records._row_in_group.to_numpy(), type=pa.int64()))
            .to_pandas()
        )
        page.index = records.index
        rows = []
        for index, row in page.iterrows():
            text = str(row.get("text", "") or "")
            items = set(
                str(x).strip().replace("Item ", "")
                for x in row.get("items", [])
            )
            scale = max(1, len(text.split()))
            rows.append(
                (
                    index,
                    len(RISK.findall(text)) / scale,
                    len(UNCERTAINTY.findall(text)) / scale,
                    float(bool(items & {"1.01", "1.02", "2.03", "2.04"})),
                    float("5.02" in items),
                    float("2.02" in items),
                )
            )
        parts.append(
            pd.DataFrame(rows, columns=["index", *TEXT_FEATURES]).set_index(
                "index"
            )
        )
    values = (
        pd.concat(parts).reindex(metadata.index)
        if parts
        else pd.DataFrame(columns=TEXT_FEATURES)
    )
    result = pd.concat(
        [metadata[["document_id", "permno", "known_at"]], values], axis=1
    )
    result.to_parquet(cache, index=False)
    return result


def text_feature_arrays(records, dates, securities):
    instants = np.array(
        [as_of_timestamp(d).value for d in dates], dtype=np.int64
    )
    result = {
        f: np.full((len(dates), len(securities)), np.nan, dtype=np.float32)
        for f in TEXT_FEATURES
    }
    locations = {int(p): i for i, p in enumerate(securities)}
    for security, frame in records.loc[records.permno.isin(securities)].groupby(
        "permno"
    ):
        frame = frame.sort_values("known_at")
        times = frame.known_at.astype("int64").to_numpy()
        right = times.searchsorted(instants, side="right")
        left = times.searchsorted(
            instants - 90 * 86_400_000_000_000, side="right"
        )
        counts, seen = right - left, right > 0
        column = locations[int(security)]
        for feature in TEXT_FEATURES:
            cumulative = np.r_[0, frame[feature].fillna(0).to_numpy().cumsum()]
            values = cumulative[right] - cumulative[left]
            if "text_" in feature:
                values = np.divide(
                    values,
                    counts,
                    out=np.full(len(counts), np.nan),
                    where=counts > 0,
                )
            result[feature][seen, column] = values[seen]
    return result


class FilingMeasurement(StrictModel):
    """Agent-selected text measurements computed locally over dated documents."""

    name: str = Field(pattern=r"^filings_custom_[a-z][a-z0-9_]{0,40}$")
    rationale: str = Field(min_length=20, max_length=800)
    phrases: list[str] = Field(min_length=1, max_length=20)
    item_codes: list[str] = Field(default_factory=list, max_length=12)
    window_days: int = Field(ge=1, le=365)
    aggregation: Literal["frequency", "document_count"]


def custom_measurements(store, specifications, cache):
    """Literal phrase matching avoids executable regex/code and remote raw text."""
    if cache.exists():
        return pd.read_parquet(cache)
    specs = {s.name: s for s in specifications}
    if len(specs) != len(specifications) or len(specs) > 12:
        raise ValueError("Use distinct bounded text measurements")
    patterns = {}
    for name, spec in specs.items():
        if any(not 1 <= len(p.strip()) <= 100 for p in spec.phrases):
            raise ValueError("Text phrases must contain 1-100 characters")
        if any(not re.fullmatch(r"[1-9]\.\d{2}", c) for c in spec.item_codes):
            raise ValueError("Use actual SEC item code syntax")
        patterns[name] = re.compile(
            "|".join(re.escape(p.strip()) for p in spec.phrases), re.I
        )
    metadata, parquet, parts = store.metadata(), pq.ParquetFile(store.path), []
    for group, records in metadata.groupby("_row_group"):
        columns = [
            x for x in ("text", "items") if x in parquet.schema_arrow.names
        ]
        page = (
            parquet.read_row_group(int(group), columns=columns)
            .take(pa.array(records._row_in_group.to_numpy(), type=pa.int64()))
            .to_pandas()
        )
        page.index = records.index
        values = {name: [] for name in specs}
        for _, row in page.iterrows():
            text = str(row.get("text", "") or "")
            items = set(
                str(x).strip().replace("Item ", "")
                for x in row.get("items", [])
            )
            words = max(1, len(text.split()))
            for name, spec in specs.items():
                matches = (
                    len(patterns[name].findall(text))
                    if not spec.item_codes
                    or items.intersection(spec.item_codes)
                    else 0
                )
                values[name].append(
                    matches / words
                    if spec.aggregation == "frequency"
                    else float(matches > 0)
                )
        parts.append(pd.DataFrame(values, index=records.index))
    result = pd.concat(
        [
            metadata[["document_id", "permno", "known_at"]],
            pd.concat(parts).reindex(metadata.index),
        ],
        axis=1,
    )
    result.to_parquet(cache, index=False)
    return result


def custom_feature_arrays(records, dates, securities, specifications):
    result = {}
    # Reuse the exact same causal window aggregation as the supplied measures.
    instants = np.array(
        [as_of_timestamp(d).value for d in dates], dtype=np.int64
    )
    locations = {int(p): i for i, p in enumerate(securities)}
    for spec in specifications:
        result[spec.name] = np.full(
            (len(dates), len(securities)), np.nan, dtype=np.float32
        )
    for security, frame in records.loc[records.permno.isin(securities)].groupby(
        "permno"
    ):
        frame = frame.sort_values("known_at")
        times = frame.known_at.astype("int64").to_numpy()
        right = times.searchsorted(instants, side="right")
        seen = right > 0
        for spec in specifications:
            left = times.searchsorted(
                instants - spec.window_days * 86_400_000_000_000, side="right"
            )
            cumulative = np.r_[
                0, frame[spec.name].fillna(0).to_numpy().cumsum()
            ]
            value = cumulative[right] - cumulative[left]
            if spec.aggregation == "frequency":
                count = right - left
                value = np.divide(
                    value,
                    count,
                    out=np.full(len(count), np.nan),
                    where=count > 0,
                )
            result[spec.name][seen, locations[int(security)]] = value[seen]
    return result
