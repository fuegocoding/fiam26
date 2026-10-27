"""Dated local access to supplied 8-K text and research-only filing features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .input_permissions import FILING_RESEARCH_FEATURES

SOURCE_LIMITATIONS = (
    "The supplied corpus is survivor-conditioned; research access is not financial admission.",
    "Supplied issuer mappings require a separate historical-identity audit.",
    "No filing history means unknown coverage, not a zero-event observation.",
)


def as_of_timestamp(value):
    """An undated time of day means the end of that New York calendar date."""
    stamp = pd.Timestamp(value).as_unit("ns")
    if stamp.tzinfo is None:
        if stamp == stamp.normalize():
            stamp = (stamp + pd.Timedelta(days=1)).tz_localize(
                "America/New_York"
            ) - pd.Timedelta(nanoseconds=1)
        else:
            stamp = stamp.tz_localize("America/New_York")
    return stamp.tz_convert("UTC")


def known_at(frame):
    """Prefer SEC acceptance; date-only records become available at day end."""
    accepted = pd.to_datetime(
        frame.get("sec_accepted_at_utc", pd.Series(pd.NaT, index=frame.index)),
        errors="coerce",
        utc=True,
    )
    reported = pd.to_datetime(
        frame.get("filing_timestamp_utc", pd.Series(pd.NaT, index=frame.index)),
        errors="coerce",
        utc=True,
    )
    precision = (
        frame.get(
            "filing_time_precision", pd.Series("date_only", index=frame.index)
        )
        .fillna("date_only")
        .astype(str)
        .str.lower()
    )
    exact = precision.isin(
        [
            "second",
            "seconds",
            "minute",
            "minutes",
            "intraday",
            "exact_timestamp",
        ]
    )
    dates = (
        pd.to_datetime(frame.filing_date, errors="coerce")
        .dt.normalize()
        .dt.as_unit("ns")
    )
    coarse = (dates + pd.Timedelta(days=1)).dt.tz_localize(
        "America/New_York"
    ) - pd.Timedelta(nanoseconds=1)
    result = coarse.dt.tz_convert("UTC").dt.as_unit("ns").copy()
    precise = exact & reported.notna()
    result.loc[precise] = reported.dt.as_unit("ns").loc[precise]
    accepted_known = accepted.notna()
    result.loc[accepted_known] = accepted.dt.as_unit("ns").loc[accepted_known]
    return result


class FilingResearchStore:
    """Page through all authorized filings without loading the text corpus at once."""

    def __init__(self, path, *, training_end, start="2011-01-01"):
        self.path = Path(path)
        self.cutoff = as_of_timestamp(training_end)
        if as_of_timestamp(start) > self.cutoff:
            raise ValueError("Filing research start must not follow its declared cutoff")
        self.start = pd.Timestamp(start).date()
        self._metadata = None

    def metadata(self):
        if self._metadata is not None:
            return self._metadata
        parquet = pq.ParquetFile(self.path)
        wanted = [
            "document_id",
            "permno",
            "filing_date",
            "filing_timestamp_utc",
            "sec_accepted_at_utc",
            "filing_time_precision",
            "form_type",
            "items",
            "is_amendment",
            "word_count",
            "text_chars",
            "characteristics_link_basis",
            "characteristics_link_date",
        ]
        columns = [c for c in wanted if c in parquet.schema_arrow.names]
        required = {"document_id", "permno", "filing_date"}
        if not required <= set(columns):
            raise ValueError(
                "Supplied filing identifiers and publication date are required"
            )
        parts = []
        for group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(group, columns=columns).to_pandas()
            frame["_row_group"], frame["_row_in_group"] = (
                group,
                np.arange(len(frame)),
            )
            dates = pd.to_datetime(frame.filing_date, errors="coerce")
            frame = frame.loc[
                (dates.dt.date >= self.start)
                & (
                    dates.dt.date
                    <= self.cutoff.tz_convert("America/New_York").date()
                )
            ].copy()
            if frame.empty:
                continue
            frame["known_at"] = known_at(frame)
            frame = frame.loc[
                frame.known_at.notna() & (frame.known_at <= self.cutoff)
            ]
            parts.append(frame)
        if parts:
            self._metadata = (
                pd.concat(parts, ignore_index=True)
                .sort_values(["known_at", "document_id"])
                .reset_index(drop=True)
            )
        else:
            self._metadata = pd.DataFrame(
                columns=columns + ["_row_group", "_row_in_group", "known_at"]
            )
        return self._metadata

    def query(
        self,
        as_of,
        *,
        permnos=None,
        offset=0,
        limit=50,
        include_text=True,
        max_text_chars=None,
    ):
        """Return a local page with original text, identity provenance and source flags.

        Pagination is a per-request resource bound, not a corpus/universe cap.
        The caller can request subsequent pages. Nothing is sent to a model API.
        """
        stamp = as_of_timestamp(as_of)
        if stamp > self.cutoff:
            raise ValueError(
                "Requested filing date crosses the training boundary"
            )
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError(
                "Use a nonnegative offset and pages of 1–200 filings"
            )
        frame = self.metadata()
        frame = frame.loc[frame.known_at <= stamp]
        if permnos is not None:
            frame = frame.loc[frame.permno.isin(permnos)]
        selected = frame.iloc[offset : offset + limit].copy()
        if len(selected):
            parquet = pq.ParquetFile(self.path)
            columns = [
                c
                for c in parquet.schema_arrow.names
                if include_text or c != "text"
            ]
            pages = []
            for group, records in selected.groupby("_row_group"):
                table = parquet.read_row_group(int(group), columns=columns)
                table = table.take(
                    pa.array(records._row_in_group.to_numpy(), type=pa.int64())
                )
                page = table.to_pandas()
                page.index = records.index
                pages.append(page)
            original = pd.concat(pages).reindex(selected.index)
            original["known_at"] = selected.known_at
            selected = original
            if include_text and max_text_chars is not None:
                if max_text_chars < 1:
                    raise ValueError("A text preview must have positive length")
                selected["text_preview_truncated"] = (
                    selected.text.fillna("").str.len() > max_text_chars
                )
                selected["text"] = selected.text.str.slice(0, max_text_chars)
        selected["source_research_only"] = True
        selected["historical_identity_verified"] = False
        return selected.drop(
            columns=["_row_group", "_row_in_group"], errors="ignore"
        ).reset_index(drop=True)

    def feature_arrays(self, dates, securities):
        """Build lagged counts with no future coverage conditioning or text API calls."""
        dates = pd.DatetimeIndex(dates)
        instants = np.array(
            [as_of_timestamp(d).value for d in dates], dtype=np.int64
        )
        if len(instants) and instants.max() > self.cutoff.value:
            raise ValueError("Feature formation exceeds authorized training")
        result = {
            f: np.full((len(dates), len(securities)), np.nan, dtype=np.float32)
            for f in FILING_RESEARCH_FEATURES
        }
        locations = {int(p): i for i, p in enumerate(securities)}
        metadata = self.metadata()
        for security, records in metadata.loc[
            metadata.permno.isin(securities)
        ].groupby("permno"):
            column = locations[int(security)]
            records = records.sort_values("known_at")
            times = records.known_at.astype("int64").to_numpy()
            right = np.searchsorted(times, instants, side="right")
            seen = right > 0
            words = (
                pd.to_numeric(
                    records.get(
                        "word_count", pd.Series(0, index=records.index)
                    ),
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy()
            )
            amendments = (
                records.get(
                    "is_amendment", pd.Series(False, index=records.index)
                )
                .fillna(False)
                .astype(int)
                .to_numpy()
            )
            word_sum = np.r_[0, np.cumsum(words)]
            amendment_sum = np.r_[0, np.cumsum(amendments)]
            for days in (30, 90, 365):
                left = np.searchsorted(
                    times,
                    instants - days * 86_400_000_000_000,
                    side="right",
                )
                count = right - left
                result[f"filings_8k_count_{days}d"][seen, column] = count[seen]
                if days == 90:
                    result["filings_8k_words_90d"][seen, column] = (
                        word_sum[right] - word_sum[left]
                    )[seen]
                if days == 365:
                    share = np.divide(
                        amendment_sum[right] - amendment_sum[left],
                        count,
                        out=np.full(len(count), np.nan),
                        where=count > 0,
                    )
                    result["filings_8k_amendment_share_365d"][seen, column] = (
                        share[seen]
                    )
        return result

    def audit(self):
        frame = self.metadata()
        return {
            "training_documents": len(frame),
            "mapped_training_issuers": int(frame.permno.nunique()),
            "latest_availability_bound_utc": frame.known_at.max().isoformat()
            if len(frame)
            else None,
            "latest_publication_date_new_york": frame.known_at.max()
            .tz_convert("America/New_York")
            .date()
            .isoformat()
            if len(frame)
            else None,
            "raw_text_query_available": True,
            "cloud_transmission": False,
            "research_only": True,
            "limitations": list(SOURCE_LIMITATIONS),
        }
