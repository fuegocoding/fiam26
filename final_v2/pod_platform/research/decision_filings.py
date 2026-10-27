"""Host-owned filing windows at native decision dates, without rereading text."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .filing_access import as_of_timestamp
from .filing_text import TEXT_FEATURES

DAY_NS = 86_400_000_000_000


class DecisionFilingFeatures:
    """Index only dated measurements; generated code receives ranked vectors.

    The index keeps publication times and cumulative sums per issuer, rather
    than materializing a daily stock-by-feature cube. A previously unseen issuer
    stays missing; a covered issuer with no recent event can have a zero count.
    """

    def __init__(self, metadata, securities, cutoff):
        self.locations = {int(p): i for i, p in enumerate(securities)}
        self.cutoff = as_of_timestamp(cutoff)
        self.indices = {}
        records = metadata.loc[metadata.permno.isin(securities)].copy()
        records["_count"] = 1.0
        records["_words"] = pd.to_numeric(records.get("word_count", pd.Series(0, index=records.index)), errors="coerce").fillna(0)
        records["_amendment"] = records.get("is_amendment", pd.Series(False, index=records.index)).fillna(False).astype(float)
        self.register(records, {
            **{f"filings_8k_count_{days}d": ("_count", days, "sum") for days in (30, 90, 365)},
            "filings_8k_words_90d": ("_words", 90, "sum"),
            "filings_8k_amendment_share_365d": ("_amendment", 365, "mean"),
        })

    def register(self, records, definitions):
        """Definitions map feature to measured column, calendar window and reducer."""
        for name, (column, days, aggregation) in definitions.items():
            if aggregation not in {"sum", "mean"} or not 1 <= days <= 365:
                raise ValueError("Use a bounded sum/mean filing window")
            groups = {}
            for security, frame in records.loc[records.permno.isin(self.locations)].groupby("permno"):
                frame = frame.sort_values("known_at")
                times = pd.DatetimeIndex(frame.known_at).as_unit("ns").asi8
                if np.any(times > self.cutoff.value):
                    raise ValueError("Filing measurement crosses the research cutoff")
                values = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(float)
                groups[self.locations[int(security)]] = (times, np.r_[0.0, values.cumsum()])
            self.indices[name] = (days, aggregation, groups)

    def register_text(self, records):
        self.register(records, {name: (name, 90, "mean" if "text_" in name else "sum") for name in TEXT_FEATURES})

    def register_custom(self, records, specifications):
        self.register(records, {s.name: (s.name, s.window_days, "mean" if s.aggregation == "frequency" else "sum") for s in specifications})

    def __call__(self, as_of, requested, mask):
        instant = as_of_timestamp(as_of).value
        if instant > self.cutoff.value:
            raise ValueError("Decision filing request crosses the training cutoff")
        result = {}
        for name in set(requested) & self.indices.keys():
            days, aggregation, groups = self.indices[name]
            raw = np.full(len(self.locations), np.nan)
            for location, (times, cumulative) in groups.items():
                right = times.searchsorted(instant, side="right")
                if not right:
                    continue
                left = times.searchsorted(instant - days * DAY_NS, side="right")
                count = right - left
                if aggregation == "sum":
                    raw[location] = cumulative[right] - cumulative[left]
                elif count:
                    raw[location] = (cumulative[right] - cumulative[left]) / count
            result[name] = (pd.Series(raw).where(mask).rank(pct=True).to_numpy() * 2 - 1).astype(np.float32)
        return result
