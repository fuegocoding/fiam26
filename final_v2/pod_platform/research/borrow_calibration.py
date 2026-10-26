"""Disclosed current-quote calibration of a simple size-tier cost model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def calibrate_borrow(quote_path, panel_path, size_asof):
    """Estimate costs without reading any return or alpha-feature columns.

    Current broker inventory is a selected sample. A rounded-up size-tier mean
    is an explicit modelling assumption, not a historical loan quote. Current
    inventory never determines historical eligibility or universe membership.
    """
    quotes = pd.read_csv(quote_path, sep="|", skiprows=1, dtype=str)
    required = {"#SYM", "CUR", "NAME", "FEERATE"}
    if not required <= set(quotes.columns):
        raise ValueError("Unrecognized broker fee schema")
    quotes["fee_bps"] = pd.to_numeric(quotes.FEERATE, errors="coerce") * 100
    quotes = quotes[(quotes.CUR == "USD") & np.isfinite(quotes.fee_bps)]
    if (quotes.fee_bps < 0).any():
        raise ValueError("Negative stock-loan fee requires explicit review")
    quotes = quotes[~quotes["#SYM"].duplicated(keep=False)]
    # The challenge explicitly permits disclosed current cost calibration.
    # Only identifiers and size are read, never held-out stock returns.
    columns = ["permno", "eom", "me", "ticker"]
    stocks = pq.read_table(
        panel_path,
        columns=columns,
        filters=[("eom", "==", pd.Timestamp(size_asof))],
        use_threads=False,
    ).to_pandas()
    if stocks.empty or stocks.permno.duplicated().any():
        raise ValueError(
            "Require a unique, populated size calibration snapshot"
        )
    stocks = stocks[~stocks.ticker.duplicated(keep=False)]
    matched = stocks.merge(
        quotes[["#SYM", "fee_bps"]], left_on="ticker", right_on="#SYM"
    )
    groups = {
        "large_cap": matched[matched.me >= 10000],
        "other_eligible": matched[(matched.me >= 500) & (matched.me < 10000)],
    }
    summaries, parameters = {}, {}
    for name, frame in groups.items():
        if len(frame) < 100:
            raise ValueError(
                "Insufficient quote matches for a broad tier model"
            )
        values = frame.fee_bps
        summaries[name] = {
            "names": len(frame),
            "mean_bps": float(values.mean()),
            "median_bps": float(values.median()),
            "p95_bps": float(values.quantile(0.95)),
        }
        key = (
            "large_cap_borrow_bps"
            if name == "large_cap"
            else "other_borrow_bps"
        )
        parameters[key] = float(np.ceil(values.mean() / 25) * 25)
    with Path(quote_path).open() as stream:
        provider_stamp = stream.readline().strip()
    return {
        "status": "current_quote_derived_cost_model",
        "source": "ftp://ftp2.interactivebrokers.com/usa.txt",
        "provider_header": provider_stamp,
        "quote_sha256": digest(quote_path),
        "panel_sha256": digest(panel_path),
        "size_asof": str(pd.Timestamp(size_asof).date()),
        "size_columns_read": columns,
        "return_columns_read": [],
        "quote_fee_units": "Annual percentage points multiplied by100 to obtain basis points",
        "calibration_policy": "Equal-name mean within each eligible size tier, rounded upward to the next25bp; no return-based fitting",
        "size_thresholds_usd": {"short_floor": 500e6, "large_cap": 10e9},
        "parameters": parameters,
        "matched_groups": summaries,
        "inventory_policy": "Current availability is not used to filter historical stocks or claim historical locates",
        "limitations": [
            "Current broker inventory is survivor-conditioned and does not cover every issuer",
            "Exact unique current-ticker matching is approximate; broker ISINs are masked",
            "Size snapshot predates the broker quote date; tiers approximate stock-loan economics",
            "Flat tier fees do not capture individual specials, changing fees or recalls",
            "Forward-dated cost calibration is disclosed under the challenge cost exception; this is not historical quote evidence",
        ],
    }
