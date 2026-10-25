"""Three-month analyst revision of the nearest open fiscal quarter per stock."""

import pathlib

from sleeves import base, ibes

_QUARTERLY_FPI = ("6", "7", "8", "9")


def path(work_dir: pathlib.Path) -> pathlib.Path:
    """Location of the near-quarter revision panel."""
    return work_dir / "conf_lm_gated" / "revisions.parquet"


def build(context: base.Context) -> pathlib.Path:
    """Writes rev3_near: consensus change over three months, scaled by price.

    Only snapshots dated on or before each month-end enter that month, and a
    quarter already announced at its snapshot is never the near quarter.
    """
    output = path(context.work_dir)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    consensus = ibes.load_consensus()
    consensus = consensus.loc[consensus.FPI.astype(str).isin(_QUARTERLY_FPI)].copy()
    consensus["eom"] = consensus.snap_month.dt.to_timestamp("M")
    announced = consensus.ANNDATS_ACT.notna() & (
        consensus.ANNDATS_ACT <= consensus.STATPERS
    )
    consensus = consensus.loc[~announced].merge(
        ibes.load_prices(),
        on=["TICKER", "STATPERS"],
        how="inner",
    )
    lagged = consensus[["TICKER", "FPEDATS", "snap_month", "MEANEST"]].rename(
        columns={"MEANEST": "lag3"}
    )
    lagged["snap_month"] = lagged.snap_month + 3
    consensus = consensus.merge(
        lagged, on=["TICKER", "FPEDATS", "snap_month"], how="left"
    )
    consensus["rev3_near"] = (consensus.MEANEST - consensus.lag3) / (consensus.PRICE)
    near = consensus.sort_values(["TICKER", "eom", "FPEDATS"]).drop_duplicates(
        ["TICKER", "eom"]
    )
    tickers = ibes.attach_ticker(context.chars[["permno", "eom"]], ibes.load_link())
    revisions = tickers.merge(
        near[["TICKER", "eom", "rev3_near"]], on=["TICKER", "eom"], how="inner"
    )
    revisions[["permno", "eom", "rev3_near"]].to_parquet(output, index=False)
    return output
