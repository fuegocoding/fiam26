"""Delayed-reporter panel of earnings surprise and post-report breadth."""

import pathlib

import pandas as pd

from sleeves import base, ibes


def path(work_dir: pathlib.Path) -> pathlib.Path:
    """Location of the delayed-reporter panel."""
    return work_dir / "e7" / "delayed.parquet"


def build(context: base.Context) -> pathlib.Path:
    """Writes each report's surprise and breadth one month after the report.

    A quarter enters the month after its report month, once the first
    post-announcement consensus snapshot is public.
    """
    output = path(context.work_dir)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    chars = context.chars
    state = ibes.build_expectation_state(chars[["permno", "eom"]])
    state["eom"] = pd.to_datetime(state.eom)
    events = state.dropna(subset=["target_ann", "target_surprise_p"])
    events = events.loc[events.target_ann > events.eom]
    events = (
        events.sort_values("eom")
        .drop_duplicates(["permno", "target_fpe"], keep="last")
        .copy()
    )
    events["report_eom"] = events.eom + pd.offsets.MonthEnd(1)
    events = events.loc[events.target_ann <= events.report_eom].rename(
        columns={"eom": "forecast_eom", "target_surprise_p": "sue"}
    )
    consensus = ibes.load_consensus()[
        ["TICKER", "STATPERS", "FPEDATS", "NUMUP", "NUMDOWN", "NUMEST"]
    ]
    key = ["permno", "target_fpe"]
    # A signal can use only the report month's and following month's public
    # snapshots. Join those date windows separately to avoid a full-history
    # ticker join, which used several GB of temporary memory.
    snapshots = consensus.sort_values("STATPERS").set_index("STATPERS", drop=False)
    formation = ibes.month_end(events.target_ann) + pd.offsets.MonthEnd(1)
    parts = []
    for cutoff, rows in events.groupby(formation, sort=True):
        known = snapshots.loc[rows.target_ann.min() : cutoff].reset_index(drop=True)
        pairs = rows[["permno", "TICKER", "target_fpe", "target_ann"]].merge(
            known, on="TICKER"
        )
        following = pairs.loc[
            (pairs.FPEDATS > pairs.target_fpe)
            & (pairs.FPEDATS <= pairs.target_fpe + pd.offsets.MonthEnd(4))
            & (pairs.STATPERS > pairs.target_ann)
        ]
        post = following.loc[
            following.FPEDATS == following.groupby(key).FPEDATS.transform("min")
        ]
        parts.append(post.sort_values("STATPERS").groupby(key).head(1))
    post = pd.concat(parts, ignore_index=True)
    post["post_breadth"] = (post.NUMUP - post.NUMDOWN) / post.NUMEST.where(
        post.NUMEST > 0
    )
    events = events.merge(
        post[key + ["post_breadth", "STATPERS"]].rename(
            columns={"STATPERS": "post_snapshot"}
        ),
        on=key,
        how="left",
    )
    delayed = events.assign(eom=events.report_eom + pd.offsets.MonthEnd(1))
    delayed = delayed.loc[delayed.post_snapshot <= delayed.eom].merge(
        chars[["permno", "eom", "sector", "size_grp", "tradable"]],
        on=["permno", "eom"],
        how="inner",
    )
    columns = [
        "permno",
        "eom",
        "sector",
        "size_grp",
        "tradable",
        "sue",
        "post_breadth",
    ]
    delayed[columns].to_parquet(output, index=False)
    return output
