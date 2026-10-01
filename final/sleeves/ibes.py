"""I/B/E/S loaders and the point-in-time analyst expectation state."""

import pathlib

import pandas as pd

CONSENSUS_FILE = "s365yejuuvprhila.csv.gz"
PRICE_FILE = "lhn8sdwvp01ym0q7.csv.gz"
SURPRISE_FILE = "b2tkpunzdr2n0s5t.csv.gz"
LINK_FILE = "j2vz7zigsgxzab3h.csv.gz"

_SNAPSHOT_TOLERANCE = pd.Timedelta(days=45)
_TARGET_MAX_AGE = pd.Timedelta(days=120)
_MAX_REPORT_LAG = pd.Timedelta(days=180)
_MAX_LINK_SCORE = 2


def month_end(values: pd.Series) -> pd.Series:
    """Rolls dates forward to their calendar month-end."""
    return pd.to_datetime(values) + pd.offsets.MonthEnd(0)


def load_link(ibes_dir: pathlib.Path) -> pd.DataFrame:
    """Dated ticker-permno links; the latest links in the extract stay open."""
    link = pd.read_csv(ibes_dir / LINK_FILE, dtype={"TICKER": str})
    link = link.dropna(subset=["PERMNO"])
    link = link[link.SCORE <= _MAX_LINK_SCORE].copy()
    link["permno"] = link.PERMNO.astype(int)
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    link.loc[link.edate >= link.edate.max(), "edate"] = pd.Timestamp(
        "2099-12-31"
    )
    return link[["TICKER", "permno", "sdate", "edate", "SCORE"]]


def load_consensus(ibes_dir: pathlib.Path) -> pd.DataFrame:
    """US-dollar consensus rows from each ticker's last snapshot per month."""
    columns = [
        "TICKER", "STATPERS", "FPI", "FPEDATS", "NUMEST", "NUMUP", "NUMDOWN",
        "MEDEST", "MEANEST", "STDEV", "ANNDATS_ACT", "USFIRM", "CURCODE",
    ]
    frame = pd.read_csv(
        ibes_dir / CONSENSUS_FILE, usecols=columns, dtype={"TICKER": str}
    )
    frame = frame[(frame.USFIRM == 1) & (frame.CURCODE == "USD")].drop(
        columns=["USFIRM", "CURCODE"]
    )
    for column in ("STATPERS", "FPEDATS", "ANNDATS_ACT"):
        frame[column] = pd.to_datetime(frame[column])
    frame = frame.drop_duplicates(
        ["TICKER", "STATPERS", "FPEDATS"], keep="last"
    )
    frame["snap_month"] = frame.STATPERS.dt.to_period("M")
    last = frame.groupby(["TICKER", "snap_month"]).STATPERS.transform("max")
    return frame[frame.STATPERS == last].reset_index(drop=True)


def load_prices(ibes_dir: pathlib.Path) -> pd.DataFrame:
    """I/B/E/S share price of each snapshot, on the per-share EPS basis."""
    frame = pd.read_csv(
        ibes_dir / PRICE_FILE,
        usecols=["TICKER", "STATPERS", "PRICE", "CURR_PRICE", "USFIRM"],
        dtype={"TICKER": str},
    )
    frame = frame[
        (frame.USFIRM == 1) & (frame.CURR_PRICE == "USD") & (frame.PRICE > 0)
    ]
    frame["STATPERS"] = pd.to_datetime(frame.STATPERS)
    return frame.drop_duplicates(["TICKER", "STATPERS"], keep="last")[
        ["TICKER", "STATPERS", "PRICE"]
    ]


def load_surprises(ibes_dir: pathlib.Path) -> pd.DataFrame:
    """Quarterly EPS actuals and surprises, one row per ticker and quarter."""
    frame = pd.read_csv(ibes_dir / SURPRISE_FILE, dtype={"TICKER": str})
    frame = frame[
        (frame.USFIRM == 1)
        & (frame.MEASURE == "EPS")
        & (frame.FISCALP == "QTR")
    ]
    frame = frame.dropna(subset=["actual", "surpmean", "anndats"]).copy()
    frame["fpe"] = month_end(
        pd.to_datetime(dict(year=frame.PYEAR, month=frame.PMON, day=1))
    )
    frame["anndats"] = pd.to_datetime(frame.anndats)
    frame = frame.sort_values("anndats").drop_duplicates(
        ["TICKER", "fpe"], keep="last"
    )
    frame = frame.sort_values(["TICKER", "anndats"])
    return frame[
        ["TICKER", "fpe", "anndats", "actual", "surpmean", "suescore"]
    ].reset_index(drop=True)


def attach_ticker(grid: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Best dated I/B/E/S ticker for every (permno, eom)."""
    merged = grid[["permno", "eom"]].merge(link, on="permno", how="inner")
    merged = merged[(merged.sdate <= merged.eom) & (merged.eom <= merged.edate)]
    merged = merged.sort_values(
        ["permno", "eom", "SCORE", "sdate"], ascending=[True, True, True, False]
    )
    return merged.drop_duplicates(["permno", "eom"])[
        ["permno", "eom", "TICKER"]
    ]


def build_expectation_state(
    grid: pd.DataFrame, ibes_dir: pathlib.Path
) -> pd.DataFrame:
    """Open-quarter consensus at each month-end plus its future surprise.

    Columns prefixed `target_` describe the later announcement of that
    quarter. They are labels for event timing, never formation features.
    """
    grid = grid[["permno", "eom"]].drop_duplicates().copy()
    grid["eom"] = month_end(grid.eom)
    state = attach_ticker(grid, load_link(ibes_dir))
    consensus = load_consensus(ibes_dir)
    snapshots = consensus[["TICKER", "STATPERS", "snap_month"]]
    snapshots = snapshots.drop_duplicates(["TICKER", "STATPERS"])
    state = _asof(state, snapshots, "STATPERS", _SNAPSHOT_TOLERANCE)
    state = state.dropna(subset=["STATPERS"])
    candidates = state.merge(
        consensus.drop(columns="snap_month"),
        on=["TICKER", "STATPERS"],
        how="inner",
    )
    state = _select_target_quarter(candidates)
    state = state.merge(
        load_prices(ibes_dir), on=["TICKER", "STATPERS"], how="inner"
    )
    targets = load_surprises(ibes_dir)[
        ["TICKER", "fpe", "anndats", "actual", "surpmean"]
    ].rename(columns={
        "fpe": "target_fpe",
        "anndats": "target_ann",
        "actual": "target_actual",
        "surpmean": "target_surpmean",
    })
    state["target_fpe"] = month_end(state.FPEDATS).to_numpy()
    state = state.merge(targets, on=["TICKER", "target_fpe"], how="left")
    # A label counts only if released after t, within a plausible report lag.
    future = (state.target_ann > state.eom) & (
        state.target_ann <= state.target_fpe + _MAX_REPORT_LAG
    )
    for column in ("target_ann", "target_actual", "target_surpmean"):
        state[column] = state[column].where(future)
    state["target_surprise_p"] = (
        state.target_actual - state.target_surpmean
    ) / state.PRICE
    keep = [
        "permno", "eom", "TICKER", "STATPERS", "FPEDATS", "PRICE",
        "target_fpe", "target_ann", "target_surprise_p",
    ]
    return state[keep].sort_values(["permno", "eom"]).reset_index(drop=True)


def _asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    right_on: str,
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    left = left.assign(eom=left.eom.astype("datetime64[ns]"))
    right = right.assign(**{right_on: right[right_on].astype("datetime64[ns]")})
    return pd.merge_asof(
        left.sort_values("eom"),
        right.sort_values(right_on),
        left_on="eom",
        right_on=right_on,
        by="TICKER",
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    )


def _select_target_quarter(candidates: pd.DataFrame) -> pd.DataFrame:
    """Earliest fiscal quarter in the snapshot not yet announced by t."""
    announced = candidates.ANNDATS_ACT.notna() & (
        candidates.ANNDATS_ACT <= candidates.eom
    )
    fresh = candidates.FPEDATS >= candidates.eom - _TARGET_MAX_AGE
    open_quarters = candidates[~announced & fresh]
    open_quarters = open_quarters.sort_values(["permno", "eom", "FPEDATS"])
    return open_quarters.drop_duplicates(["permno", "eom"])
