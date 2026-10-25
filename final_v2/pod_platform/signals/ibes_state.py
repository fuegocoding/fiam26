"""Point-in-time I/B/E/S state (faithful port of final/sleeves/ibes.py).

Same clock rules as the first round: dated ticker links with SCORE <= 2,
last consensus snapshot per ticker-month, earliest unannounced fiscal
quarter as the target, first-reported EPS only (no revisions), conflicting
records rejected. Loaders read external.parquet tables.
"""

from __future__ import annotations

import pandas as pd

_SNAPSHOT_TOLERANCE = pd.Timedelta(days=45)
_TARGET_MAX_AGE = pd.Timedelta(days=120)
_MAX_REPORT_LAG = pd.Timedelta(days=180)
_MAX_LINK_SCORE = 2

_EXTERNAL: str | None = None

_TABLE_COLUMNS = {
    "ibes_link": ["TICKER", "PERMNO", "sdate", "edate", "SCORE"],
    "ibes_consensus": [
        "TICKER",
        "STATPERS",
        "FPI",
        "CURCODE",
        "NUMEST",
        "NUMUP",
        "NUMDOWN",
        "MEDEST",
        "MEANEST",
        "STDEV",
        "USFIRM",
        "FPEDATS",
        "ANNDATS_ACT",
    ],
    "ibes_price": ["TICKER", "STATPERS", "PRICE", "CURR_PRICE", "USFIRM"],
    "ibes_surprise": [
        "TICKER",
        "USFIRM",
        "MEASURE",
        "FISCALP",
        "PYEAR",
        "PMON",
        "anndats",
        "actual",
        "surpmean",
        "suescore",
    ],
}


def set_external_path(path: str) -> None:
    global _EXTERNAL
    _EXTERNAL = str(path)


def _read(table: str) -> pd.DataFrame:
    if _EXTERNAL is None:
        raise ValueError("set_external_path must be called first")
    ext = pd.read_parquet(_EXTERNAL, columns=_TABLE_COLUMNS[table] + ["table"])
    return ext[ext["table"] == table][_TABLE_COLUMNS[table]].reset_index(
        drop=True
    )


def _unique_records(
    frame: pd.DataFrame, keys: list[str], source: str
) -> pd.DataFrame:
    frame = frame.drop_duplicates()
    if frame.duplicated(keys).any():
        raise ValueError(
            f"Conflicting {source} records without revision timestamps"
        )
    return frame


def month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values) + pd.offsets.MonthEnd(0)


def load_link() -> pd.DataFrame:
    link = _read("ibes_link")
    link = link.dropna(subset=["PERMNO"])
    link = link[link.SCORE <= _MAX_LINK_SCORE].copy()
    link["permno"] = link.PERMNO.astype(int)
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    return link[["TICKER", "permno", "sdate", "edate", "SCORE"]]


def load_consensus() -> pd.DataFrame:
    frame = _read("ibes_consensus")
    frame = frame[(frame.USFIRM == 1) & (frame.CURCODE == "USD")].drop(
        columns=["USFIRM", "CURCODE"]
    )
    for column in ("STATPERS", "FPEDATS", "ANNDATS_ACT"):
        frame[column] = pd.to_datetime(frame[column])
    frame = _unique_records(
        frame, ["TICKER", "STATPERS", "FPEDATS"], "consensus"
    )
    frame["snap_month"] = frame.STATPERS.dt.to_period("M")
    last = frame.groupby(["TICKER", "snap_month"]).STATPERS.transform("max")
    return frame[frame.STATPERS == last].reset_index(drop=True)


def load_prices() -> pd.DataFrame:
    frame = _read("ibes_price")
    frame = frame[
        (frame.USFIRM == 1) & (frame.CURR_PRICE == "USD") & (frame.PRICE > 0)
    ]
    frame["STATPERS"] = pd.to_datetime(frame["STATPERS"])
    frame = frame[["TICKER", "STATPERS", "PRICE"]]
    return _unique_records(frame, ["TICKER", "STATPERS"], "price")


def load_surprises() -> pd.DataFrame:
    frame = _read("ibes_surprise")
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
    keys = ["TICKER", "fpe", "anndats"]
    frame = frame.drop_duplicates(keys + ["actual", "surpmean", "suescore"])
    if frame.duplicated(keys).any():
        raise ValueError("Conflicting EPS records without revision timestamps")
    frame = frame.sort_values("anndats").drop_duplicates(
        ["TICKER", "fpe"], keep="first"
    )
    frame = frame.sort_values(["TICKER", "anndats"])
    return frame[
        ["TICKER", "fpe", "anndats", "actual", "surpmean", "suescore"]
    ].reset_index(drop=True)


def attach_ticker(grid: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    merged = grid[["permno", "eom"]].merge(link, on="permno", how="inner")
    merged = merged[(merged.sdate <= merged.eom) & (merged.eom <= merged.edate)]
    merged = merged.sort_values(
        ["permno", "eom", "SCORE", "sdate"], ascending=[True, True, True, False]
    )
    return merged.drop_duplicates(["permno", "eom"])[
        ["permno", "eom", "TICKER"]
    ]


def build_expectation_state(grid: pd.DataFrame) -> pd.DataFrame:
    grid = grid[["permno", "eom"]].drop_duplicates().copy()
    grid["eom"] = month_end(grid.eom)
    state = attach_ticker(grid, load_link())
    consensus = load_consensus()
    snapshots = consensus[["TICKER", "STATPERS", "snap_month"]].drop_duplicates(
        ["TICKER", "STATPERS"]
    )
    state = _asof(state, snapshots, "STATPERS", _SNAPSHOT_TOLERANCE)
    state = state.dropna(subset=["STATPERS"])
    candidates = state.merge(
        consensus.drop(columns="snap_month"),
        on=["TICKER", "STATPERS"],
        how="inner",
    )
    state = _select_target_quarter(candidates)
    state = state.merge(load_prices(), on=["TICKER", "STATPERS"], how="inner")
    targets = load_surprises()[
        ["TICKER", "fpe", "anndats", "actual", "surpmean"]
    ].rename(
        columns={
            "fpe": "target_fpe",
            "anndats": "target_ann",
            "actual": "target_actual",
            "surpmean": "target_surpmean",
        }
    )
    state["target_fpe"] = month_end(state.FPEDATS).to_numpy()
    state = state.merge(targets, on=["TICKER", "target_fpe"], how="left")
    future = (state.target_ann > state.eom) & (
        state.target_ann <= state.target_fpe + _MAX_REPORT_LAG
    )
    for column in ("target_ann", "target_actual", "target_surpmean"):
        state[column] = state[column].where(future)
    state["target_surprise_p"] = (
        state.target_actual - state.target_surpmean
    ) / state.PRICE
    keep = [
        "permno",
        "eom",
        "TICKER",
        "STATPERS",
        "FPEDATS",
        "PRICE",
        "target_fpe",
        "target_ann",
        "target_surprise_p",
    ]
    return state[keep].sort_values(["permno", "eom"]).reset_index(drop=True)


def _asof(
    left: pd.DataFrame, right: pd.DataFrame, right_on: str, tolerance=None
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
    announced = candidates.ANNDATS_ACT.notna() & (
        candidates.ANNDATS_ACT <= candidates.eom
    )
    fresh = candidates.FPEDATS >= candidates.eom - _TARGET_MAX_AGE
    open_quarters = candidates[~announced & fresh]
    open_quarters = open_quarters.sort_values(["permno", "eom", "FPEDATS"])
    return open_quarters.drop_duplicates(["permno", "eom"])
