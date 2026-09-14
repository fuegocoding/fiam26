"""Sell-side expectations, analyst-error history, graph peer surprises and the surprise model.

The research behind this module is summarized in ``docs/research_log.md``. This
is the reusable part: a point-in-time description of what the Street expects
for every (permno, month-end) and a walk-forward model of where it is wrong.

    PYTHONPATH=.deps OPENBLAS_NUM_THREADS=4 python -m src.graph.expectations                       # challenge schedule
    PYTHONPATH=.deps OPENBLAS_NUM_THREADS=4 python -m src.graph.expectations --validation-years 0  # prequential variant

Sources: WRDS I/B/E/S summary extracts in ``local/`` (US firms, USD), the
characteristics panel in ``data/raw/`` and the monthly economic graph in
``output/economic_graph_refined/`` (built by ``src.graph.economic_graph``).

- ``s365yejuuvprhila``: quarterly EPS consensus snapshots (statsum, FPI 6-9)
- ``lhn8sdwvp01ym0q7``: actuals and pricing snapshots (actpsum; PRICE)
- ``b2tkpunzdr2n0s5t``: quarterly EPS surprise history (surpsum)
- ``cln59hjwxfpwgvza``: recommendation consensus snapshots (recdsum)
- ``j2vz7zigsgxzab3h``: I/B/E/S ticker to CRSP PERMNO link (iclink)

Timing rules
------------
- A consensus snapshot (STATPERS, mid-month) is usable at month-end t when
  STATPERS <= t and t - STATPERS <= 45 days.
- ACTUAL and ANNDATS_ACT are filled retroactively on snapshot rows. They only
  decide whether a fiscal quarter had already been announced by t
  (ANNDATS_ACT <= t); their values are never used before that date.
- Surprise-history rows become usable once anndats <= t.
- Every per-share value is on the split-adjusted I/B/E/S basis and is scaled by
  the I/B/E/S PRICE of the same snapshot, which uses the same basis.
- Columns prefixed ``target_`` describe the future announcement of the next
  unreported quarter. They are labels, never features.
- statsum rounds MEANEST to cents on the adjusted basis (noisy revisions for
  heavy splitters); surprise labels use surpsum, which keeps full precision.

Surprise model
--------------
Target: within-month rank of the next analyst EPS surprise (actual minus the
last pre-announcement consensus, scaled by price). Features: Street state,
analyst-error history, default-graph peer signals and the 147 panel
characteristics plus the seasonal EPS forecast gap, all ranked within month.
XGBoost with fixed parameters, refit once per test year 2021-2026 on
``src.graph.walk_forward.split_masks`` (holding month t+1, label public by the
training end). Validation rows are scored, never fitted.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from src.graph.walk_forward import split_masks


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
LOCAL = ROOT / "local"
GRAPH_ROOT = ROOT / "output" / "economic_graph_refined"
OUTPUT = ROOT / "output" / "expectations"
CHARACTERISTICS_FILE = RAW / "chars_final_with_names.parquet"
FACTOR_LIST_FILE = RAW / "factor_char_list.csv"
CONSENSUS_FILE = LOCAL / "s365yejuuvprhila.csv.gz"
PRICING_FILE = LOCAL / "lhn8sdwvp01ym0q7.csv.gz"
SURPRISE_FILE = LOCAL / "b2tkpunzdr2n0s5t.csv.gz"
RECOMMENDATION_FILE = LOCAL / "cln59hjwxfpwgvza.csv.gz"
LINK_FILE = LOCAL / "j2vz7zigsgxzab3h.csv.gz"
MONTH_DIRECTORY = re.compile(r"^\d{4}-\d{2}$")

SNAPSHOT_TOLERANCE = pd.Timedelta(days=45)
TARGET_MAX_AGE = pd.Timedelta(days=120)
DEFAULT_REPORT_LAG_DAYS = 40
MAX_REPORT_LAG = pd.Timedelta(days=180)
MAX_LINK_SCORE = 2
FRESH_PEER_DAYS = 92
ALIGN_DAYS = 46
TEST_YEARS = tuple(range(2021, 2027))

STREET = ["cons_p", "disp_p", "breadth", "staleness_days", "rev1_p", "rev3_p", "NUMEST", "MEANREC",
          "rec_chg3", "rec_breadth", "sue_last_p", "last_suescore", "days_since_ann"]
HISTORY = ["hist_sue4_p", "hist_sue8_p", "sue_vol8_p", "beat_rate8", "exp_growth_p", "days_overdue", "rev3_next_p"]
GRAPH = ["peer_surprise_recent", "peer_surprise_aligned", "peer_revision3", "peer_beat_rate8"]
MODEL_PARAMS = {
    "objective": "reg:squarederror", "eta": 0.03, "max_depth": 5, "min_child_weight": 200,
    "subsample": 0.7, "colsample_bytree": 0.7, "lambda": 5.0, "tree_method": "hist", "seed": 7,
}
MODEL_ROUNDS = 400


def month_end(values) -> pd.Series:
    return pd.to_datetime(values) + pd.offsets.MonthEnd(0)


def winsorize(values, quantile: float = 0.01) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.isfinite(values).sum() < 10:
        return values
    low, high = np.nanquantile(values, [quantile, 1 - quantile])
    return np.clip(values, low, high)


# --------------------------------------------------------------------------
# I/B/E/S loaders
# --------------------------------------------------------------------------

def load_link(path: Path = LINK_FILE) -> pd.DataFrame:
    """Dated ticker-permno links; the latest links in the extract stay open."""
    link = pd.read_csv(path, dtype={"TICKER": str})
    link = link.dropna(subset=["PERMNO"])
    link = link[link.SCORE <= MAX_LINK_SCORE].copy()
    link["permno"] = link.PERMNO.astype(int)
    link["sdate"] = pd.to_datetime(link.sdate)
    link["edate"] = pd.to_datetime(link.edate)
    link.loc[link.edate >= link.edate.max(), "edate"] = pd.Timestamp("2099-12-31")
    return link[["TICKER", "permno", "sdate", "edate", "SCORE"]]


def load_consensus(path: Path = CONSENSUS_FILE) -> pd.DataFrame:
    columns = ["TICKER", "STATPERS", "FPI", "FPEDATS", "NUMEST", "NUMUP", "NUMDOWN",
               "MEDEST", "MEANEST", "STDEV", "ANNDATS_ACT", "USFIRM", "CURCODE"]
    frame = pd.read_csv(path, usecols=columns, dtype={"TICKER": str})
    frame = frame[(frame.USFIRM == 1) & (frame.CURCODE == "USD")].drop(columns=["USFIRM", "CURCODE"])
    for column in ("STATPERS", "FPEDATS", "ANNDATS_ACT"):
        frame[column] = pd.to_datetime(frame[column])
    frame = frame.drop_duplicates(["TICKER", "STATPERS", "FPEDATS"], keep="last")
    frame["snap_month"] = frame.STATPERS.dt.to_period("M")
    last = frame.groupby(["TICKER", "snap_month"]).STATPERS.transform("max")
    return frame[frame.STATPERS == last].reset_index(drop=True)


def load_prices(path: Path = PRICING_FILE) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["TICKER", "STATPERS", "PRICE", "CURR_PRICE", "USFIRM"],
                        dtype={"TICKER": str})
    frame = frame[(frame.USFIRM == 1) & (frame.CURR_PRICE == "USD") & (frame.PRICE > 0)]
    frame["STATPERS"] = pd.to_datetime(frame.STATPERS)
    return frame.drop_duplicates(["TICKER", "STATPERS"], keep="last")[["TICKER", "STATPERS", "PRICE"]]


def load_surprises(path: Path = SURPRISE_FILE) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"TICKER": str})
    frame = frame[(frame.USFIRM == 1) & (frame.MEASURE == "EPS") & (frame.FISCALP == "QTR")]
    frame = frame.dropna(subset=["actual", "surpmean", "anndats"]).copy()
    frame["fpe"] = month_end(pd.to_datetime(dict(year=frame.PYEAR, month=frame.PMON, day=1)))
    frame["anndats"] = pd.to_datetime(frame.anndats)
    frame = frame.sort_values("anndats").drop_duplicates(["TICKER", "fpe"], keep="last")
    frame["report_lag_days"] = (frame.anndats - frame.fpe).dt.days
    frame = frame.sort_values(["TICKER", "anndats"])
    frame["lag_median4"] = (
        frame.groupby("TICKER").report_lag_days
        .transform(lambda s: s.rolling(4, min_periods=1).median())
    )
    return frame[["TICKER", "fpe", "anndats", "actual", "surpmean", "surpstdev",
                  "suescore", "lag_median4"]].reset_index(drop=True)


def load_recommendations(path: Path = RECOMMENDATION_FILE) -> pd.DataFrame:
    columns = ["TICKER", "STATPERS", "MEANREC", "NUMREC", "NUMUP", "NUMDOWN", "USFIRM"]
    frame = pd.read_csv(path, usecols=columns, dtype={"TICKER": str})
    frame = frame[frame.USFIRM == 1].drop(columns="USFIRM")
    frame["STATPERS"] = pd.to_datetime(frame.STATPERS)
    frame["snap_month"] = frame.STATPERS.dt.to_period("M")
    frame = frame.sort_values("STATPERS").drop_duplicates(["TICKER", "snap_month"], keep="last")
    return frame.rename(columns={"STATPERS": "rec_statpers", "NUMUP": "rec_up", "NUMDOWN": "rec_down"})


# --------------------------------------------------------------------------
# Expectation state
# --------------------------------------------------------------------------

def attach_ticker(grid: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Best dated I/B/E/S ticker for every (permno, eom)."""
    merged = grid[["permno", "eom"]].merge(link, on="permno", how="inner")
    merged = merged[(merged.sdate <= merged.eom) & (merged.eom <= merged.edate)]
    merged = merged.sort_values(["permno", "eom", "SCORE", "sdate"], ascending=[True, True, True, False])
    return merged.drop_duplicates(["permno", "eom"])[["permno", "eom", "TICKER"]]


def _asof(left: pd.DataFrame, right: pd.DataFrame, right_on: str, tolerance=None) -> pd.DataFrame:
    left = left.assign(eom=left.eom.astype("datetime64[ns]"))
    right = right.assign(**{right_on: right[right_on].astype("datetime64[ns]")})
    return pd.merge_asof(
        left.sort_values("eom"), right.sort_values(right_on),
        left_on="eom", right_on=right_on, by="TICKER",
        direction="backward", allow_exact_matches=True, tolerance=tolerance,
    )


def select_target_quarter(candidates: pd.DataFrame) -> pd.DataFrame:
    """Earliest fiscal quarter in the snapshot not yet announced by month-end t."""
    announced = candidates.ANNDATS_ACT.notna() & (candidates.ANNDATS_ACT <= candidates.eom)
    fresh = candidates.FPEDATS >= candidates.eom - TARGET_MAX_AGE
    open_quarters = candidates[~announced & fresh]
    open_quarters = open_quarters.sort_values(["permno", "eom", "FPEDATS"])
    return open_quarters.drop_duplicates(["permno", "eom"])


def seasonal_persistence_by_year(surprises: pd.DataFrame, years: range) -> dict[int, float]:
    """Pooled AR(1) slope of seasonal EPS changes, fit only on earlier announcements."""
    actual = surprises.set_index(["TICKER", "fpe"]).actual
    frame = surprises[["TICKER", "fpe", "anndats", "actual"]].copy()
    for months, name in ((3, "a3"), (12, "a12"), (15, "a15")):
        key = pd.MultiIndex.from_arrays([frame.TICKER, month_end(frame.fpe - pd.DateOffset(months=months))])
        frame[name] = actual.reindex(key).to_numpy()
    scale = frame[["actual", "a3", "a12", "a15"]].abs().mean(axis=1) + 0.01
    frame["d"] = (frame.actual - frame.a12) / scale
    frame["d_lag"] = (frame.a3 - frame.a15) / scale
    frame = frame.dropna(subset=["d", "d_lag"])
    for column in ("d", "d_lag"):
        low, high = frame[column].quantile([0.01, 0.99])
        frame[column] = frame[column].clip(low, high)
    slopes = {}
    for year in years:
        past = frame[frame.anndats < pd.Timestamp(year=year, month=1, day=1)]
        denominator = float(past.d_lag @ past.d_lag)
        slopes[year] = float(past.d @ past.d_lag / denominator) if len(past) > 1000 and denominator > 0 else np.nan
    return slopes


def build_expectation_state(grid: pd.DataFrame) -> pd.DataFrame:
    """Expectation state and future-announcement labels for (permno, eom) rows."""
    grid = grid[["permno", "eom"]].drop_duplicates().copy()
    grid["eom"] = month_end(grid.eom)
    state = attach_ticker(grid, load_link())

    consensus = load_consensus()
    snapshots = consensus[["TICKER", "STATPERS", "snap_month"]].drop_duplicates(["TICKER", "STATPERS"])
    state = _asof(state, snapshots, "STATPERS", SNAPSHOT_TOLERANCE).dropna(subset=["STATPERS"])

    candidates = state.merge(consensus.drop(columns="snap_month"), on=["TICKER", "STATPERS"], how="inner")
    state = select_target_quarter(candidates)
    state = state.merge(load_prices(), on=["TICKER", "STATPERS"], how="inner")

    price = state.PRICE
    state["cons_p"] = state.MEANEST / price
    state["disp_p"] = state.STDEV / price
    state["breadth"] = (state.NUMUP - state.NUMDOWN) / state.NUMEST
    state["staleness_days"] = (state.eom - state.STATPERS).dt.days

    lagged = consensus[["TICKER", "FPEDATS", "snap_month", "MEANEST"]]
    for months in (1, 3):
        shifted = lagged.assign(snap_month=lagged.snap_month + months)
        shifted = shifted.rename(columns={"MEANEST": f"meanest_lag{months}"})
        state = state.merge(shifted, on=["TICKER", "FPEDATS", "snap_month"], how="left")
        state[f"rev{months}_p"] = (state.MEANEST - state[f"meanest_lag{months}"]) / state.PRICE

    recommendations = load_recommendations()
    state = _asof(state, recommendations.drop(columns="snap_month"), "rec_statpers", SNAPSHOT_TOLERANCE)
    state["rec_month"] = state.rec_statpers.dt.to_period("M")
    rec_lag = recommendations[["TICKER", "snap_month", "MEANREC"]].assign(
        snap_month=recommendations.snap_month + 3).rename(
        columns={"snap_month": "rec_month", "MEANREC": "meanrec_lag3"})
    state = state.merge(rec_lag, on=["TICKER", "rec_month"], how="left")
    state["rec_chg3"] = state.MEANREC - state.meanrec_lag3
    state["rec_breadth"] = (state.rec_up - state.rec_down) / state.NUMREC

    surprises = load_surprises()
    last = surprises.rename(columns={
        "fpe": "last_fpe", "anndats": "last_ann", "actual": "last_actual",
        "surpmean": "last_surpmean", "suescore": "last_suescore", "surpstdev": "last_surpstdev",
    })
    state = _asof(state, last, "last_ann")
    price = state.PRICE
    state["sue_last_p"] = (state.last_actual - state.last_surpmean) / price
    state["days_since_ann"] = (state.eom - state.last_ann).dt.days
    lag_days = state.lag_median4.fillna(DEFAULT_REPORT_LAG_DAYS)
    state["expected_ann"] = state.FPEDATS + pd.to_timedelta(lag_days, unit="D")
    next_month_end = state.eom + pd.offsets.MonthEnd(1)
    state["expected_report_next_month"] = (state.expected_ann > state.eom) & (state.expected_ann <= next_month_end)

    # Own time-series fundamental forecast (Foster seasonal AR(1)) from reported quarters only.
    by_quarter = surprises.set_index(["TICKER", "fpe"])
    target_fpe = month_end(state.FPEDATS)
    for months, name in ((3, "a3"), (12, "a12"), (15, "a15")):
        key = pd.MultiIndex.from_arrays([state.TICKER, month_end(target_fpe - pd.DateOffset(months=months))])
        values = by_quarter.actual.reindex(key).to_numpy()
        announced = pd.to_datetime(by_quarter.anndats.reindex(key).to_numpy())
        known = announced.notna() & (announced <= pd.DatetimeIndex(state.eom))
        state[name] = np.where(known, values, np.nan)
    years = range(int(state.eom.dt.year.min()), int(state.eom.dt.year.max()) + 1)
    phi = state.eom.dt.year.map(seasonal_persistence_by_year(surprises, years))
    state["eps_hat"] = state.a12 + (phi * (state.a3 - state.a15)).fillna(0.0)
    state["fgap_p"] = (state.eps_hat - state.MEANEST) / price
    state["fgap_disp"] = (state.eps_hat - state.MEANEST) / (state.STDEV.fillna(0) + 0.001 * price)

    targets = surprises[["TICKER", "fpe", "anndats", "actual", "surpmean"]].rename(columns={
        "fpe": "target_fpe", "anndats": "target_ann", "actual": "target_actual",
        "surpmean": "target_surpmean",
    })
    state["target_fpe"] = target_fpe.to_numpy()
    state = state.merge(targets, on=["TICKER", "target_fpe"], how="left")
    price = state.PRICE
    next_month_end = state.eom + pd.offsets.MonthEnd(1)
    # A label must be released after t and within a plausible reporting window of its quarter end.
    future = (state.target_ann > state.eom) & (state.target_ann <= state.target_fpe + MAX_REPORT_LAG)
    for column in ("target_ann", "target_actual", "target_surpmean"):
        state[column] = state[column].where(future)
    state["target_surprise_p"] = (state.target_actual - state.target_surpmean) / price
    state["target_error_vs_now_p"] = (state.target_actual - state.MEANEST) / price
    state["target_future_revision_p"] = (state.target_surpmean - state.MEANEST) / price
    state["target_reports_next_month"] = future & (state.target_ann <= next_month_end)

    keep = [
        "permno", "eom", "TICKER", "STATPERS", "FPI", "FPEDATS", "PRICE",
        "NUMEST", "MEANEST", "STDEV", "cons_p", "disp_p", "breadth", "staleness_days",
        "rev1_p", "rev3_p", "MEANREC", "NUMREC", "rec_chg3", "rec_breadth",
        "last_fpe", "last_ann", "sue_last_p", "last_suescore", "days_since_ann",
        "expected_ann", "expected_report_next_month", "eps_hat", "fgap_p", "fgap_disp",
        "target_fpe", "target_ann", "target_surprise_p", "target_error_vs_now_p",
        "target_future_revision_p", "target_reports_next_month",
    ]
    return state[keep].sort_values(["permno", "eom"]).reset_index(drop=True)


def analyst_history(state: pd.DataFrame) -> pd.DataFrame:
    """Analyst-error history, consensus-implied growth, next-quarter revision and overdue reports."""
    surprises = load_surprises().sort_values(["TICKER", "anndats"]).copy()
    surprises["surprise"] = surprises.actual - surprises.surpmean
    surprises["beat"] = (surprises.surprise > 0).astype(float)
    grouped = surprises.groupby("TICKER")
    surprises["hist_sue4"] = grouped.surprise.transform(lambda s: s.rolling(4, min_periods=2).mean())
    surprises["hist_sue8"] = grouped.surprise.transform(lambda s: s.rolling(8, min_periods=4).mean())
    surprises["sue_vol8"] = grouped.surprise.transform(lambda s: s.rolling(8, min_periods=4).std())
    surprises["beat_rate8"] = grouped.beat.transform(lambda s: s.rolling(8, min_periods=4).mean())
    history = surprises[["TICKER", "anndats", "hist_sue4", "hist_sue8", "sue_vol8", "beat_rate8"]]

    frame = state[["permno", "eom", "TICKER", "PRICE", "MEANEST", "FPEDATS", "expected_ann"]].copy()
    frame = _asof(frame, history.rename(columns={"anndats": "hist_ann"}), "hist_ann")
    for column in ("hist_sue4", "hist_sue8", "sue_vol8"):
        frame[f"{column}_p"] = frame[column] / frame.PRICE

    # Same quarter one year earlier, only if already announced by t.
    by_quarter = surprises.set_index(["TICKER", "fpe"])
    key = pd.MultiIndex.from_arrays([frame.TICKER, month_end(frame.FPEDATS - pd.DateOffset(months=12))])
    announced = pd.DatetimeIndex(by_quarter.anndats.reindex(key).to_numpy())
    known = announced <= pd.DatetimeIndex(frame.eom)
    last_year = np.where(known, by_quarter.actual.reindex(key).to_numpy(), np.nan)
    frame["exp_growth_p"] = (frame.MEANEST - last_year) / frame.PRICE

    # Overdue: the usual report date has passed and the target quarter is still open at t.
    overdue = (frame.eom - frame.expected_ann).dt.days
    frame["days_overdue"] = overdue.clip(lower=0).where(frame.expected_ann.notna())

    consensus = load_consensus()[["TICKER", "FPEDATS", "snap_month", "MEANEST"]]
    now = pd.DataFrame({
        "TICKER": consensus.TICKER, "next_fpe": month_end(consensus.FPEDATS).astype("datetime64[ns]"),
        "snap_month": consensus.snap_month, "next_now": consensus.MEANEST,
    }).drop_duplicates(["TICKER", "next_fpe", "snap_month"])
    past = now.assign(snap_month=now.snap_month + 3).rename(columns={"next_now": "next_lag3"})
    frame["next_fpe"] = month_end(frame.FPEDATS + pd.DateOffset(months=3)).astype("datetime64[ns]")
    frame["snap_month"] = frame.eom.dt.to_period("M")
    frame = frame.merge(now, on=["TICKER", "next_fpe", "snap_month"], how="left")
    frame = frame.merge(past, on=["TICKER", "next_fpe", "snap_month"], how="left")
    frame["rev3_next_p"] = (frame.next_now - frame.next_lag3) / frame.PRICE
    return frame[["permno", "eom", *HISTORY]]


# --------------------------------------------------------------------------
# Graph peer signals
# --------------------------------------------------------------------------

def to_days(values) -> np.ndarray:
    """Datetime-like values to float days since epoch (NaN for missing)."""
    dates = pd.to_datetime(pd.Series(values))
    return (dates - pd.Timestamp("1970-01-01")).dt.days.to_numpy(dtype=float)


def weighted_peer_mean(matrix: sparse.csr_matrix, values) -> np.ndarray:
    """Row-weighted peer mean renormalized over peers with observed values."""
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    mass = matrix @ observed.astype(float)
    total = matrix @ np.where(observed, values, 0.0)
    return np.divide(total, mass, out=np.full(len(values), np.nan), where=mass > 0)


def peer_surprise(matrix: sparse.csr_matrix, surprise, peer_quarter_end, peer_announced,
                  target_quarter_end, eom_days: float) -> dict[str, np.ndarray]:
    """Weighted mean of peers' latest analyst surprises announced within 92 days of t.

    ``recent`` uses every fresh peer surprise; ``aligned`` keeps peers whose reported
    quarter ends within 46 days of i's next unreported quarter (same calendar quarter).
    """
    coo = matrix.tocoo()
    rows, cols, weights = coo.row, coo.col, coo.data
    n = matrix.shape[0]
    base = (np.isfinite(surprise[cols]) & (peer_announced[cols] <= eom_days)
            & (peer_announced[cols] > eom_days - FRESH_PEER_DAYS) & np.isfinite(target_quarter_end[rows]))
    aligned = np.abs(peer_quarter_end[cols] - target_quarter_end[rows]) <= ALIGN_DAYS
    values = np.where(np.isfinite(surprise), surprise, 0.0)[cols]
    out = {}
    for name, mask in (("recent", base), ("aligned", base & aligned)):
        mass = np.bincount(rows, weights * mask, minlength=n)
        total = np.bincount(rows, weights * mask * values, minlength=n)
        out[name] = np.divide(total, mass, out=np.full(n, np.nan), where=mass > 0)
    return out


def graph_peer_features(state: pd.DataFrame, graph_root: Path = GRAPH_ROOT) -> pd.DataFrame:
    """Peer surprises, revisions and beat rates from each month's frozen economic graph."""
    columns = ["permno", "eom", "last_ann", "last_fpe", "sue_last_p", "target_fpe", "rev3_p", "beat_rate8"]
    by_month = dict(tuple(state[columns].groupby("eom")))
    frames = []
    for directory in sorted(p for p in graph_root.iterdir() if p.is_dir() and MONTH_DIRECTORY.fullmatch(p.name)):
        eom = pd.Timestamp(directory.name).to_period("M").to_timestamp("M")
        if eom not in by_month:
            continue
        nodes = pd.read_parquet(directory / "nodes.parquet", columns=["permno"])
        matrix = sparse.load_npz(directory / "adjacency.npz").tocsr()
        frame = nodes.merge(by_month[eom].drop(columns="eom"), on="permno", how="left", validate="one_to_one")
        eom_days = float(to_days([eom])[0])
        announced = to_days(frame.last_ann)
        surprise = frame.sue_last_p.to_numpy(dtype=float).copy()
        surprise[~((announced <= eom_days) & (announced > eom_days - FRESH_PEER_DAYS))] = np.nan
        shocks = peer_surprise(matrix, winsorize(surprise), to_days(frame.last_fpe), announced,
                               to_days(frame.target_fpe), eom_days)
        frames.append(pd.DataFrame({
            "permno": frame.permno.to_numpy(), "eom": eom,
            "peer_surprise_recent": shocks["recent"], "peer_surprise_aligned": shocks["aligned"],
            "peer_revision3": weighted_peer_mean(matrix, winsorize(frame.rev3_p.to_numpy(dtype=float))),
            "peer_beat_rate8": weighted_peer_mean(matrix, frame.beat_rate8.to_numpy(dtype=float)),
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Walk-forward surprise model
# --------------------------------------------------------------------------

def build_model_panel() -> tuple[pd.DataFrame, list[str]]:
    characteristics = list(pd.read_csv(FACTOR_LIST_FILE)["variable"])
    chars = pq.read_table(CHARACTERISTICS_FILE, columns=["permno", "eom", "size_grp", *characteristics]).to_pandas()
    chars["eom"] = month_end(chars.eom).astype("datetime64[ns]")
    chars = chars.drop_duplicates(["permno", "eom"])
    state = build_expectation_state(chars[["permno", "eom"]])
    state["eom"] = state.eom.astype("datetime64[ns]")
    state = state.merge(analyst_history(state), on=["permno", "eom"], how="left")
    peers = graph_peer_features(state)
    peers["eom"] = peers.eom.astype("datetime64[ns]")
    panel = (state.merge(peers, on=["permno", "eom"], how="left")
                  .merge(chars, on=["permno", "eom"], how="left"))
    features = [*STREET, *HISTORY, *GRAPH, *characteristics, "fgap_p", "fgap_disp"]
    return panel.sort_values(["eom", "permno"]).reset_index(drop=True), features


def fit_booster(x_train: np.ndarray, y_train: np.ndarray):
    import xgboost as xgb

    matrix = xgb.QuantileDMatrix(x_train, y_train)
    try:
        booster = xgb.train({**MODEL_PARAMS, "device": "cuda"}, matrix, MODEL_ROUNDS)
    except xgb.core.XGBoostError:
        booster = xgb.train({**MODEL_PARAMS, "device": "cpu", "nthread": 16}, matrix, MODEL_ROUNDS)
    booster.set_param({"device": "cpu"})  # predictions run on host (numpy) arrays
    return booster


def monthly_rank_ic(months: pd.Series, actual: np.ndarray, predicted: np.ndarray) -> float:
    frame = pd.DataFrame({"eom": months.to_numpy(), "actual": actual, "predicted": predicted}).dropna()
    by_month = frame.groupby("eom").apply(lambda g: g.actual.corr(g.predicted, method="spearman"),
                                          include_groups=False)
    return float(by_month.mean()) if len(by_month) else np.nan


def walk_forward_surprise(panel: pd.DataFrame, features: list[str],
                          validation_years: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit once per test year on training rows only; score validation and test rows."""
    x = (panel.groupby("eom")[features].rank(pct=True) - 0.5).to_numpy(dtype=np.float32)
    target = panel.target_surprise_p.to_numpy(dtype=float)
    predictions, folds = [], []
    for year in TEST_YEARS:
        masks = split_masks(panel.eom, 1, year, label_dates=panel.target_ann, validation_years=validation_years)
        train = masks["train"] & np.isfinite(target)
        if not train.any() or not masks["test"].any():
            continue
        # Rank the label within month among this fold's training rows only, so labels announced
        # after the training end never shape the training target.
        y_train = (panel.loc[train].groupby("eom").target_surprise_p.rank(pct=True) - 0.5).to_numpy(dtype=np.float32)
        booster = fit_booster(x[train], y_train)
        fold = {"test_year": year, "train_rows": int(train.sum()),
                "validation_rows": int(masks["validation"].sum()), "test_rows": int(masks["test"].sum())}
        for split in ("validation", "test"):
            rows = masks[split]
            fold[f"{split}_surprise_rank_ic"] = (
                monthly_rank_ic(panel.eom[rows], target[rows], booster.inplace_predict(x[rows])) if rows.any() else np.nan)
        folds.append(fold)
        test_rows = masks["test"]
        predictions.append(panel.loc[test_rows, ["permno", "eom"]].assign(
            test_year=year, pred_surprise=booster.inplace_predict(x[test_rows])))
        print(f"test year {year}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in fold.items() if k != "test_year"),
            flush=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-years", type=int, choices=(0, 2), default=2)
    parser.add_argument("--save-panel", action="store_true", help="also write the full feature panel")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel, features = build_model_panel()
    print(f"panel {panel.shape}, {len(features)} features", flush=True)
    if args.save_panel:
        panel.to_parquet(OUTPUT / "panel.parquet", index=False)
    predictions, folds = walk_forward_surprise(panel, features, args.validation_years)
    suffix = "" if args.validation_years == 2 else "_no_validation"
    predictions.to_parquet(OUTPUT / f"surprise_predictions{suffix}.parquet", index=False)
    folds.to_csv(OUTPUT / f"surprise_folds{suffix}.csv", index=False)
    print(folds.round(4).to_string(index=False))
    print(f"mean test surprise rank IC: {folds.test_surprise_rank_ic.mean():.4f}")


if __name__ == "__main__":
    main()
