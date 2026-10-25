"""Point-in-time monthly stock tokens and graph peers for attention training."""

import json
import pathlib

import numpy as np
import pandas as pd
from scipy import sparse

from . import earnings
from .. import ibes_state as ibes

FEATURES_FILE = "features.npy"
PEERS_FILE = "peers.npz"
ROWS_FILE = "rows.parquet"
TASKS = ("return", "eps", "risk")
PEERS = 15
CHARACTERISTICS = [
    "me",
    "ret_3_1",
    "ret_12_1",
    "beta_60m",
    "ivol_capm_21d",
    "be_me",
    "sale_me",
    "gp_at",
    "debt_me",
    "sale_gr1",
    "niq_su",
    "saleq_su",
]
# Formation months from here on have no realized next month in the panel.
_FIRST_UNUSED_MONTH = pd.Timestamp("2026-08-01")
_ROW_COLUMNS = [
    "permno",
    "eom",
    "target_month",
    "anndats",
    *(f"{task}_label" for task in TASKS),
]


def build(
    output_dir: pathlib.Path,
    panel_path: pathlib.Path,
    snapshot_dir: pathlib.Path,
) -> pathlib.Path:
    """Writes formation-time tokens, labels and peer lists for graph nodes.

    Tokens are monthly ranks plus observed flags, return lags, past EPS
    surprises (announced strictly before month-end). Pretrained forecasts are
    excluded because their training cutoff predates neither all folds nor all
    training rows.
    Labels are next-month return rank, next EPS surprise and absolute return
    rank, all in [-1, 1]. Row zero of the features file is padding, so a
    peer index i points at feature row i.

    Args:
        output_dir: Destination; reused when its rows file already exists.
        panel_path: Characteristics panel parquet.
        snapshot_dir: Monthly graph snapshots from `builder`.

    Returns:
        The output directory.
    """
    if (output_dir / ROWS_FILE).exists():
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "permno",
        "eom",
        "ret_exc",
        "ret_exc_lead1m",
        "gics",
        "size_grp",
        *CHARACTERISTICS,
    ]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel = panel.sort_values(["eom", "permno"]).reset_index(drop=True)
    panel["eom"] = pd.to_datetime(panel.eom)
    if panel.duplicated(["permno", "eom"]).any():
        raise ValueError("Duplicate panel keys")
    events = earnings.dated_eps_events(ibes.load_surprises(), ibes.load_link())
    raw = pd.concat(
        [
            panel[CHARACTERISTICS],
            _calendar_lags(panel),
            _past_news(panel, events),
        ],
        axis=1,
    )
    tokens = _rank_features(raw, panel.eom)
    # Keep surprise sign, magnitude and recency as well as their ranks.
    extras = np.column_stack(
        [
            np.tanh(raw.eps_last.fillna(0).to_numpy() / 4),
            np.exp(-raw.eps_age.fillna(10000).to_numpy() / 90),
        ]
    )
    tokens = np.concatenate([tokens, extras.astype(np.float32)], axis=1)
    future = earnings.attach_next_eps(panel[["permno", "eom"]], events)
    panel = panel.merge(
        future[["permno", "eom", "anndats", "next_suescore"]],
        on=["permno", "eom"],
        how="left",
        validate="one_to_one",
    )
    panel["target_month"] = panel.eom + pd.offsets.MonthEnd(1)
    by_month = panel.groupby("eom")
    panel["return_label"] = by_month.ret_exc_lead1m.rank(pct=True) * 2 - 1
    panel["eps_label"] = np.tanh(panel.next_suescore / 4)
    panel["risk_label"] = (
        panel.ret_exc_lead1m.abs().groupby(panel.eom).rank(pct=True) * 2 - 1
    )
    lookup = panel.set_index(["permno", "eom"]).index
    blocks, node_tokens, peer_index, peer_weight = [], [], [], []
    offset = 0
    for month in pd.DatetimeIndex(panel.eom.unique()).sort_values():
        if month >= _FIRST_UNUSED_MONTH:
            continue
        directory = snapshot_dir / f"{month:%Y-%m}"
        nodes = pd.read_parquet(directory / "nodes.parquet", columns=["permno"])
        nodes = nodes.reset_index(drop=True)
        if nodes.permno.duplicated().any():
            raise ValueError(f"Duplicate graph nodes: {directory}")
        metadata = json.loads((directory / "metadata.json").read_text())
        if pd.Timestamp(metadata["characteristic_cutoff"]) > month:
            raise ValueError(f"Future graph snapshot: {directory}")
        positions = lookup.get_indexer(
            pd.MultiIndex.from_arrays([nodes.permno, [month] * len(nodes)])
        )
        if (positions < 0).any():
            raise ValueError(f"Graph nodes missing from the panel: {directory}")
        blocks.append(panel.iloc[positions])
        node_tokens.append(tokens[positions])
        index, weight = _peer_arrays(
            sparse.load_npz(directory / "adjacency.npz"), offset
        )
        peer_index.append(index)
        peer_weight.append(weight)
        offset += len(positions)
    rows = pd.concat(blocks, ignore_index=True)
    padding = np.zeros((1, tokens.shape[1]), np.float32)
    np.save(output_dir / FEATURES_FILE, np.vstack([padding, *node_tokens]))
    np.savez(
        output_dir / PEERS_FILE,
        indices=np.vstack(peer_index),
        weights=np.vstack(peer_weight),
    )
    rows[_ROW_COLUMNS].to_parquet(output_dir / ROWS_FILE, index=False)
    print(
        f"Graph tokens: {len(rows):,} stock-months, {tokens.shape[1]} features",
        flush=True,
    )
    return output_dir


def _calendar_lags(panel: pd.DataFrame, count: int = 12) -> pd.DataFrame:
    """Excess return lags by calendar month; a missing month stays missing."""
    values = panel.set_index(["permno", "eom"]).ret_exc
    result = pd.DataFrame(index=panel.index)
    for lag in range(count):
        dates = panel.eom if lag == 0 else panel.eom - pd.offsets.MonthEnd(lag)
        keys = pd.MultiIndex.from_arrays([panel.permno, dates])
        result[f"return_lag{lag}"] = values.reindex(keys).to_numpy()
    return result


def _past_news(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Last three EPS surprises reported strictly before each month-end."""
    right = events.sort_values(["permno", "anndats"]).copy()
    for lag in (1, 2):
        right[f"eps_lag{lag}"] = right.groupby("permno").next_suescore.shift(lag)
    right = right.rename(
        columns={"next_suescore": "eps_last", "anndats": "past_report"}
    )
    right["past_report"] = right.past_report.astype("datetime64[ns]")
    left = panel[["permno", "eom"]].copy()
    left["row"] = np.arange(len(left))
    left["eom"] = left.eom.astype("datetime64[ns]")
    joined = pd.merge_asof(
        left.sort_values("eom"),
        right.sort_values("past_report"),
        by="permno",
        left_on="eom",
        right_on="past_report",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values("row")
    joined["eps_age"] = (joined.eom - joined.past_report).dt.days.clip(upper=730)
    return joined[["eps_last", "eps_lag1", "eps_lag2", "eps_age"]].set_axis(panel.index)


def _rank_features(values: pd.DataFrame, months: pd.Series) -> np.ndarray:
    """Monthly ranks on [-1, 1] (missing as 0) plus observed-value flags."""
    values = values.replace([np.inf, -np.inf], np.nan)
    ranked = values.groupby(months).rank(pct=True).mul(2).sub(1)
    observed = values.notna().to_numpy(np.float32)
    return np.concatenate([ranked.fillna(0).to_numpy(np.float32), observed], axis=1)


def _peer_arrays(
    matrix: sparse.csr_matrix, offset: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top graph peers per node as global feature rows; zero is padding."""
    matrix = matrix.tocsr()
    indices = np.zeros((matrix.shape[0], PEERS), dtype=np.int64)
    weights = np.zeros_like(indices, dtype=np.float32)
    if matrix.shape[0] != matrix.shape[1] or matrix.diagonal().any():
        raise ValueError("Expected a square peer graph without self edges")
    for i in range(matrix.shape[0]):
        start, stop = matrix.indptr[i : i + 2]
        columns = matrix.indices[start:stop]
        values = matrix.data[start:stop]
        valid = np.isfinite(values) & (values > 0)
        columns, values = columns[valid], values[valid]
        take = np.argsort(-values, kind="stable")[:PEERS]
        columns, values = columns[take], values[take]
        if len(columns):
            indices[i, : len(columns)] = columns + offset + 1
            weights[i, : len(columns)] = values / values.sum()
    return indices, weights
