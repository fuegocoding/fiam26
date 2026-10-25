"""Risk overlay: peer-squeeze shrink, run-up cap, expected-shortfall scale,
edge throttle (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 17; port of first-round
risk.py).

All inputs are dated at or before formation: panel history through the
formation month, and score spreads from earlier return months only. The
rules, constants and math are the first-round implementation.

1. Peer EVT squeeze: a GPD on pooled peer monthly returns gives each short's
   squeeze probability; the riskiest PEER_TOP_SHARE keep PEER_SCALE of their
   weight and the freed weight goes to same-sector shorts, capped at
   MAX_SHORT_MULTIPLE x the average short; excess stays cash.
2. Run-up cap: a short that rallied last month is scaled to RUNUP_CAP x
   its target exposure.
3. Expected shortfall: the book is replayed on its holdings' past monthly
   returns; when tail risk exceeds ES_TOLERANCE x its trailing norm the book
   is scaled down, never below ES_FLOOR.
4. Edge throttle: gross follows the t-stat of the recent top-minus-bottom
   score spread in EDGE_STEP steps within [EDGE_FLOOR, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["permno", "eom", "gics", "size_grp", "ret_exc", "rmax1_21d"]

PEER_WINDOW_MONTHS = 36
PEER_MIN_OBSERVATIONS = 300
SQUEEZE_MOVE = 0.5
PEER_TOP_SHARE = 0.05
PEER_SCALE = 0.5
MAX_SHORT_MULTIPLE = 1.5
RUNUP_CAP = 1.5
ES_WINDOW_MONTHS = 36
ES_NORM_MONTHS = 24
ES_MIN_MONTHS = 24
ES_TAIL = 0.05
ES_TOLERANCE = 1.0
ES_FLOOR = 0.5
EDGE_MONTHS = 12
EDGE_MIN_MONTHS = 6
EDGE_TAIL = 0.05
EDGE_FLOOR = 0.5
EDGE_STEP = 0.25
MIN_PRICE = 5.0
GEV_WINDOW_MONTHS = 60
GEV_MIN_MONTHS = 24
JUMP_CLIP = 3.0


@dataclass
class OverlayResult:
    weights: dict
    stats: pd.DataFrame
    summary: dict = field(default_factory=dict)


def protect(
    weights: dict[str, float],
    month: pd.Period,
    panel: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    max_abs_net: float | None = None,
) -> OverlayResult:
    book = pd.Series(weights, dtype=float)
    if book.empty:
        return OverlayResult(weights, pd.DataFrame(), {})
    formation = month - 1
    history = panel.loc[panel["eom"] <= formation]
    now = history.loc[history["eom"] == formation].drop_duplicates("permno").set_index("permno")
    shorts = book.index[book < 0]

    probability = _peer_squeeze_probability(history, now, formation)
    flagged = probability.rank(pct=True).gt(1 - PEER_TOP_SHARE)
    flagged = flagged.reindex(shorts, fill_value=False)
    sector = now["gics"].astype("string").str[:2].reindex(shorts).fillna("na")
    adjusted = _shrink_flagged(book, flagged, sector)

    rally = now["ret_exc"].reindex(shorts).fillna(0.0).clip(lower=0.0)
    capped = (1 + rally) > RUNUP_CAP
    adjusted[shorts] *= np.minimum(1.0, RUNUP_CAP / (1 + rally))

    shortfall, norm = _expected_shortfall(adjusted, history, formation)
    scale = 1.0
    if np.isfinite(norm) and shortfall > 0:
        scale = float(np.clip(ES_TOLERANCE * norm / shortfall, ES_FLOOR, 1.0))
    edge_t, edge_scale = _edge_throttle(scores, history, month)
    adjusted = adjusted * scale * edge_scale

    net_capped = False
    if max_abs_net is not None and max_abs_net >= 0.0:
        net_val = float(adjusted.sum())
        long_mask = adjusted > 0
        short_mask = adjusted < 0
        long_sum = float(adjusted[long_mask].sum())
        short_sum = float(-adjusted[short_mask].sum())
        if net_val > max_abs_net and long_sum > 0:
            target_long = short_sum + max_abs_net
            adjusted[long_mask] *= target_long / long_sum
            net_capped = True
        elif net_val < -max_abs_net and short_sum > 0:
            target_short = long_sum + max_abs_net
            adjusted[short_mask] *= target_short / short_sum
            net_capped = True

    table = pd.DataFrame(
        {
            "peer_squeeze_probability": probability.reindex(book.index),
            "jump_1_in_100_months": _jump_level(history, shorts, formation).reindex(book.index),
            "risk_multiplier": (adjusted / book).to_numpy(),
        },
        index=book.index,
    )
    summary = {
        "peer_flagged_shorts": int(flagged.sum()),
        "runup_capped_shorts": int(capped.sum()),
        "expected_shortfall": float(shortfall) if np.isfinite(shortfall) else None,
        "shortfall_norm": float(norm) if np.isfinite(norm) else None,
        "risk_scale": scale,
        "edge_t": float(edge_t) if np.isfinite(edge_t) else None,
        "edge_scale": edge_scale,
        "short_weight_in_flagged": float(-adjusted[flagged[flagged].index].sum()) if flagged.any() else 0.0,
        "post_overlay_net_capped": net_capped,
    }
    return OverlayResult(adjusted.to_dict(), table, summary)


def _edge_throttle(scores: pd.DataFrame, history: pd.DataFrame, month: pd.Period):
    past = scores.loc[
        (scores["return_month"] < month)
        & (scores["return_month"] >= month - EDGE_MONTHS)
        & (scores["prc"] >= MIN_PRICE)
    ]
    realized = history[["permno", "eom", "ret_exc"]].rename(columns={"eom": "return_month"})
    past = past.merge(realized, on=["return_month", "permno"]).dropna(subset=["final_score", "ret_exc"])
    rank = past.groupby("return_month")["final_score"].rank(pct=True)
    top = past[rank > 1 - EDGE_TAIL].groupby("return_month")["ret_exc"].mean()
    bottom = past[rank <= EDGE_TAIL].groupby("return_month")["ret_exc"].mean()
    spread = (top - bottom).dropna()
    if len(spread) < EDGE_MIN_MONTHS or spread.std() == 0:
        return np.nan, 1.0
    t = float(spread.mean() / spread.std() * math.sqrt(len(spread)))
    raw = float(np.clip(0.5 + 0.25 * t, EDGE_FLOOR, 1.0))
    steps = round((raw - EDGE_FLOOR) / EDGE_STEP)
    return t, min(1.0, EDGE_FLOOR + steps * EDGE_STEP)


def _peer_keys(frame: pd.DataFrame) -> pd.DataFrame:
    gics = frame["gics"].astype("string")
    return pd.DataFrame(
        {
            "industry": gics.str[:6].fillna("na").to_numpy(),
            "sector": gics.str[:2].fillna("na").to_numpy(),
            "size": frame["size_grp"].astype("string").fillna("na").to_numpy(),
        },
        index=frame.index,
    )


def _squeeze_probability(returns: np.ndarray) -> float:
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return float("nan")
    threshold = np.quantile(returns, 0.95)
    excess = np.sort(returns[returns > threshold] - threshold)
    observed = float((returns > SQUEEZE_MOVE).mean())
    if len(excess) < 10 or SQUEEZE_MOVE <= threshold:
        return observed
    plotting = (np.arange(1, len(excess) + 1) - 0.35) / len(excess)
    a0, a1 = excess.mean(), ((1 - plotting) * excess).mean()
    if a0 - 2 * a1 <= 0:
        return observed
    shape = float(np.clip(2 - a0 / (a0 - 2 * a1), -0.5, 0.9))
    scale = 2 * a0 * a1 / (a0 - 2 * a1)
    base = 1 + shape * (SQUEEZE_MOVE - threshold) / scale
    if base <= 0:
        return 0.0
    return len(excess) / len(returns) * base ** (-1 / shape)


def _peer_squeeze_probability(history: pd.DataFrame, now: pd.DataFrame, formation: pd.Period):
    window = history.loc[history["eom"] > formation - PEER_WINDOW_MONTHS]
    pooled = _peer_keys(window).assign(ret=window["ret_exc"].clip(-1.0, 5.0))
    current = _peer_keys(now)
    probability = pd.Series(np.nan, index=now.index)
    for keys in (["industry", "size"], ["industry"], ["sector"]):
        grouped = pooled.groupby(keys)["ret"]
        stats = grouped.apply(lambda r: _squeeze_probability(r.to_numpy()))
        stats = stats[grouped.size() >= PEER_MIN_OBSERVATIONS]
        lookup = (
            pd.MultiIndex.from_frame(current[keys]) if len(keys) > 1 else pd.Index(current[keys[0]])
        )
        found = pd.Series(stats.reindex(lookup).to_numpy(), index=now.index)
        probability = probability.fillna(found)
    return probability


def _shrink_flagged(book: pd.Series, flagged: pd.Series, sector: pd.Series) -> pd.Series:
    adjusted = book.copy()
    shorts = book[book < 0]
    if not flagged.any():
        return adjusted
    limit = MAX_SHORT_MULTIPLE * shorts.mean()
    new = shorts.where(~flagged, shorts * PEER_SCALE)
    for group in sector[flagged].unique():
        freed = (shorts - new)[flagged & (sector == group)].sum()
        receivers = ~flagged & (sector == group)
        if not receivers.any():
            receivers = ~flagged
        if receivers.any():
            share = new[receivers] / new[receivers].sum()
            new[receivers] = (new[receivers] + freed * share).clip(lower=limit)
    adjusted[new.index] = new
    return adjusted


def _expected_shortfall(book: pd.Series, history: pd.DataFrame, formation: pd.Period):
    span = ES_WINDOW_MONTHS + ES_NORM_MONTHS
    window = history.loc[
        (history["eom"] > formation - span) & history["permno"].isin(book.index)
    ].drop_duplicates(["eom", "permno"])
    returns = window.pivot(index="eom", columns="permno", values="ret_exc")
    returns = returns.reindex(columns=book.index).fillna(0.0).clip(-1.0, 3.0)
    pnl = returns.to_numpy() @ book.to_numpy()

    def shortfall(series: np.ndarray) -> float:
        if len(series) < ES_MIN_MONTHS:
            return np.nan
        losses = np.sort(-series)[::-1]
        return float(losses[: max(1, math.ceil(ES_TAIL * len(losses)))].mean())

    current = shortfall(pnl[-ES_WINDOW_MONTHS:])
    earlier = [
        shortfall(pnl[max(0, len(pnl) - lag - ES_WINDOW_MONTHS) : len(pnl) - lag])
        for lag in range(1, ES_NORM_MONTHS + 1)
    ]
    earlier = [value for value in earlier if np.isfinite(value)]
    norm = float(np.median(earlier)) if len(earlier) >= 6 else np.nan
    return current, norm


def _jump_level(history: pd.DataFrame, permnos: pd.Index, formation: pd.Period):
    window = history.loc[
        (history["eom"] > formation - GEV_WINDOW_MONTHS) & history["permno"].isin(permnos)
    ].drop_duplicates(["eom", "permno"])
    maxima = window.pivot(index="eom", columns="permno", values="rmax1_21d")
    if maxima.empty:
        return pd.Series(dtype=float)
    block = maxima.clip(upper=JUMP_CLIP).to_numpy(float)
    ordered = np.sort(block, axis=0)
    count = np.isfinite(block).sum(axis=0).astype(float)
    rank = np.arange(1, len(block) + 1, dtype=float)[:, None]
    values = np.where(rank <= count, ordered, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = values.sum(0) / count
        b1 = ((rank - 1) / (count - 1) * values).sum(0) / count
        b2 = ((rank - 1) * (rank - 2) / ((count - 1) * (count - 2)) * values).sum(0) / count
        l2 = 2 * b1 - b0
        skew = np.clip((6 * b2 - 6 * b1 + b0) / l2, -0.5, 0.9)
        c = 2 / (3 + skew) - math.log(2) / math.log(3)
        k = np.clip(7.8590 * c + 2.9554 * c * c, -0.9, 0.9)
        k = np.where(np.abs(k) < 1e-6, 1e-6, k)
        gamma = np.exp([math.lgamma(1 + v) for v in k])
        alpha = l2 * k / ((1 - 2.0 ** (-k)) * gamma)
        location = b0 + alpha * (gamma - 1) / k
        level = location + alpha * (1 - (-math.log(0.99)) ** k) / k
    level = np.where((count >= GEV_MIN_MONTHS) & (l2 > 0), level, np.nan)
    return pd.Series(level, index=maxima.columns)
