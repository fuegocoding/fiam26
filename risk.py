"""Adjusts portfolio weights using historical risk and signal strength."""

import math

import numpy as np
import pandas as pd

PANEL_COLUMNS = ["permno", "eom", "gics", "size_grp", "ret_exc", "rmax1_21d"]

# Peer EVT: a GPD fitted to the pooled monthly returns of a stock's industry
# x size peers over three years gives its chance of a squeeze month. Shorts in
# the riskiest PEER_TOP_SHARE of stocks keep PEER_SCALE of their weight; the
# freed weight goes to the same sector's other shorts, so the book stays
# sector neutral, but no receiver exceeds MAX_SHORT_MULTIPLE x the average
# short; any excess is held as cash rather than concentrated.
PEER_WINDOW_MONTHS = 36
PEER_MIN_OBSERVATIONS = 300
SQUEEZE_MOVE = 0.5  # A month up 50% or more counts as a squeeze.
PEER_TOP_SHARE = 0.05
PEER_SCALE = 0.5
MAX_SHORT_MULTIPLE = 1.5
# Run-up cap, the monthly stand-in for the live rule. A short that rallied
# last month is cut so its exposure stays within RUNUP_CAP x its target. Live,
# the cap runs daily: at each close, a short above 1.5x its target is bought
# back to 1.5x while the model still ranks it a short, which keeps margin and
# recall pressure contained instead of closing the position outright.
RUNUP_CAP = 1.5
# Expected shortfall: the book is replayed on its holdings' past monthly
# returns and its ES (mean of the worst ES_TAIL months) compared with the
# median ES the same book showed at the previous ES_NORM_MONTHS month-ends.
# When tail risk rises above ES_TOLERANCE x that norm, the whole book is
# scaled down in proportion, never below ES_FLOOR and never up.
ES_WINDOW_MONTHS = 36
ES_NORM_MONTHS = 24
ES_MIN_MONTHS = 24
ES_TAIL = 0.05
ES_TOLERANCE = 1.0
ES_FLOOR = 0.5
# Edge throttle: exposure follows the edge the model has shown lately. The
# realized return spread between the top and bottom EDGE_TAIL of scores over
# the last EDGE_MONTHS gives a t-statistic; gross is scaled by 0.5 + 0.25 x t
# within [EDGE_FLOOR, 1], in EDGE_STEP steps so positions are resized rarely.
EDGE_MONTHS = 12
EDGE_MIN_MONTHS = 6
EDGE_TAIL = 0.05
EDGE_FLOOR = 0.5
EDGE_STEP = 0.25
MIN_PRICE = 5.0
# Own EVT: a GEV fitted to a short's monthly maxima of daily returns gives its
# 1-in-100-month one-day jump, reported with every short.
GEV_WINDOW_MONTHS = 60
GEV_MIN_MONTHS = 24
JUMP_CLIP = 3.0


def protect(
    weights: dict[int, float],
    month: pd.Period,
    panel: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[dict[int, float], pd.DataFrame, dict]:
    """Applies the risk rules to one month's target book.

    Only panel rows dated at or before formation, the month before `month`,
    and scores of earlier return months are used.

    Args:
        weights: Target weight by permno; shorts are negative.
        month: Return month of the book.
        panel: Monthly panel with PANEL_COLUMNS and `eom` as a monthly period.
        scores: Final scores with return_month (monthly period), permno,
            final_score and prc.

    Returns:
        The adjusted weights, squeeze statistics by permno and a summary of
        what each rule did.
    """
    book = pd.Series(weights, dtype=float)
    if book.empty:
        return weights, pd.DataFrame(), {}
    formation = month - 1
    history = panel.loc[panel.eom <= formation]
    now = history.loc[history.eom == formation].drop_duplicates("permno")
    now = now.set_index("permno")
    shorts = book.index[book < 0]

    probability = _peer_squeeze_probability(history, now, formation)
    flagged = probability.rank(pct=True).gt(1 - PEER_TOP_SHARE)
    flagged = flagged.reindex(shorts, fill_value=False)
    sector = now.gics.astype("string").str[:2].reindex(shorts).fillna("na")
    adjusted = _shrink_flagged(book, flagged, sector)

    rally = now.ret_exc.reindex(shorts).fillna(0.0).clip(lower=0.0)
    capped = (1 + rally) > RUNUP_CAP
    adjusted[shorts] *= np.minimum(1.0, RUNUP_CAP / (1 + rally))

    shortfall, norm = _expected_shortfall(adjusted, history, formation)
    scale = 1.0
    if np.isfinite(norm) and shortfall > 0:
        scale = float(np.clip(ES_TOLERANCE * norm / shortfall, ES_FLOOR, 1.0))
    edge_t, edge_scale = _edge_throttle(scores, history, month)
    adjusted *= scale * edge_scale

    table = pd.DataFrame(
        {
            "peer_squeeze_probability": probability.reindex(book.index),
            "jump_1_in_100_months": _jump_level(history, shorts, formation).reindex(
                book.index
            ),
            "risk_multiplier": adjusted / book,
        }
    )
    summary = {
        "peer_flagged_shorts": int(flagged.sum()),
        "runup_capped_shorts": int(capped.sum()),
        "expected_shortfall": shortfall,
        "shortfall_norm": norm,
        "risk_scale": scale,
        "edge_t": edge_t,
        "edge_scale": edge_scale,
        "short_weight_in_flagged": float(-adjusted[flagged[flagged].index].sum()),
    }
    return adjusted.to_dict(), table, summary


def _edge_throttle(
    scores: pd.DataFrame, history: pd.DataFrame, month: pd.Period
) -> tuple[float, float]:
    """T-statistic of the recent top-minus-bottom spread and the gross scale."""
    past = scores.loc[
        (scores.return_month < month)
        & (scores.return_month >= month - EDGE_MONTHS)
        & (scores.prc >= MIN_PRICE)
    ]
    realized = history[["permno", "eom", "ret_exc"]].rename(
        columns={"eom": "return_month"}
    )
    past = past.merge(realized, on=["return_month", "permno"]).dropna(
        subset=["final_score", "ret_exc"]
    )
    rank = past.groupby("return_month").final_score.rank(pct=True)
    top = past[rank > 1 - EDGE_TAIL].groupby("return_month").ret_exc.mean()
    bottom = past[rank <= EDGE_TAIL].groupby("return_month").ret_exc.mean()
    spread = (top - bottom).dropna()
    if len(spread) < EDGE_MIN_MONTHS or spread.std() == 0:
        return np.nan, 1.0
    t = float(spread.mean() / spread.std() * math.sqrt(len(spread)))
    raw = float(np.clip(0.5 + 0.25 * t, EDGE_FLOOR, 1.0))
    steps = round((raw - EDGE_FLOOR) / EDGE_STEP)
    return t, min(1.0, EDGE_FLOOR + steps * EDGE_STEP)


def _peer_squeeze_probability(
    history: pd.DataFrame, now: pd.DataFrame, formation: pd.Period
) -> pd.Series:
    """P(squeeze month) from the most specific peer group with enough data."""
    window = history.loc[history.eom > formation - PEER_WINDOW_MONTHS]
    pooled = _peer_keys(window).assign(ret=window.ret_exc.clip(-1.0, 5.0))
    current = _peer_keys(now)
    probability = pd.Series(np.nan, index=now.index)
    for keys in (["industry", "size"], ["industry"], ["sector"]):
        grouped = pooled.groupby(keys).ret
        stats = grouped.apply(lambda r: _squeeze_probability(r.to_numpy()))
        stats = stats[grouped.size() >= PEER_MIN_OBSERVATIONS]
        lookup = (
            pd.MultiIndex.from_frame(current[keys])
            if len(keys) > 1
            else pd.Index(current[keys[0]])
        )
        found = pd.Series(stats.reindex(lookup).to_numpy(), index=now.index)
        probability = probability.fillna(found)
    return probability


def _peer_keys(frame: pd.DataFrame) -> pd.DataFrame:
    gics = frame.gics.astype("string")
    return pd.DataFrame(
        {
            "industry": gics.str[:6].fillna("na").to_numpy(),
            "sector": gics.str[:2].fillna("na").to_numpy(),
            "size": frame.size_grp.astype("string").fillna("na").to_numpy(),
        },
        index=frame.index,
    )


def _squeeze_probability(returns: np.ndarray) -> float:
    """P(return > SQUEEZE_MOVE) from a GPD (PWM) above the 95th percentile."""
    returns = returns[np.isfinite(returns)]
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


def _shrink_flagged(
    book: pd.Series, flagged: pd.Series, sector: pd.Series
) -> pd.Series:
    """Scales flagged shorts down and hands their weight to sector peers."""
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


def _expected_shortfall(
    book: pd.Series, history: pd.DataFrame, formation: pd.Period
) -> tuple[float, float]:
    """Replayed ES of the book now, and its median at earlier month-ends."""
    span = ES_WINDOW_MONTHS + ES_NORM_MONTHS
    window = history.loc[
        (history.eom > formation - span) & history.permno.isin(book.index)
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


def _jump_level(
    history: pd.DataFrame, permnos: pd.Index, formation: pd.Period
) -> pd.Series:
    """GEV 1-in-100-month level of each stock's largest daily return."""
    window = history.loc[
        (history.eom > formation - GEV_WINDOW_MONTHS) & history.permno.isin(permnos)
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
        b2 = ((rank - 1) * (rank - 2) / ((count - 1) * (count - 2)) * values).sum(
            0
        ) / count
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
