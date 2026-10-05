"""Builds a sector- and beta-neutral long/short book from monthly scores."""

import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import risk
import timing
from config import NAV

# Every dollar limit derives from NAV (config.py) or a stock's ADV.
GROSS = 1.00  # Long plus short notional over NAV.
N_PER_SIDE = 100
MIN_PRICE = 5.00  # Share price, in dollars, for longs and shorts.
EXECUTION_DAYS = 5
LONG_ADV_PARTICIPATION = 0.05
SHORT_ADV_PARTICIPATION = 0.02
REQUIRE_VERIFIED_LABELS = True
# Names per GICS sector follow the sector's share of eligible stocks, with as
# many shorts as longs, so the book is dollar neutral inside every sector.
SECTOR_NEUTRAL = True
MIN_SECTOR_NAMES = 4
# A holding stays while it still ranks within (1 + BUFFER) x its quota.
BUFFER = 0.5
# Restore the committed NAV switch for this timing-only comparison.
# It was selected on test-period research results, a disclosed selection bias.
ENABLE_MICRO_SLEEVE_SWITCH = True
MICRO_SLEEVES_MAX_NAV = 50_000_000.0
# A stock with this share of its shares outstanding already sold short cannot
# be borrowed at the size or fee the cost model assumes, so it is never
# shorted (FINRA short interest, see borrow.py).  Unknown short interest passes.
SHORT_INTEREST_CAP = 0.15
# Long and short notionals are scaled so their 60-month betas offset; each
# side stays within BETA_TILT of half the gross, keeping net exposure small.
BETA_NEUTRAL = True
BETA_TILT = 0.25
BETA_BOUNDS = (0.2, 3.0)

_UNVERIFIED_LABEL = "no_verified_label_for_observation_period"
_ATTRIBUTES = [
    "final_score",
    "ticker",
    "company_name",
    "ticker_name_reference_date",
    "ticker_name_source",
    "ticker_name_status",
    "prc",
    "size_grp",
    "dolvol_126d",
    "sector",
    "beta_60m",
]


def dollar_limits(nav: float = NAV) -> dict[str, float]:
    """Book-level and per-name dollar sizes implied by a NAV."""
    side = nav * GROSS / 2
    return {
        "nav": float(nav),
        "gross_notional_dollars": float(nav * GROSS),
        "side_notional_dollars": float(side),
        "position_dollars": float(side / N_PER_SIDE),
    }


def scored_for_nav(predictions: pd.DataFrame, nav: float) -> pd.DataFrame:
    """Copy of the predictions carrying the score set that fits the NAV.

    Above the committed $50M threshold, use the score without the two
    micro-cap sleeves. This fixed switch retains historical selection bias.
    """
    frame = predictions.copy()
    if (
        ENABLE_MICRO_SLEEVE_SWITCH
        and nav > MICRO_SLEEVES_MAX_NAV
        and "final_score_liquid" in frame
    ):
        frame["final_score"] = frame.final_score_liquid
        frame["available_count"] = frame.available_count_liquid
    return frame


def optimize(
    predictions: pd.DataFrame,
    panel_path: pathlib.Path,
    nav: float = NAV,
    short_interest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds each month's book from the final score, then applies risk rules.

    A holding absent from a month's predictions has already exited at its
    last price, so only names present that month can be kept.

    Args:
        predictions: One row per stock-month with return_month, permno,
            final_score, price, liquidity, sector, beta and labels.
        panel_path: Characteristics panel holding the risk rules' history.
        nav: Net asset value that sets every dollar size.
        short_interest: Optional borrow.build table (permno, return_month,
            short_pct); stocks at or above SHORT_INTEREST_CAP are not shorted.

    Returns:
        Target positions per month, and per-month exposures and coverage.
    """
    timing.validate_predictions(predictions)
    frame = scored_for_nav(predictions, nav)
    frame["return_month"] = pd.to_datetime(frame.return_month).dt.to_period("M")
    if short_interest is not None:
        crowded = short_interest[["permno", "return_month", "short_pct"]].copy()
        crowded["return_month"] = pd.to_datetime(crowded.return_month).dt.to_period("M")
        frame = frame.merge(crowded, on=["permno", "return_month"], how="left")
    else:
        frame["short_pct"] = np.nan
    panel = pd.read_parquet(panel_path, columns=risk.PANEL_COLUMNS)
    panel["eom"] = pd.to_datetime(panel.eom).dt.to_period("M")
    scores = frame[["return_month", "permno", "final_score", "prc"]]
    closed = set()
    if "ret_exc_lead1m" in pq.ParquetFile(panel_path).schema.names:
        outcomes = pd.read_parquet(
            panel_path, columns=["permno", "eom", "ret_exc_lead1m"]
        )
        outcomes = outcomes.loc[outcomes.ret_exc_lead1m.isna()]
        settled = pd.to_datetime(outcomes.eom).dt.to_period("M") + 1
        closed = set(zip(settled, outcomes.permno, strict=True))
    previous: dict[int, float] = {}
    positions, diagnostics = [], []
    for month, current in frame.groupby("return_month", sort=True):
        # Only settled prior-month closures can clear the retention buffer.
        previous = {p: w for p, w in previous.items() if (month - 1, p) not in closed}
        prev_long = {p for p, w in previous.items() if w > 0}
        prev_short = {p for p, w in previous.items() if w < 0}
        weights, coverage = _book(current, nav, prev_long, prev_short)
        weights, squeeze, rules = risk.protect(weights, month, panel, scores)
        attributes = current.set_index("permno")[_ATTRIBUTES]
        for permno, weight in weights.items():
            row = attributes.loc[permno]
            positions.append(
                {
                    "return_month": month,
                    "permno": permno,
                    "ticker": row.ticker,
                    "company_name": row.company_name,
                    "ticker_name_reference_date": row.ticker_name_reference_date,
                    "ticker_name_source": row.ticker_name_source,
                    "ticker_name_status": row.ticker_name_status,
                    "prc": float(row.prc),
                    "size_grp": row.size_grp,
                    "sector": row.sector,
                    "beta_60m": float(row.beta_60m),
                    "dolvol_126d": float(row.dolvol_126d),
                    "final_score": float(row.final_score),
                    "target_weight": weight,
                    "target_dollars": weight * nav,
                    "side": "long" if weight > 0 else "short",
                    **squeeze.loc[permno].to_dict(),
                }
            )
        book = pd.Series(weights, dtype=float)
        sectors = attributes.sector.reindex(book.index)
        names = set(weights) | set(previous)
        traded = sum(abs(weights.get(p, 0.0) - previous.get(p, 0.0)) for p in names)
        diagnostics.append(
            {
                "return_month": month,
                "long_notional": float(book[book > 0].sum()),
                "short_notional": float(-book[book < 0].sum()),
                "net_exposure": float(book.sum()),
                "sector_net_abs": float(book.groupby(sectors).sum().abs().sum()),
                "beta_exposure": float(
                    (book * attributes.beta_60m.reindex(book.index).fillna(1.0)).sum()
                ),
                "turnover_nav": float(traded),
                **coverage,
                **rules,
            }
        )
        previous = weights
    return pd.DataFrame(positions), pd.DataFrame(diagnostics)


def write(
    portfolio: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: pathlib.Path,
    nav: float = NAV,
) -> dict:
    """Writes the book, its diagnostics and the construction settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stored = portfolio.copy()
    stored["return_month"] = stored.return_month.dt.to_timestamp("M")
    stored.to_parquet(output_dir / "portfolio.parquet", index=False, compression="zstd")
    stored = diagnostics.copy()
    stored["return_month"] = stored.return_month.astype(str)
    stored.to_csv(output_dir / "optimizer_diagnostics.csv", index=False)
    config = {
        **dollar_limits(nav),
        "names_per_side": N_PER_SIDE,
        "min_price_dollars": MIN_PRICE,
        "execution_days": EXECUTION_DAYS,
        "adv_participation": {
            "long": LONG_ADV_PARTICIPATION,
            "short": SHORT_ADV_PARTICIPATION,
        },
        "require_verified_point_in_time_labels": REQUIRE_VERIFIED_LABELS,
        "sector_neutral": SECTOR_NEUTRAL,
        "rank_buffer": BUFFER,
        "short_interest_cap": SHORT_INTEREST_CAP,
        "micro_sleeves_max_nav": MICRO_SLEEVES_MAX_NAV,
        "micro_sleeve_switch_enabled": ENABLE_MICRO_SLEEVE_SWITCH,
        "parameter_source": "committed_fixed_recipe_with_timing_and_liquidation_fixes",
        "beta_neutral": BETA_NEUTRAL,
        "risk": {
            "peer_top_share": risk.PEER_TOP_SHARE,
            "peer_scale": risk.PEER_SCALE,
            "squeeze_move": risk.SQUEEZE_MOVE,
            "max_short_multiple": risk.MAX_SHORT_MULTIPLE,
            "runup_cap": risk.RUNUP_CAP,
            "expected_shortfall_tail": risk.ES_TAIL,
            "expected_shortfall_norm_months": risk.ES_NORM_MONTHS,
            "expected_shortfall_tolerance": risk.ES_TOLERANCE,
            "expected_shortfall_floor": risk.ES_FLOOR,
            "edge_months": risk.EDGE_MONTHS,
            "edge_tail": risk.EDGE_TAIL,
            "edge_floor": risk.EDGE_FLOOR,
            "edge_step": risk.EDGE_STEP,
        },
        "months": int(portfolio.return_month.nunique()),
        "average_turnover_nav": float(diagnostics.turnover_nav.mean()),
        "average_net_exposure": float(diagnostics.net_exposure.mean()),
        "max_abs_net_exposure": float(diagnostics.net_exposure.abs().max()),
    }
    (output_dir / "optimizer_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    return config


def _book(
    frame: pd.DataFrame, nav: float, prev_long: set, prev_short: set
) -> tuple[dict[int, float], dict]:
    """Equal-weight sector-matched longs and shorts that ADV can absorb."""
    side = GROSS / 2
    one = nav * side / N_PER_SIDE
    valid = frame.loc[
        (frame.available_count > 0)
        & np.isfinite(frame.final_score)
        & np.isfinite(frame.prc)
        & (frame.dolvol_126d > 0)
    ]
    if REQUIRE_VERIFIED_LABELS:
        valid = valid.loc[
            valid.ticker.notna()
            & valid.company_name.notna()
            & valid.ticker_name_status.ne(_UNVERIFIED_LABEL)
        ]
    long_ok = valid[
        (valid.prc >= MIN_PRICE)
        & (valid.dolvol_126d * LONG_ADV_PARTICIPATION * EXECUTION_DAYS >= one)
    ]
    short_ok = valid[
        (valid.prc >= MIN_PRICE)
        & (valid.dolvol_126d * SHORT_ADV_PARTICIPATION * EXECUTION_DAYS >= one)
    ]
    scarce = short_ok.short_pct >= SHORT_INTEREST_CAP  # NaN compares False
    short_ok = short_ok.loc[~scarce]
    if SECTOR_NEUTRAL:
        counts = pd.concat([long_ok.sector, short_ok.sector]).value_counts()
        quota = _quota(counts[counts >= MIN_SECTOR_NAMES], N_PER_SIDE)
        groups = [
            (long_ok[long_ok.sector == s], short_ok[short_ok.sector == s], q)
            for s, q in quota.items()
            if q > 0
        ]
    else:
        groups = [(long_ok, short_ok, N_PER_SIDE)]
    longs, shorts = [], []
    for lo, so, k in groups:
        k = min(k, len(lo), len(so))
        short_ids = _pick(so, k, prev_short, ascending=True)
        long_ids = _pick(lo[~lo.permno.isin(short_ids)], k, prev_long, ascending=False)
        k = min(len(short_ids), len(long_ids))
        longs += long_ids[:k]
        shorts += short_ids[:k]
    coverage = {
        "candidates": len(valid),
        "long_eligible": len(long_ok),
        "short_eligible": len(short_ok),
        "selected_longs": len(longs),
        "selected_shorts": len(shorts),
        "short_blocked": int(scarce.sum()),
    }
    if not longs or not shorts:
        return {}, coverage
    long_notional = short_notional = side
    if BETA_NEUTRAL:
        beta = frame.set_index("permno").beta_60m.fillna(1.0).clip(*BETA_BOUNDS)
        long_beta = float(beta.reindex(longs).mean())
        short_beta = float(beta.reindex(shorts).mean())
        long_notional = float(
            np.clip(
                GROSS * short_beta / (long_beta + short_beta),
                side * (1 - BETA_TILT),
                side * (1 + BETA_TILT),
            )
        )
        short_notional = GROSS - long_notional
    weights = {int(p): long_notional / len(longs) for p in longs}
    weights.update({int(p): -short_notional / len(shorts) for p in shorts})
    return weights, coverage


def _quota(counts: pd.Series, total: int) -> pd.Series:
    """Largest-remainder split of `total` names in proportion to `counts`."""
    base = counts / counts.sum() * total
    quota = np.floor(base).astype(int)
    remainder = (base - quota).sort_values(ascending=False)
    for group in remainder.index[: total - int(quota.sum())]:
        quota[group] += 1
    return quota


def _pick(candidates: pd.DataFrame, k: int, incumbents: set, ascending: bool) -> list:
    """Best k by score, first keeping incumbents ranked within the buffer."""
    if k <= 0 or candidates.empty:
        return []
    order = candidates.sort_values(
        ["final_score", "permno"], ascending=[ascending, True]
    ).permno.to_numpy()
    zone = set(order[: int(np.ceil(k * (1 + BUFFER)))]) if BUFFER else set()
    kept = [p for p in order if p in incumbents and p in zone][:k]
    taken = set(kept)
    return kept + [p for p in order if p not in taken][: k - len(kept)]
