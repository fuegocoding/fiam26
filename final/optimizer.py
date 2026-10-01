"""Builds a sector- and beta-neutral long/short book from monthly scores."""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

# Every dollar limit derives from NAV or a stock's ADV: change NAV to resize.
NAV = 10_000_000.0
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
# Long and short notionals are scaled so their 60-month betas offset; each
# side stays within BETA_TILT of half the gross, keeping net exposure small.
BETA_NEUTRAL = True
BETA_TILT = 0.25
BETA_BOUNDS = (0.2, 3.0)

_UNVERIFIED_LABEL = "no_verified_label_for_observation_period"
_ATTRIBUTES = [
    "final_score", "ticker", "company_name", "ticker_name_reference_date",
    "ticker_name_source", "ticker_name_status", "prc", "size_grp",
    "dolvol_126d", "sector", "beta_60m",
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


def optimize(
    predictions: pd.DataFrame, nav: float = NAV
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds each month's book from the final score.

    A holding absent from a month's predictions has already exited at its
    last price, so only names present that month can be kept.

    Args:
        predictions: One row per stock-month with return_month, permno,
            final_score, price, liquidity, sector, beta and labels.
        nav: Net asset value that sets every dollar size.

    Returns:
        Target positions per month, and per-month exposures and coverage.
    """
    frame = predictions.copy()
    frame["return_month"] = pd.to_datetime(frame.return_month).dt.to_period("M")
    previous: dict[int, float] = {}
    positions, diagnostics = [], []
    for month, current in frame.groupby("return_month", sort=True):
        prev_long = {p for p, w in previous.items() if w > 0}
        prev_short = {p for p, w in previous.items() if w < 0}
        weights, coverage = _book(current, nav, prev_long, prev_short)
        attributes = current.set_index("permno")[_ATTRIBUTES]
        for permno, weight in weights.items():
            row = attributes.loc[permno]
            positions.append({
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
            })
        book = pd.Series(weights, dtype=float)
        sectors = attributes.sector.reindex(book.index)
        names = set(weights) | set(previous)
        traded = sum(
            abs(weights.get(p, 0.0) - previous.get(p, 0.0)) for p in names
        )
        diagnostics.append({
            "return_month": month,
            "long_notional": float(book[book > 0].sum()),
            "short_notional": float(-book[book < 0].sum()),
            "net_exposure": float(book.sum()),
            "sector_net_abs": float(book.groupby(sectors).sum().abs().sum()),
            "beta_exposure": float(
                (book * attributes.beta_60m.reindex(book.index).fillna(1.0))
                .sum()
            ),
            "turnover_nav": float(traded),
            **coverage,
        })
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
    stored.to_parquet(
        output_dir / "portfolio.parquet", index=False, compression="zstd"
    )
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
        "beta_neutral": BETA_NEUTRAL,
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
    if SECTOR_NEUTRAL:
        counts = pd.concat([long_ok.sector, short_ok.sector]).value_counts()
        quota = _quota(counts[counts >= MIN_SECTOR_NAMES], N_PER_SIDE)
        groups = [
            (long_ok[long_ok.sector == s], short_ok[short_ok.sector == s], q)
            for s, q in quota.items() if q > 0
        ]
    else:
        groups = [(long_ok, short_ok, N_PER_SIDE)]
    longs, shorts = [], []
    for lo, so, k in groups:
        k = min(k, len(lo), len(so))
        short_ids = _pick(so, k, prev_short, ascending=True)
        long_ids = _pick(
            lo[~lo.permno.isin(short_ids)], k, prev_long, ascending=False
        )
        k = min(len(short_ids), len(long_ids))
        longs += long_ids[:k]
        shorts += short_ids[:k]
    coverage = {
        "candidates": len(valid), "long_eligible": len(long_ok),
        "short_eligible": len(short_ok), "selected_longs": len(longs),
        "selected_shorts": len(shorts),
    }
    if not longs or not shorts:
        return {}, coverage
    long_notional = short_notional = side
    if BETA_NEUTRAL:
        beta = (
            frame.set_index("permno").beta_60m.fillna(1.0).clip(*BETA_BOUNDS)
        )
        long_beta = float(beta.reindex(longs).mean())
        short_beta = float(beta.reindex(shorts).mean())
        long_notional = float(np.clip(
            GROSS * short_beta / (long_beta + short_beta),
            side * (1 - BETA_TILT), side * (1 + BETA_TILT),
        ))
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


def _pick(
    candidates: pd.DataFrame, k: int, incumbents: set, ascending: bool
) -> list:
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


def main() -> None:
    """Re-sizes a saved predictions file, e.g. after changing NAV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=pathlib.Path, required=True,
        help="predictions.parquet written by combiner.py",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path,
        help="defaults to the folder of the predictions file",
    )
    args = parser.parse_args()
    portfolio, diagnostics = optimize(pd.read_parquet(args.predictions))
    output_dir = args.output_dir or args.predictions.parent
    print(json.dumps(write(portfolio, diagnostics, output_dir), indent=2))


if __name__ == "__main__":
    main()
