"""Every feasible ranked prefix, measured without dollar capital or fill assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pod_plans import construction_designs, plan_signal


def mean_evidence(values):
    """Monthly mean and a Newey-West interval; exploratory, not multiplicity corrected."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count < 2:
        return {
            "months": count,
            "mean": None,
            "hac_se": None,
            "lower_95": None,
            "first_half_mean": None,
            "second_half_mean": None,
        }
    mean = float(values.mean())
    centered = values - mean
    variance = float(centered @ centered / count)
    lags = min(3, count - 1)
    for lag in range(1, lags + 1):
        variance += (
            2
            * (1 - lag / (lags + 1))
            * float(centered[lag:] @ centered[:-lag] / count)
        )
    se = float(np.sqrt(max(0, variance) / count))
    return {
        "months": count,
        "mean": mean,
        "hac_se": se,
        "lower_95": mean - 1.96 * se,
        "first_half_mean": float(values[: count // 2].mean()),
        "second_half_mean": float(values[count // 2 :].mean()),
        "interval_scope": "pointwise_monthly_HAC_not_multiple_search_correction",
    }


def signal_prefixes(plan, panel, *, signal=None):
    """Compare symmetric long/short tails for every available integer count.

    Future realized returns are used only to score a previously formed signal.
    Residual returns benchmark the declared controls; they are not forecasts,
    executable portfolio returns or after-cost alpha. Short tails respect the
    $500m formation floor. All supplied eligible stocks enter the ranking.
    """
    # Discover predictive strength before fitting entry/exit or book size.
    if signal is None:
        signal = plan_signal(plan, panel, apply_thresholds=False)
    designs = construction_designs(panel, plan.controls)
    rows, information = [], []
    for t, date in enumerate(panel.dates[:-1]):
        mask, design, inverse = designs[t]
        # Cohort and rank membership are fixed at formation. An unfilled entry
        # remains cash in the supplied realized label; do not promote another
        # name after seeing the execution session's quote availability.
        valid = mask & np.isfinite(signal[t])
        indices = np.flatnonzero(valid)
        if len(indices) < 5:
            continue
        raw = panel.returns[t, indices].astype(float)
        if not np.isfinite(raw).all():
            raise ValueError(
                "Return labels must encode the accepted missing-entry cash rule"
            )
        local = design[np.flatnonzero(valid[mask])]
        realized = raw - local @ (np.linalg.pinv(local) @ raw)
        known = np.nan_to_num(signal[t, mask]).astype(float)
        known -= design @ (inverse @ known)
        forecasts = known[valid[mask]]
        rank_x = pd.Series(forecasts).rank().to_numpy()
        rank_y = pd.Series(realized).rank().to_numpy()
        ic = (
            float(np.corrcoef(rank_x, rank_y)[0, 1])
            if rank_x.std() and rank_y.std()
            else np.nan
        )
        shortable = panel.frames[t].me.to_numpy()[indices] >= 500
        if shortable.sum() > 4:
            sx = pd.Series(forecasts[shortable]).rank().to_numpy()
            sy = pd.Series(realized[shortable]).rank().to_numpy()
            short_ic = (
                float(np.corrcoef(sx, sy)[0, 1])
                if sx.std() and sy.std()
                else np.nan
            )
        else:
            short_ic = np.nan
        information.append(
            (
                date,
                panel.execution_dates[t],
                panel.settlement_dates[t],
                ic,
                short_ic,
            )
        )
        long = np.flatnonzero(forecasts > 1e-12)
        short = np.flatnonzero((forecasts < -1e-12) & shortable)
        long = long[np.argsort(-forecasts[long], kind="stable")]
        short = short[np.argsort(forecasts[short], kind="stable")]
        pairs = min(len(long), len(short))
        if pairs == 0:
            continue
        counts = np.arange(1, pairs + 1)
        controlled = (
            np.cumsum(realized[long[:pairs]])
            - np.cumsum(realized[short[:pairs]])
        ) / counts
        raw_spread = (
            np.cumsum(raw[long[:pairs]]) - np.cumsum(raw[short[:pairs]])
        ) / counts
        rows.append(
            pd.DataFrame(
                {
                    "formation_date": date,
                    "execution_date": panel.execution_dates[t],
                    "settlement_date": panel.settlement_dates[t],
                    "names": counts * 2,
                    "controlled_spread": controlled,
                    "gross_spread": raw_spread,
                    "controlled_long_mean": np.cumsum(realized[long[:pairs]])
                    / counts,
                    "controlled_short_mean": np.cumsum(realized[short[:pairs]])
                    / counts,
                }
            )
        )
    prefixes = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "formation_date",
                "settlement_date",
                "names",
                "controlled_spread",
                "gross_spread",
            ]
        )
    )
    ic = pd.DataFrame(
        information,
        columns=[
            "formation_date",
            "execution_date",
            "settlement_date",
            "rank_ic",
            "shortable_rank_ic",
        ],
    )
    return prefixes, ic


def summarize_prefixes(prefixes):
    rows = []
    for count, group in prefixes.groupby("names", sort=True):
        evidence = mean_evidence(group.controlled_spread)
        rows.append(
            {
                "names": int(count),
                **evidence,
                "annualized_arithmetic_controlled_spread": evidence["mean"] * 12
                if evidence["mean"] is not None
                else None,
                "annualized_arithmetic_gross_spread": float(
                    group.gross_spread.mean() * 12
                ),
            }
        )
    return pd.DataFrame(rows)


def broad_positive_regions(curve, minimum_months=24):
    """Report contiguous supported regions; never promote a single best point."""
    if curve.empty:
        return []
    supported = curve.loc[
        (curve.months >= minimum_months)
        & (curve.lower_95 > 0)
        & (curve.first_half_mean > 0)
        & (curve.second_half_mean > 0)
    ]
    regions = []
    for _, group in supported.groupby((supported.names.diff() != 2).cumsum()):
        regions.append(
            {
                "minimum_names": int(group.names.min()),
                "maximum_names": int(group.names.max()),
                "evaluated_counts": len(group),
                "minimum_monthly_lower_95": float(group.lower_95.min()),
            }
        )
    return regions


def freeze_prefix_choice(candidates, cutoff, minimum_months=24):
    """Select a broad supported prefix using only outcomes settled by cutoff."""
    eligible = []
    for identity, prefixes in candidates.items():
        prior = prefixes.loc[prefixes.settlement_date <= cutoff]
        curve = summarize_prefixes(prior)
        regions = broad_positive_regions(curve, minimum_months)
        for region in regions:
            # An isolated count does not establish a usable concentration range.
            if region["evaluated_counts"] < 5:
                continue
            region_curve = curve.loc[
                curve.names.between(
                    region["minimum_names"], region["maximum_names"]
                )
            ]
            score = float(region_curve.lower_95.median())
            eligible.append((score, identity, region["maximum_names"]))
    if not eligible:
        return None
    _, identity, count = max(eligible)
    return {
        "strategy_id": identity,
        "names": count,
        "selection_asof": str(pd.Timestamp(cutoff).date()),
    }
