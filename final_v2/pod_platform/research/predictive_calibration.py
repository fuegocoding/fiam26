"""Dated, peer-relative forecast calibration in return units.

Stocks form observations, native reviews form equally weighted bin effects, and
formation months form the uncertainty sample. Portfolio profits and the
dimensionless strategy-selection score never enter this calibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_scope import cohort_mask


SCORE_BINS = 5
MINIMUM_PEER_NAMES = 8
CALIBRATION_PRIOR_MONTHS = 12.0


def declared_horizon(strategy):
    # Kept compatible with saved proposals created before explicit horizons.
    try:
        from .forecast_labels import prediction_horizon
    except ImportError:
        return int(getattr(strategy, "forecast_horizon_days", None) or strategy.expected_holding_days)
    return int(prediction_horizon(strategy))


def horizon_statistics(values, horizon):
    """Monthly mean with a horizon-aware Bartlett/Newey-West standard error."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"months": len(values), "mean": None, "standard_error": None,
            "t_stat": None, "hac_lags": 0}
    mean = float(values.mean())
    centered = values - mean
    lags = min(max(3, int(np.ceil(horizon / 21)) - 1), len(values) - 2)
    variance = float(centered @ centered / len(values))
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(
            centered[lag:] @ centered[:-lag] / len(values))
    se = float(np.sqrt(max(0, variance) / (len(values) - 1)))
    return {"months": len(values), "mean": mean, "standard_error": se,
        "t_stat": float(mean / se) if se > 1e-12 else None, "hac_lags": lags}


def peer_frame(strategy, raw, scores, columns):
    """Industry/size peers fixed using the contemporaneous formation cohort."""
    columns = np.asarray(columns, dtype=int)
    sector = raw.gics.fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    digits = 4 if any(len(p) >= 4 for p in strategy.universe.gics_prefixes) else 2
    data = pd.DataFrame({"column": columns, "stock_id": raw.index.to_numpy()[columns],
        "score": np.asarray(scores, float)[columns], "cap": raw.me.to_numpy()[columns],
        "industry": sector.str[:digits].to_numpy()[columns]})
    data["cap_bin"] = data.groupby("industry").cap.transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False, duplicates="drop"))
    data["peer"] = data.industry + "_" + data.cap_bin.astype(str)
    data["peer_size"] = data.groupby("peer").score.transform("size")
    return data


def binned_active_forecasts(frame):
    """Map only nonzero forecasts to reproducible within-peer score quintiles."""
    active = (np.isfinite(frame.score) & (np.abs(frame.score) > 1e-12)
        & (frame.peer_size >= MINIMUM_PEER_NAMES))
    data = frame.loc[active].copy()
    if data.empty:
        return data.assign(signal_side=pd.Series(dtype=str), score_bin=pd.Series(dtype=int))
    count = data.groupby("peer").score.transform("size")
    percentile = (data.groupby("peer").score.rank() - 0.5) / count
    data["score_bin"] = np.minimum(SCORE_BINS - 1, (SCORE_BINS * percentile).astype(int))
    data["signal_side"] = np.where(data.score > 0, "positive", "negative")
    return data


def append_calibration_observations(rows, strategy, frame, outcome, *, decision_date,
        formation_date, settlement_date, horizon, outcome_horizon=None, forecast_age=0,
        normalize_return_units=False):
    """Compute every active stock effect, retain review/bin sufficient statistics."""
    observed_horizon = horizon if outcome_horizon is None else int(outcome_horizon)
    factor = horizon / observed_horizon if normalize_return_units and observed_horizon > 0 else 1.0
    assumption = ("constant_peer_relative_return_per_session" if normalize_return_units
        else "full_declared_horizon_return")
    common = {"decision_date": pd.Timestamp(decision_date),
        "formation_date": pd.Timestamp(formation_date),
        "settlement_date": settlement_date, "horizon_sessions": horizon,
        "outcome_horizon_sessions": max(0, observed_horizon),
        "forecast_age_sessions": int(forecast_age), "return_normalization_factor": factor,
        "return_unit_assumption": assumption}
    rows.append({**common, "observation_type": "review", "stock_id": None,
        "signal_side": "none", "score_bin": -1, "peer_relative_return": np.nan,
        "cohort_names": len(frame)})
    if outcome is None:
        return
    data = frame.assign(realized_return=np.asarray(outcome, float))
    total = data.groupby("peer").realized_return.transform("sum")
    data["peer_return"] = (total - data.realized_return) / (data.peer_size - 1).clip(lower=1)
    data["raw_peer_relative_return"] = data.realized_return - data.peer_return
    # Pool only calibration effects in common declared-horizon units. Rank IC
    # uses actual remaining-window outcomes without rescaling. This assumes
    # constant expected peer-relative return per session, rather than extending
    # the position or reading returns after its known expiry.
    data["peer_relative_return"] = data.raw_peer_relative_return * factor
    active = binned_active_forecasts(data)
    active["peer_hit"] = np.where(active.signal_side == "positive", 1, -1) * active.peer_relative_return > 0
    bins = active.groupby(["signal_side", "score_bin"], sort=True).agg(
        peer_relative_return=("peer_relative_return", "mean"),
        raw_peer_relative_return=("raw_peer_relative_return", "mean"),
        realized_return=("realized_return", "mean"), peer_return=("peer_return", "mean"),
        peer_hit_rate=("peer_hit", "mean"), stock_forecasts=("stock_id", "size"),
        peer_groups=("peer", "nunique"), minimum_peer_names=("peer_size", "min"),
        mean_score=("score", "mean")).reset_index()
    for record in bins.to_dict("records"):
        rows.append({**common, **record, "observation_type": "bin", "cohort_names": len(frame)})


def monthly_calibration_observations(observations):
    """Average stocks, then native reviews, then retain one effect per month/bin.

    A missing outcome on any native review keeps the entire formation month
    unavailable. Review counts and stock counts are coverage, never iid sample
    sizes for confidence intervals.
    """
    columns = ["formation_month", "settlement_date", "signal_side", "score_bin",
        "peer_relative_return", "peer_hit_rate", "stock_forecasts", "native_reviews",
        "cohort_reviews", "horizon_sessions", "first_decision_date", "last_decision_date",
        "raw_peer_relative_return", "outcome_horizon_sessions", "minimum_outcome_horizon_sessions",
        "maximum_outcome_horizon_sessions", "forecast_age_sessions", "maximum_forecast_age_sessions",
        "return_normalization_factor", "return_unit_assumption"]
    if observations is None or observations.empty:
        return pd.DataFrame(columns=columns)
    data = observations.copy()
    for name in ("formation_date", "decision_date", "settlement_date"):
        data[name] = pd.to_datetime(data[name])
    data["formation_month"] = data.formation_date.dt.to_period("M")
    # Old compact tables are already in full-horizon units.
    defaults = {"raw_peer_relative_return": data.peer_relative_return,
        "outcome_horizon_sessions": data.horizon_sessions, "forecast_age_sessions": 0,
        "return_normalization_factor": 1.0, "return_unit_assumption": "full_declared_horizon_return"}
    for name, value in defaults.items():
        if name not in data:
            data[name] = value
    rows = []
    for month, cohort in data.groupby("formation_month", sort=True):
        reviews = cohort.loc[cohort.observation_type == "review"]
        if reviews.empty:
            # Legacy or externally supplied observations still need every review
            # represented; all rows must carry an outcome settlement timestamp.
            reviews = cohort.drop_duplicates(["formation_date", "decision_date"])
        complete = reviews.settlement_date.notna().all() and cohort.settlement_date.notna().all()
        settlement = max(cohort.settlement_date.max(), month.end_time.normalize()) if complete else pd.NaT
        forecasts = cohort.loc[(cohort.observation_type == "bin")
            & np.isfinite(cohort.peer_relative_return)].copy()
        if forecasts.empty:
            continue
        native = forecasts.groupby(["formation_date", "decision_date", "signal_side", "score_bin"])
        native = native.agg(peer_relative_return=("peer_relative_return", "mean"),
            raw_peer_relative_return=("raw_peer_relative_return", "mean"),
            peer_hit_rate=("peer_hit_rate", "mean"), stock_forecasts=("stock_forecasts", "sum"),
            outcome_horizon_sessions=("outcome_horizon_sessions", "first"),
            forecast_age_sessions=("forecast_age_sessions", "first"),
            return_normalization_factor=("return_normalization_factor", "first")).reset_index()
        for (side, score_bin), group in native.groupby(["signal_side", "score_bin"], sort=True):
            rows.append({"formation_month": str(month), "settlement_date": settlement,
                "signal_side": side, "score_bin": int(score_bin),
                "peer_relative_return": float(group.peer_relative_return.mean()) if complete else np.nan,
                "peer_hit_rate": float(group.peer_hit_rate.mean()) if complete else np.nan,
                "stock_forecasts": int(group.stock_forecasts.sum()), "native_reviews": len(group),
                "cohort_reviews": len(reviews), "horizon_sessions": int(cohort.horizon_sessions.iloc[0]),
                "first_decision_date": reviews.decision_date.min(),
                "last_decision_date": reviews.decision_date.max(),
                "raw_peer_relative_return": float(group.raw_peer_relative_return.mean()) if complete else np.nan,
                "outcome_horizon_sessions": float(group.outcome_horizon_sessions.mean()),
                "minimum_outcome_horizon_sessions": int(group.outcome_horizon_sessions.min()),
                "maximum_outcome_horizon_sessions": int(group.outcome_horizon_sessions.max()),
                "forecast_age_sessions": float(group.forecast_age_sessions.mean()),
                "maximum_forecast_age_sessions": int(group.forecast_age_sessions.max()),
                "return_normalization_factor": float(group.return_normalization_factor.mean()),
                "return_unit_assumption": "|".join(sorted(cohort.return_unit_assumption.unique()))})
    return pd.DataFrame(rows, columns=columns)


def _conservative_mean(stats, minimum, uncertainty_haircut):
    """Shrink effects towards zero; never turn missing uncertainty into proof."""
    n, mean, se = stats["months"], stats["mean"], stats["standard_error"]
    supported = n >= max(2, minimum) and mean is not None and se is not None and se > 1e-12
    if not supported:
        return 0.0, False
    shrunk = mean * n / (n + CALIBRATION_PRIOR_MONTHS)
    # Keep the estimated standard error unshrunk: prior shrinkage is not extra
    # evidence and does not manufacture a narrower confidence interval.
    conservative = np.sign(shrunk) * max(0, abs(shrunk) - uncertainty_haircut * se)
    return float(conservative), True


def expected_target_edge(trial, asof, target, scores, formation_frame=None, minimum=12,
        uncertainty_haircut=1.0, entry_mask=None, minimum_coverage=0.65):
    """Calibrate actual target weights using only labels settled at ``asof``.

    Return units are annualized peer-relative contribution per unit of sleeve
    capital, using 252 / declared-horizon simple scaling. Age-aware observations
    pool remaining-window effects assuming constant peer-relative return per
    session; raw effects and durations remain in the evidence table. Zero-score
    positions are passive hedges with alpha zero, supplying no bearish evidence.
    """
    strategy = trial["strategy"]
    horizon = declared_horizon(strategy)
    target, scores = np.asarray(target, float), np.asarray(scores, float)
    empty = {"method": "settled_monthly_peer_bin_calibration_v1",
        "expected_annual_unit_edge": 0.0, "expected_annual_unit_edge_raw": 0.0,
        "funding_supported": False, "forecast_coverage": 0.0,
        "passive_hedge_gross": 0.0, "horizon_sessions": horizon, "sides": {}, "bins": [],
        "reason": "missing_calibration_or_formation_frame",
        "uncertainty_scope": "monthly_HAC_not_multiple_search_corrected"}
    if target.shape != scores.shape or target.ndim != 1:
        raise ValueError("Target weights and native scores must be aligned one-dimensional vectors")
    if not np.isfinite(target).all():
        raise ValueError("Target weights must all be finite")
    observations = trial.get("predictive_calibration")
    if (observations is None or observations.empty or formation_frame is None or asof is None):
        return empty
    if len(formation_frame) != len(target):
        raise ValueError("Formation frame must use the same security order as targets and scores")
    if uncertainty_haircut < 0 or not np.isfinite(uncertainty_haircut):
        raise ValueError("Uncertainty haircut must be finite and nonnegative")
    asof = pd.Timestamp(asof)
    start = asof - pd.DateOffset(months=strategy.evidence_lookback_months)
    available = observations.copy()
    available["settlement_date"] = pd.to_datetime(available.settlement_date)
    available = available.loc[(available.settlement_date > start)
        & (available.settlement_date <= asof) & np.isfinite(available.peer_relative_return)]
    if "horizon_sessions" in available:
        available = available.loc[available.horizon_sessions == horizon]
    scope = cohort_mask(formation_frame, strategy.universe)
    if entry_mask is not None:
        scope &= np.asarray(entry_mask, bool)
    frame = peer_frame(strategy, formation_frame, scores, np.flatnonzero(scope))
    current = binned_active_forecasts(frame)
    # A nonzero score outside the declared cohort is not a forecast to fund.
    in_scope = np.zeros(len(target), bool)
    in_scope[frame.column.to_numpy(dtype=int)] = True
    nonzero_score = np.isfinite(scores) & (np.abs(scores) > 1e-12)
    active_position = np.isfinite(target) & (np.abs(target) > 1e-12)
    passive = active_position & np.isfinite(scores) & ~nonzero_score
    if getattr(strategy, "hedge_policy", None) == "liquid_sector":
        # Construction chooses the liquid short names independently of their
        # raw signal, which may happen to be nonzero. They are still hedges.
        passive |= active_position & (target < 0)
    forecast_position = active_position & ~passive
    active_gross = float(np.abs(target[forecast_position]).sum())
    passive_gross = float(np.abs(target[passive]).sum())
    estimates = {}
    bins = []
    for (side, score_bin), group in available.groupby(["signal_side", "score_bin"], sort=True):
        # Require unique monthly effects even for externally loaded duplicate rows.
        values = group.groupby("formation_month").peer_relative_return.mean().sort_index()
        stats = horizon_statistics(values, horizon)
        effect, measurable = _conservative_mean(stats, minimum, uncertainty_haircut)
        key = (str(side), int(score_bin))
        estimates[key] = (effect, measurable, stats, values)
        bins.append({"signal_side": key[0], "score_bin": key[1], **stats,
            "shrunk_conservative_horizon_effect": effect, "measurable": measurable})
    alpha = np.zeros(len(target), float)
    raw_alpha = np.zeros(len(target), float)
    covered = np.zeros(len(target), bool)
    position_bins = {}
    for item in current.itertuples(index=False):
        key = (item.signal_side, int(item.score_bin))
        position_bins[int(item.column)] = key
        estimate = estimates.get(key)
        if estimate is not None and estimate[1]:
            alpha[item.column] = estimate[0]
            raw_alpha[item.column] = estimate[2]["mean"]
            covered[item.column] = True
    alpha[passive] = 0.0
    raw_alpha[passive] = 0.0
    covered[passive] = False
    covered_gross = float(np.abs(target[forecast_position & covered]).sum())
    coverage = covered_gross / active_gross if active_gross > 1e-12 else 0.0
    annualizer = 252.0 / horizon
    annual = float(np.dot(np.nan_to_num(target), alpha) * annualizer)
    annual_raw = float(np.dot(np.nan_to_num(target), raw_alpha) * annualizer)
    sides = {}
    for name, sign in (("long", 1), ("short", -1)):
        positions = forecast_position & (sign * target > 0)
        gross = float(np.abs(target[positions]).sum())
        key_weights = {}
        for column in np.flatnonzero(positions & covered):
            key = position_bins[column]
            key_weights[key] = key_weights.get(key, 0.0) + float(target[column])
        # Side evidence is a contemporaneous target-weighted monthly effect. A
        # month missing any used bin is unavailable, rather than reweighted after
        # observing which labels or names survive.
        monthly = None
        for key, weight in key_weights.items():
            values = estimates[key][3].rename(str(key)) * weight
            monthly = values.to_frame() if monthly is None else monthly.join(values, how="outer")
        side_values = monthly.dropna().sum(axis=1) / gross if monthly is not None and gross > 0 else []
        stats = horizon_statistics(side_values, horizon)
        side_effect, measurable = _conservative_mean(stats, minimum, uncertainty_haircut)
        side_coverage = float(np.abs(target[positions & covered]).sum()) / gross if gross > 0 else 0.0
        contribution = float(np.dot(np.nan_to_num(target[positions]), alpha[positions]) * annualizer)
        sides[name] = {**stats, "gross": gross, "forecast_coverage": side_coverage,
            "mean_peer_relative_return": stats["mean"],
            "conservative_peer_relative_return": side_effect,
            "expected_annual_contribution": contribution,
            "supported": bool(measurable and side_effect > 0 and side_coverage >= minimum_coverage),
            "passive_hedge_gross": float(np.abs(target[passive & (sign * target > 0)]).sum()),
            "reason": "passive_hedge_only" if gross == 0 and np.any(passive & (sign * target > 0))
                else "no_active_forecast_positions" if gross == 0 else "settled_peer_relative_evidence"}
    active_sides = [side for side in sides.values() if side["gross"] > 1e-12]
    funded = bool(active_gross > 0 and coverage >= minimum_coverage and annual > 0
        and all(side["supported"] for side in active_sides))
    reason = ("positive_calibrated_peer_edge" if funded else
        "no_active_forecast_positions" if active_gross <= 1e-12 else
        "insufficient_settled_bin_coverage" if coverage < minimum_coverage else
        "unsupported_active_side" if not all(side["supported"] for side in active_sides) else
        "nonpositive_calibrated_peer_edge")
    return {**empty, "expected_annual_unit_edge": annual,
        "expected_annual_unit_edge_raw": annual_raw, "funding_supported": funded,
        "forecast_coverage": coverage, "active_forecast_gross": active_gross,
        "passive_hedge_gross": passive_gross,
        "out_of_scope_forecast_gross": float(np.abs(target[forecast_position & ~in_scope]).sum()),
        "asof": str(asof.date()), "latest_settlement":
            str(available.settlement_date.max().date()) if not available.empty else None,
        "minimum_months": max(2, minimum), "minimum_coverage": minimum_coverage,
        "prior_months": CALIBRATION_PRIOR_MONTHS, "uncertainty_haircut": uncertainty_haircut,
        "annualization": "simple_252_over_declared_horizon", "sides": sides, "bins": bins,
        "calibration_return_unit_assumption": sorted(available.return_unit_assumption.unique())
            if "return_unit_assumption" in available else ["full_declared_horizon_return"],
        "calibration_forecast_age_sessions_max": float(available.maximum_forecast_age_sessions.max())
            if not available.empty and "maximum_forecast_age_sessions" in available else None,
        "calibration_outcome_horizon_sessions_min": float(available.minimum_outcome_horizon_sessions.min())
            if not available.empty and "minimum_outcome_horizon_sessions" in available else None,
        "reason": reason}
