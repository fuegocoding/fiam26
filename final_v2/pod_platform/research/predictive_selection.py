"""Dated cohort forecast evidence and the declared four-component selector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy_scope import scoped_panel
from .predictive_calibration import (
    append_calibration_observations,
    declared_horizon,
    horizon_statistics,
    peer_frame,
)


ROBUSTNESS_CHECKS = {
    "label_start_plus_one_session": "Start the outcome window one session later.",
    "horizon_80_percent": "Use 80% of the declared prediction horizon.",
    "horizon_120_percent": "Use 120% of the declared prediction horizon.",
    "trim_score_tails_5_percent": "Remove the outer 5% of active scores at each end.",
}
EXPANDED_ROBUSTNESS_CHECKS = {
    "issuer_subgroup_0": "Forecast stability in the first fixed issuer subgroup; not a training holdout.",
    "issuer_subgroup_1": "Forecast stability in the second fixed issuer subgroup; not a training holdout.",
    "issuer_out_of_fit": "Executable forecasts from models excluding each evaluated issuer from all training rows.",
    "forward_disjoint_blocks": "Positive IC in two forward blocks with nonoverlapping outcome windows.",
    "within_peer_shuffled_signal": "Mean IC exceeds a 95% within-peer shuffled-score control.",
}
SHUFFLED_CONTROLS = 32
SCORE_WEIGHTS = {"P": 0.40, "C": 0.25, "R": 0.20, "B": 0.15}


def _outcome(market, columns, begin, horizon):
    finish = begin + horizon
    if horizon < 1 or begin >= len(market.dates) or finish >= len(market.dates):
        return None, pd.NaT
    path = market.returns[begin + 1 : finish + 1, columns].astype(np.float64, copy=False)
    valid = np.logical_and.accumulate(np.isfinite(path) & (path >= -1), axis=0)
    # Match the runtime's last-known-price exit, rather than selecting survivors.
    values = np.prod(np.where(valid, 1 + path, 1), axis=0) - 1
    return values, market.dates[finish]


def _rank_pairs(frame, outcome, active, *, trim=False):
    if outcome is None:
        return None
    data = frame.assign(outcome=outcome)
    data["peer_size"] = data.groupby("peer").score.transform("size")
    data["y"] = (data.groupby("peer").outcome.rank() - 0.5) / data.peer_size
    active = active.copy()
    if trim and active.sum() >= 8:
        low, high = np.quantile(data.score.to_numpy()[active], [0.05, 0.95])
        active &= data.score.between(low, high).to_numpy()
    data = data.loc[active & (data.peer_size >= 8)].copy()
    data["active_size"] = data.groupby("peer").score.transform("size")
    data = data.loc[data.active_size >= 4].copy()
    if len(data) < 8:
        return data.assign(x=np.nan)
    data["x"] = (data.groupby("peer").score.rank() - 0.5) / data.active_size
    return data


def _rank_ic(frame, outcome, active, *, trim=False):
    data = _rank_pairs(frame, outcome, active, trim=trim)
    if data is None:
        return np.nan, 0
    if len(data) < 8 or data.x.std() <= 1e-12 or data.y.std() <= 1e-12:
        return np.nan, len(data)
    return float(np.corrcoef(data.x, data.y)[0, 1]), len(data)


def _shuffled_ic(frame, outcome, active):
    """Conditional null preserving each peer's score multiset and outcome rows.

    Fixed issuer hash orderings are reused across dates, avoiding independently
    redrawing an issuer's permutation on every review. This control does not
    correct for selection across all hypotheses tried by the research agents.
    """
    data = _rank_pairs(frame, outcome, active)
    if (data is None or len(data) < 8 or data.x.std() <= 1e-12
            or data.y.std() <= 1e-12):
        return {}
    x, y = data.x.to_numpy(), data.y.to_numpy()
    groups = data.groupby("peer", sort=False).indices.values()
    groups = list(groups)
    ids = data.stock_id.astype(str).to_numpy()
    base_order = np.argsort(ids, kind="stable")
    identity_rank = np.empty(len(data), int)
    identity_rank[base_order] = np.arange(len(data))
    results = {}
    for repetition in range(SHUFFLED_CONTROLS):
        keys = pd.util.hash_array(
            np.char.add(ids.astype(str), f"/shuffle/{repetition}").astype(object), categorize=False)
        permuted = x.copy()
        for group in groups:
            destination = group[np.argsort(identity_rank[group], kind="stable")]
            source = group[np.argsort(keys[group], kind="stable")]
            permuted[destination] = x[source]
        results[f"shuffle_ic_{repetition:02d}"] = float(np.corrcoef(permuted, y)[0, 1])
    return results


def forecast_ic_observations(strategy, panel, market, decisions, calibration=None, issuer_oof=None,
        forecast_metadata=None):
    """Measure native forecasts at their declared horizon inside their own cohort.

    No portfolio weights or profits enter these observations. When vintage
    metadata is supplied, a reused forecast is measured only until its original
    origin plus declared horizon; reviewing it does not renew its lifetime.
    Perturbations use their own fixed origin-relative endpoints. A monthly
    cohort becomes usable only after all native reviews have settled.
    """
    panel = scoped_panel(panel, strategy.universe)
    horizon = declared_horizon(strategy)
    horizons = {
        "base": (0, horizon),
        "label_start_plus_one_session": (1, horizon),
        "horizon_80_percent": (0, max(1, int(round(horizon * 0.8)))),
        "horizon_120_percent": (0, max(horizon + 1, int(round(horizon * 1.2)))),
        "trim_score_tails_5_percent": (0, horizon),
    }
    rows = []
    for day, (formation, scores) in sorted(decisions.items()):
        origin = day
        if forecast_metadata is not None:
            vintage = forecast_metadata.get(day, forecast_metadata.get(str(day)))
            if vintage is None or "origin_day" not in vintage:
                raise ValueError("Age-aware evidence requires an origin for every native forecast")
            origin = vintage["origin_day"]
            if (isinstance(origin, bool) or not isinstance(origin, (int, np.integer))
                    or not 0 <= origin <= day):
                raise ValueError("Forecast origin must be an observed session at or before its review")
        age = day - origin
        raw = panel.frames[formation]
        # The perturbation changes the label, never the formation cohort. An
        # unavailable shifted entry stays in the cohort with a cash outcome.
        columns = np.flatnonzero(panel.mask[formation] & market.tradable[day])
        data = peer_frame(strategy, raw, scores, columns)
        active = (np.isfinite(data.score) & (np.abs(data.score) > 1e-12)).to_numpy()
        outcome_cache = {}
        common = {"decision_date": market.dates[day], "formation_date": panel.dates[formation]}
        for variant, (shift, length) in horizons.items():
            key = (shift, length)
            observed_length = origin + length - day - shift if forecast_metadata is not None else length
            if key not in outcome_cache:
                outcome, settlement = _outcome(market, columns, day + shift, observed_length)
                if shift and outcome is not None:
                    outcome = np.where(market.tradable[day + shift, columns], outcome, 0.0)
                outcome_cache[key] = outcome, settlement
            outcome, settlement = outcome_cache[key]
            ic, count = _rank_ic(data, outcome, active,
                trim=variant == "trim_score_tails_5_percent")
            row = {**common, "variant": variant, "settlement_date": settlement,
                "rank_ic": ic, "stock_forecasts": count,
                "forecast_origin_date": market.dates[origin], "forecast_age_sessions": age,
                "outcome_horizon_sessions": max(0, observed_length)}
            if variant == "base":
                row.update(_shuffled_ic(data, outcome, active))
                if calibration is not None:
                    append_calibration_observations(calibration, strategy, data, outcome,
                        **common, settlement_date=settlement, horizon=horizon,
                        outcome_horizon=observed_length, forecast_age=age,
                        normalize_return_units=forecast_metadata is not None)
            rows.append(row)
        outcome, settlement = outcome_cache[(0, horizon)]
        hashes = pd.util.hash_array(data.stock_id.astype(str).to_numpy(), categorize=False)
        for subgroup in (0, 1):
            ic, count = _rank_ic(data, outcome, active & ((hashes % 2) == subgroup))
            rows.append({**common, "variant": f"issuer_subgroup_{subgroup}",
                "settlement_date": settlement, "rank_ic": ic, "stock_forecasts": count,
                "forecast_origin_date": market.dates[origin], "forecast_age_sessions": age,
                "outcome_horizon_sessions": max(0, horizon - age) if forecast_metadata is not None else horizon})
        if issuer_oof is not None and day in issuer_oof:
            oof_formation, oof_scores = issuer_oof[day]
            if oof_formation != formation:
                raise ValueError("Issuer-heldout evidence must use the same native formation and review dates")
            oof_data = peer_frame(strategy, raw, oof_scores, columns)
            oof_active = (np.isfinite(oof_data.score) & (np.abs(oof_data.score) > 1e-12)).to_numpy()
            ic, count = _rank_ic(oof_data, outcome, oof_active)
            rows.append({**common, "variant": "issuer_out_of_fit",
                "settlement_date": settlement, "rank_ic": ic, "stock_forecasts": count,
                "forecast_origin_date": market.dates[origin], "forecast_age_sessions": age,
                "outcome_horizon_sessions": max(0, horizon - age) if forecast_metadata is not None else horizon})
    columns = ["variant", "decision_date", "formation_date", "settlement_date", "rank_ic", "stock_forecasts"]
    return pd.DataFrame(rows).reindex(columns=[*columns, "forecast_origin_date", "forecast_age_sessions",
        "outcome_horizon_sessions",
        *[f"shuffle_ic_{i:02d}" for i in range(SHUFFLED_CONTROLS)]])


def monthly_ic_observations(observations):
    """Seal formation cohorts without merging month ends moved by weekends."""
    shuffle_columns = [name for name in observations if name.startswith("shuffle_ic_")]
    columns = ["variant", "formation_month", "settlement_date", "rank_ic",
        "stock_forecasts", "native_reviews", "first_decision_date", "last_decision_date", *shuffle_columns]
    if observations.empty:
        return pd.DataFrame(columns=columns)
    data = observations.copy()
    for name in ("formation_date", "decision_date", "settlement_date"):
        data[name] = pd.to_datetime(data[name])
    data = data.assign(formation_month=data.formation_date.dt.to_period("M"))
    rows = []
    for (variant, month), group in data.groupby(["variant", "formation_month"], sort=True):
        complete = group.settlement_date.notna().all()
        settlement = max(group.settlement_date.max(), month.end_time.normalize()) if complete else pd.NaT
        finite = group.rank_ic[np.isfinite(group.rank_ic)]
        row = {"variant": variant, "formation_month": str(month),
            "settlement_date": settlement,
            "rank_ic": float(finite.mean()) if len(finite) and complete else np.nan,
            "stock_forecasts": int(group.stock_forecasts.sum()), "native_reviews": len(group),
            "first_decision_date": group.decision_date.min(), "last_decision_date": group.decision_date.max()}
        for name in shuffle_columns:
            # The entire valid-IC cohort must also have a control observation.
            controls = group.loc[np.isfinite(group.rank_ic), name]
            row[name] = float(controls.mean()) if complete and len(controls) and controls.notna().all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _ic_statistics(values, horizon):
    return horizon_statistics(values, horizon)


def _variant_check(name, description, base, window, minimum):
    perturbed = window.loc[window.variant == name]
    common = base.merge(perturbed, on="formation_month", suffixes=("_base", "_perturbed"))
    performed = len(common) >= max(2, minimum)
    effect = float(common.rank_ic_perturbed.mean()) if performed else None
    return {"name": name, "description": description, "performed": performed,
        "months": len(common), "mean_ic": effect,
        "passed": bool(effect > 0) if performed else None,
        "reason": "measured" if performed else "not_applicable_or_insufficient_settled_evidence"}


def _expanded_checks(base, window, minimum):
    checks = [_variant_check(name, description, base, window, minimum)
        for name, description in EXPANDED_ROBUSTNESS_CHECKS.items()
        if name in {"issuer_subgroup_0", "issuer_subgroup_1", "issuer_out_of_fit"}]
    name = "forward_disjoint_blocks"
    forward = {"name": name, "description": EXPANDED_ROBUSTNESS_CHECKS[name],
        "performed": False, "passed": None, "months": 0,
        "reason": "missing_decision_dates_or_insufficient_disjoint_cohorts"}
    if "first_decision_date" in base and len(base) >= max(6, minimum):
        recent = base.tail(3)
        first_recent_review = pd.to_datetime(recent.first_decision_date).min()
        prior = base.loc[base.settlement_date < first_recent_review].tail(3)
        if pd.notna(first_recent_review) and len(prior) == 3:
            forward.update(performed=True, passed=bool(prior.rank_ic.mean() > 0
                and recent.rank_ic.mean() > 0), months=6,
                prior_mean_ic=float(prior.rank_ic.mean()), recent_mean_ic=float(recent.rank_ic.mean()),
                prior_last_settlement=str(prior.settlement_date.max().date()),
                recent_first_decision=str(first_recent_review.date()),
                reason="outcome_windows_disjoint_between_forward_blocks")
    checks.append(forward)
    name = "within_peer_shuffled_signal"
    null = {"name": name, "description": EXPANDED_ROBUSTNESS_CHECKS[name],
        "performed": False, "passed": None, "months": 0,
        "reason": "missing_or_insufficient_settled_shuffle_controls"}
    controls = [f"shuffle_ic_{i:02d}" for i in range(SHUFFLED_CONTROLS)]
    if set(controls) <= set(base):
        common = base.loc[np.isfinite(base[controls]).all(axis=1)]
        if len(common) >= max(2, minimum):
            observed = float(common.rank_ic.mean())
            shuffled = common[controls].mean().to_numpy()
            percentile = float(np.quantile(shuffled, 0.95))
            null.update(performed=True, passed=bool(observed > 0 and observed > percentile),
                months=len(common), mean_ic=observed, null_95_percentile=percentile,
                null_mean_ic=float(shuffled.mean()), permutations=SHUFFLED_CONTROLS,
                empirical_upper_tail=float((1 + np.sum(shuffled >= observed)) / (1 + len(shuffled))),
                reason="conditional_within_peer_control_not_search_multiplicity_correction")
    checks.append(null)
    return checks


def predictive_score(trial, asof, minimum=12):
    """Apply S=.40P+.25C+.20R+.15B to settled predictive observations only."""
    strategy = trial["strategy"]
    suite = trial.get("robustness_suite", "basic")
    if suite not in {"basic", "expanded"}:
        raise ValueError("robustness_suite must be basic or expanded")
    horizon = declared_horizon(strategy)
    observations = trial.get("predictive_monthly")
    empty = {"method": "predictive_score_v1" if suite == "basic" else "predictive_score_v2",
        "robustness_suite": suite, "score": None, "rank_eligible": False,
        "components": None, "weights": SCORE_WEIGHTS, "months": 0,
        "reason": "missing_predictive_observations"}
    if observations is None or observations.empty or asof is None:
        return empty
    asof = pd.Timestamp(asof)
    start = asof - pd.DateOffset(months=strategy.evidence_lookback_months)
    observations = observations.copy()
    observations["settlement_date"] = pd.to_datetime(observations.settlement_date)
    window = observations.loc[(observations.settlement_date > start)
        & (observations.settlement_date <= asof) & np.isfinite(observations.rank_ic)]
    base = window.loc[window.variant == "base"].sort_values("formation_month")
    stats = _ic_statistics(base.rank_ic, horizon)
    recent = base.loc[base.settlement_date > asof - pd.DateOffset(months=12)]
    if len(base) < max(2, minimum) or recent.empty or stats["t_stat"] is None:
        return {**empty, "months": len(base), "ic": stats, "asof": str(asof.date()),
            "reason": "insufficient_settled_predictive_history"}
    # Basic is intentionally the original four-check formula for controlled
    # comparisons; expanded adds checks without reinterpreting the weights.
    checks = [_variant_check(name, description, base, window, minimum)
        for name, description in ROBUSTNESS_CHECKS.items()]
    if suite == "expanded":
        checks.extend(_expanded_checks(base, window, minimum))
    performed = [check for check in checks if check["performed"]]
    components = {
        "P": float(np.clip(stats["t_stat"] / 3, 0, 1)),
        "C": float((base.rank_ic > 0).mean()),
        "R": float(np.clip(0.5 + recent.rank_ic.mean() / (2 * 0.05), 0, 1)),
        "B": float(sum(check["passed"] for check in performed) / len(performed)) if performed else 0.0,
    }
    score = float(sum(SCORE_WEIGHTS[key] * value for key, value in components.items()))
    return {**empty, "score": score,
        "rank_eligible": score > 0, "components": components, "weights": SCORE_WEIGHTS,
        "months": len(base), "ic": stats, "recent_months": len(recent),
        "recent_mean_ic": float(recent.rank_ic.mean()), "robustness_checks": checks,
        "robustness_checks_performed": len(performed), "robustness_unknown_scores_zero": not performed,
        "asof": str(asof.date()), "horizon_sessions": horizon,
        "latest_settlement": str(base.settlement_date.max().date()),
        "reason": "rank_eligible_by_predictive_score" if score > 0 else "zero_predictive_score",
        "uncertainty_scope": "monthly_HAC_not_multiple_search_corrected",
    }
