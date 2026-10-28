"""Shared evidence criteria for research feedback, ownership and funding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .edge_curve import mean_evidence
from .predictive_selection import predictive_score


def candidate_history(trial, known):
    """Respect each hypothesis's declared recent history and evidence inception."""
    known = np.asarray(known, dtype=int)
    available = trial.get("evidence_available", np.ones(len(trial["monthly"]), dtype=bool))
    known = known[available[known]]
    strategy = trial["strategy"]
    return known[-strategy.evidence_lookback_months:]


def chronological_evidence(values, width):
    """Monthly rolling training diagnostics, never untouched validation."""
    values = np.asarray(values, float)
    windows = []
    for end in range(width, len(values) + 1):
        local = values[end - width:end]
        evidence = mean_evidence(local)
        windows.append({"start_offset": end - width, "end_offset": end,
            "return": float(np.prod(1 + local) - 1), **evidence})
    means = [w["mean"] * 12 for w in windows]
    return {"scope": "adaptive_training_chronological_diagnostic",
        "window_months": width, "windows": windows,
        "stride_months": 1, "evaluated_windows": len(windows),
        "positive_windows": sum(x > 0 for x in means),
        "worst_annual_unit_mean": min(means) if means else None,
        "latest_annual_unit_mean": means[-1] if means else None}


def compare_refinement(candidate, parent, known, minimum, uncertainty_haircut):
    """Compare predictive scores on common decision cohorts known at the cutoff."""
    reviews = [review_candidate(t, known, minimum, uncertainty_haircut) for t in (candidate, parent)]
    asof = reviews[0]["predictive_selection"].get("asof")
    if asof is None:
        return {"months": 0, "candidate_score": None, "parent_score": None}
    common = None
    for trial in (candidate, parent):
        data = trial.get("predictive_monthly", pd.DataFrame())
        if data.empty:
            return {"months": 0, "candidate_score": None, "parent_score": None}
        first = pd.Timestamp(asof) - pd.DateOffset(months=trial["strategy"].evidence_lookback_months)
        eligible = data.loc[(data.variant == "base") & (data.settlement_date > first)
            & (data.settlement_date <= pd.Timestamp(asof))
            & np.isfinite(data.rank_ic), "formation_month"].to_numpy()
        common = eligible if common is None else np.intersect1d(common, eligible)
    scores = []
    for trial in (candidate, parent):
        restricted = {**trial, "predictive_monthly": trial["predictive_monthly"].loc[
            trial["predictive_monthly"].formation_month.isin(common)]}
        scores.append(predictive_score(restricted, asof, minimum)["score"])
    return {"months": len(common), "method": "common_cohort_predictive_score",
        "candidate_score": scores[0], "parent_score": scores[1]}


def review_candidate(trial, known, minimum=24, uncertainty_haircut=1.0):
    """Rank predictive evidence; historical portfolio returns describe risk."""
    eligible = candidate_history(trial, known)
    values = np.asarray(trial["monthly"], float)[eligible]
    evidence = mean_evidence(values)
    expected = None
    if evidence["hac_se"] is not None:
        expected = float((evidence["mean"] * len(values) / (len(values) + 12)
                          - uncertainty_haircut * evidence["hac_se"]) * 12)
    known = np.asarray(known, int)
    dates = trial.get("outcome_dates")
    asof = pd.DatetimeIndex(dates)[known].max() if dates is not None and len(known) else None
    prediction = predictive_score(trial, asof, minimum)
    supported = prediction["rank_eligible"]
    legacy = trial.get("funding_method", "predictive") == "legacy_returns"
    return {
        **evidence,
        "evidence_lookback_months": trial["strategy"].evidence_lookback_months,
        "chronological": chronological_evidence(values, trial["strategy"].validation_window_months),
        "expected_annual_unit_edge": expected if legacy else None,
        "historical_return_diagnostic": {"uncertainty_adjusted_annual_mean": expected,
            "used_for_funding": legacy},
        "predictive_selection": prediction,
        "selection_score": prediction["score"],
        "funding_supported": bool(len(values) >= minimum and expected is not None and expected > 0) if legacy else None,
        "funding_reason": "legacy_return_evidence" if legacy else "calibrate_current_stock_targets_at_allocation",
        "supported": supported,
        "review_reason": prediction["reason"],
    }


def supported_candidates(trials, pod, known, minimum=24, *,
                         uncertainty_haircut=1.0, maximum=3, maximum_correlation=.85):
    """Rank by the four-component predictive score, retaining distinct sleeves."""
    rows = []
    for identity, trial in trials.items():
        if trial["strategy"].pod != pod:
            continue
        evidence = review_candidate(trial, known, minimum, uncertainty_haircut)
        if evidence["supported"]:
            rows.append((evidence["selection_score"], identity))
    selected = []
    known = np.asarray(known, dtype=int)
    for _, identity in sorted(rows, key=lambda row: (-row[0], row[1])):
        trial = trials[identity]
        duplicate = False
        for old in selected:
            other = trials[old]
            common = np.intersect1d(candidate_history(trial, known), candidate_history(other, known))
            for item in (trial, other):
                common = common[item.get("evidence_available", np.ones(len(item["monthly"]), dtype=bool))[common]]
            x, y = (np.asarray(item["monthly"])[common] for item in (trial, other))
            correlation = float(np.corrcoef(x, y)[0, 1]) if (
                len(common) >= minimum and np.std(x) > 1e-12 and np.std(y) > 1e-12
            ) else 1.
            if correlation >= maximum_correlation:
                duplicate = True
                break
        if not duplicate:
            selected.append(identity)
        if len(selected) >= maximum:
            break
    return selected
