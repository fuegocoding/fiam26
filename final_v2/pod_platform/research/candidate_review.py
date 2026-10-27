"""Shared evidence criteria for research feedback, ownership and funding."""

from __future__ import annotations

import numpy as np

from .edge_curve import mean_evidence


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
    """Compare declared scopes/windows on identical settled months."""
    common = np.intersect1d(candidate_history(candidate, known), candidate_history(parent, known))
    width = min(candidate["strategy"].validation_window_months, parent["strategy"].validation_window_months)
    scores = []
    for trial in (candidate, parent):
        review = review_candidate(trial, common, minimum, uncertainty_haircut)
        chronological = chronological_evidence(np.asarray(trial["monthly"])[common], width)
        edge, worst = review["expected_annual_unit_edge"], chronological["worst_annual_unit_mean"]
        scores.append(min(edge, worst) if len(common) >= minimum and edge is not None and worst is not None else None)
    return {"months": len(common), "window_months": width,
        "candidate_score": scores[0], "parent_score": scores[1]}


def review_candidate(trial, known, minimum=24, uncertainty_haircut=1.0):
    """Assess only settled unit returns; validity alone is not positive evidence."""
    eligible = candidate_history(trial, known)
    values = np.asarray(trial["monthly"], float)[eligible]
    evidence = mean_evidence(values)
    expected = None
    if evidence["hac_se"] is not None:
        expected = float((evidence["mean"] * len(values) / (len(values) + 12)
                          - uncertainty_haircut * evidence["hac_se"]) * 12)
    supported = len(values) >= minimum and expected is not None and expected > 0
    return {
        **evidence,
        "evidence_lookback_months": trial["strategy"].evidence_lookback_months,
        "chronological": chronological_evidence(values, trial["strategy"].validation_window_months),
        "expected_annual_unit_edge": expected,
        "supported": supported,
        "review_reason": "positive_uncertainty_adjusted_unit_edge" if supported else (
            "insufficient_settled_history" if len(values) < minimum
            else "nonpositive_uncertainty_adjusted_unit_edge"
        ),
    }


def supported_candidates(trials, pod, known, minimum=24, *,
                         uncertainty_haircut=1.0, maximum=3, maximum_correlation=.85):
    """Rank supported candidates and retain distinct return evidence."""
    rows = []
    for identity, trial in trials.items():
        if trial["strategy"].pod != pod:
            continue
        evidence = review_candidate(trial, known, minimum, uncertainty_haircut)
        if evidence["supported"]:
            rows.append((evidence["lower_95"], evidence["expected_annual_unit_edge"], identity))
    selected = []
    known = np.asarray(known, dtype=int)
    for _, _, identity in sorted(rows, reverse=True):
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
