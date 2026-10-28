"""Continuous funding on aggregate stock orders and correlated return evidence."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .mandate import GICS, frame_sectors


def allocate_jointly(
    edges,
    caps,
    pods,
    targets,
    carried_dollars,
    frame,
    settings,
    nav,
    covariance,
    *,
    risk_aversion=2.0,
):
    """Maximize expected dollars under portfolio risk and pod limits.

    Forecast evidence never sees capital. This construction step uses current
    aggregate portfolio positions and prior covariance.
    All-cash is feasible; no constraint forces a losing sleeve to be funded.
    """
    edges, caps, weights, carried = (
        np.asarray(x, float) for x in (edges, caps, targets, carried_dollars)
    )
    if len(edges) == 0:
        return np.zeros(0), {"status": "no_positive_unit_edge"}
    cap = frame.me.fillna(0).to_numpy(float) * 1e6
    sectors = frame_sectors(frame, weights.shape[1])
    sector_targets = np.array([weights[:, sectors == code].sum(axis=1) for code in (*GICS, 'unknown')])
    covariance = np.asarray(covariance, float)
    covariance = (covariance + covariance.T) / 2
    if (
        covariance.shape != (len(edges), len(edges))
        or not np.isfinite(covariance).all()
    ):
        raise ValueError("Finite matching prior covariance is required")
    # Empirical covariance is PSD; a numerical eigenvalue floor prevents a
    # tiny rounding error from rewarding unlimited correlated risk.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    groups = [
        np.array([p == pod for p in pods], float) for pod in sorted(set(pods))
    ]
    upper = np.minimum(caps / nav, settings["max_pod_capital"])
    initial = np.zeros(len(edges))

    def components(u):
        risk = 0.5 * risk_aversion * float(u @ covariance @ u)
        return float(u @ edges), risk

    def objective(u):
        gain, risk = components(u)
        return -gain + risk

    def gradient(u):
        return -edges + risk_aversion * covariance @ u

    def constraints(u):
        target = u @ weights
        return np.r_[
            1 - u.sum(),
            [settings["max_pod_capital"] - group @ u for group in groups],
            # Numerical feasibility tolerance, not an economic short allowance.
            target[cap < 500e6] + 1e-12,
            settings["single_name_limit"] - target,
            settings["single_name_limit"] + target,
            settings["gross_limit"] - np.abs(target).sum(),
            0.5 - target.sum(),
            0.5 + target.sum(),
            settings.get('sector_net_limit', .2) - sector_targets @ u,
            settings.get('sector_net_limit', .2) + sector_targets @ u,
            # Unknown classifications cannot silently provide sector breadth.
            1e-12 - np.abs(target[sectors == 'unknown']),
        ]

    # A feasible positive-edge direction provides an objective check against
    # cash. Tiny signed projection residuals must not pin the solver to zero.
    best_objective = objective(initial)
    for i, edge in enumerate(edges):
        if edge <= 0 or np.any(weights[i, cap < 500e6] < -1e-12):
            continue
        vector = weights[i]
        limit = min(upper[i], 1.0)
        limit = min(limit, settings["gross_limit"] / max(np.abs(vector).sum(), 1e-12),
                    settings["single_name_limit"] / max(np.abs(vector).max(), 1e-12),
                    0.5 / max(abs(vector.sum()), 1e-12))
        optimum = edge / (risk_aversion * covariance[i, i]) if risk_aversion * covariance[i, i] > 0 else limit
        candidate = np.zeros(len(edges))
        candidate[i] = min(limit, optimum)
        candidate_objective = objective(candidate)
        if np.min(constraints(candidate)) >= -1e-10 and candidate_objective < best_objective:
            initial, best_objective = candidate, candidate_objective

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=list(zip(np.zeros(len(edges)), upper)),
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": 120, "ftol": 1e-10},
    )
    valid = result.success and np.min(constraints(result.x)) >= -1e-7
    improves_start = valid and objective(result.x) <= best_objective + 1e-12
    u = np.maximum(result.x, 0) if improves_start else initial
    gain, risk = components(u)
    return u * nav, {
        "status": "solved" if improves_start else (
            "funded_feasible_fallback" if np.any(initial > 0) else "unfunded_optimizer_failure"
        ),
        "solver_message": str(result.message),
        "expected_dollars": gain * nav,
        "allocation_scope": "Fund, pod and portfolio risk limits; all trades fill in full at observed prices.",
        "risk_penalty_dollars": risk * nav,
        "objective_improvement_over_cash_dollars": (gain - risk) * nav,
        "competition_hurdle_dollars": 0.04 * nav,
        "expected_surplus_over_TB_plus_four_dollars": (gain - 0.04) * nav,
        "covariance_annualized": covariance.tolist(),
    }
