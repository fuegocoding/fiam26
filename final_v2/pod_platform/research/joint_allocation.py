"""Continuous funding on aggregate stock orders and correlated return evidence."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def capital_bound(target, carried_dollars, frame, settings, budget):
    target, carried = (
        np.asarray(target, float),
        np.asarray(carried_dollars, float),
    )
    adv = frame.dolvol_126d.fillna(0).to_numpy(float)
    cap = frame.me.fillna(0).to_numpy(float) * 1e6
    allowance = (
        adv
        * settings["max_daily_participation"]
        * settings["order_ttl_sessions"]
    )
    # Exits may continue beyond order expiry; no new exposure can use that exception.
    active = np.abs(target) > 1e-10
    high = float(budget)
    if active.any():
        same_side_dollars = np.maximum(
            carried[active] * np.sign(target[active]), 0
        )
        upper = (same_side_dollars + allowance[active]) / np.abs(target[active])
        high = min(high, float(upper.min()))
    short = target < -1e-10
    if short.any():
        inventory = np.minimum(
            cap * settings["max_borrow_fraction_market_cap"],
            adv * settings["max_borrow_adv_days"],
        )
        inventory[cap < 500e6] = 0
        high = min(high, float(np.min(inventory[short] / -target[short])))
    return max(0.0, high)


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
    recurring_impacts=None,
):
    """Maximize expected net dollars under shared ADV, borrow and pod budgets.

    Forecast evidence never sees capital. This construction step uses current
    aggregate incremental stock orders, conservative impact and prior covariance.
    All-cash is feasible; no constraint forces a losing sleeve to be funded.
    """
    edges, caps, weights, carried = (
        np.asarray(x, float) for x in (edges, caps, targets, carried_dollars)
    )
    if len(edges) == 0:
        return np.zeros(0), {"status": "no_positive_unit_edge"}
    adv = frame.dolvol_126d.fillna(0).to_numpy(float)
    cap = frame.me.fillna(0).to_numpy(float) * 1e6
    sigma = frame.rvol_21d.fillna(0.04).clip(0.005, 0.3).to_numpy(float)
    coefficients = np.divide(
        settings["impact_eta"] * sigma,
        np.sqrt(adv),
        out=np.zeros(len(adv)),
        where=adv > 0,
    )
    liquidity = (
        adv
        * settings["max_daily_participation"]
        * settings["order_ttl_sessions"]
        / nav
    )
    positive_limit = np.maximum(carried, 0) / nav + liquidity
    negative_limit = np.maximum(-carried, 0) / nav + liquidity
    recurring = (
        np.zeros(len(edges))
        if recurring_impacts is None
        else np.asarray(recurring_impacts, float)
    )
    if (
        recurring.shape != edges.shape
        or not np.isfinite(recurring).all()
        or np.any(recurring < 0)
    ):
        raise ValueError(
            "Nonnegative matching prior recurring impact coefficients are required"
        )
    inventory = np.minimum(
        cap * settings["max_borrow_fraction_market_cap"],
        adv * settings["max_borrow_adv_days"],
    )
    inventory[cap < 500e6] = 0
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
        order = nav * (u @ weights) - carried
        immediate = float(coefficients @ np.abs(order) ** 1.5) / nav
        recurrent = (
            np.sqrt(len(edges)) * float(recurring @ (nav * u) ** 1.5) / nav
        )
        risk = 0.5 * risk_aversion * float(u @ covariance @ u)
        return float(u @ edges), immediate, recurrent, risk

    def objective(u):
        gain, immediate, recurrent, risk = components(u)
        return -gain + immediate + recurrent + risk

    def constraints(u):
        target = u @ weights
        return np.r_[
            1 - u.sum(),
            [settings["max_pod_capital"] - group @ u for group in groups],
            positive_limit - target,
            negative_limit + target,
            target + inventory / nav,
            settings["single_name_limit"] - target,
            settings["single_name_limit"] + target,
            settings["gross_limit"] - np.abs(target).sum(),
            0.5 - target.sum(),
            0.5 + target.sum(),
        ]

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=list(zip(np.zeros(len(edges)), upper)),
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": 120, "ftol": 1e-10},
    )
    valid = result.success and np.min(constraints(result.x)) >= -1e-7
    u = np.maximum(result.x, 0) if valid else initial
    gain, immediate, recurrent, risk = components(u)
    impact = immediate + recurrent
    return u * nav, {
        "status": "solved" if valid else "unfunded_optimizer_failure",
        "solver_message": str(result.message),
        "expected_net_fixed_friction_dollars": gain * nav,
        "one_time_current_order_impact_dollars": immediate * nav,
        "prior_estimated_annual_recurring_impact_dollars": recurrent * nav,
        "recurring_impact_policy": "Prior native unit-order coefficients; sqrt(number_of_sleeves) conservative shared-direction bound. Current order cost is charged once over the one-year funding horizon.",
        "capacity_scope": "Directional risk-increase bounds; forced reductions can take longer than order TTL. Actual staged fills remain authoritative.",
        "risk_penalty_dollars": risk * nav,
        "expected_net_trading_dollars": (gain - impact) * nav,
        "competition_hurdle_dollars": 0.04 * nav,
        "expected_surplus_over_TB_plus_four_dollars": (gain - impact - 0.04)
        * nav,
        "covariance_annualized": covariance.tolist(),
        "prior_recurring_impact_coefficients": recurring.tolist(),
    }
