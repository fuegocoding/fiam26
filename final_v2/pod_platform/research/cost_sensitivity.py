"""Predeclared frozen-instruction sensitivities, without signal retuning."""

from __future__ import annotations

import numpy as np

from .edge_capacity import measure_edge
from .training_tactical import monthly_net

CASES = (
    (
        "competition_conservative_borrow_50_500bp",
        {"large_cap_borrow_bps": 50, "other_borrow_bps": 500},
    ),
    ("half_proxy_same_borrow", {"spread_proxy_scale": 0.5}),
    ("one_and_half_proxy_same_borrow", {"spread_proxy_scale": 1.5}),
    (
        "ten_bp_floor_only_same_borrow",
        {"spread_proxy_scale": 0.0, "cost_bps_one_way": 10},
    ),
    (
        "proxy_borrow_50bp_all",
        {"large_cap_borrow_bps": 50, "other_borrow_bps": 50},
    ),
    (
        "proxy_borrow_25_100bp",
        {"large_cap_borrow_bps": 25, "other_borrow_bps": 100},
    ),
    (
        "ten_bp_floor_borrow_25_100bp",
        {
            "spread_proxy_scale": 0.0,
            "cost_bps_one_way": 10,
            "large_cap_borrow_bps": 25,
            "other_borrow_bps": 100,
        },
    ),
    (
        "frictionless_reference",
        {
            "spread_proxy_scale": 0.0,
            "cost_bps_one_way": 0,
            "large_cap_borrow_bps": 0,
            "other_borrow_bps": 0,
        },
    ),
)


def frozen_cost_sensitivity(market, schedule, panel, settings, baseline):
    """Same forecast, support, review dates and target weights in every case.

    NAV compounds separately under each cost assumption. These are illustrative
    scenarios, not fitted quotes or an authorization to select optimistic fees.
    No dollar capital or market impact is included in this zero-size unit test.
    """

    def summarize(name, overrides, result):
        monthly = monthly_net(result["net"], market, panel)
        sigma = float(np.std(monthly, ddof=1))
        active = market.dates >= panel.execution_dates[0]
        return {
            "scenario": name,
            "spread_proxy_scale": overrides.get(
                "spread_proxy_scale", settings.get("spread_proxy_scale", 1.0)
            ),
            "cost_floor_bps_one_way": overrides.get(
                "cost_bps_one_way", settings["cost_bps_one_way"]
            ),
            "large_cap_borrow_bps": overrides.get(
                "large_cap_borrow_bps", settings["large_cap_borrow_bps"]
            ),
            "other_borrow_bps": overrides.get(
                "other_borrow_bps", settings["other_borrow_bps"]
            ),
            "cagr": float(np.expm1(np.log1p(monthly).mean() * 12)),
            "annualized_arithmetic_return_monthly": float(monthly.mean() * 12),
            "sharpe_monthly": float(monthly.mean() / sigma * np.sqrt(12))
            if sigma > 1e-12
            else None,
            "daily_annualized_borrow_drag": float(
                np.mean(result["borrow"][active]) * 252
            ),
            "daily_annualized_trading_drag": float(
                np.mean(result["trading_drag"][active]) * 252
            ),
            "capital_assumption": None,
            "includes_market_impact": False,
            "calibration_status": "unfitted_parameter_sensitivity",
        }

    rows = [summarize("configured_reference", {}, baseline)]
    for name, overrides in CASES:
        result = measure_edge(
            market, schedule, {**settings, **overrides}, keep_holdings=False
        )
        rows.append(summarize(name, overrides, result))
    return rows
