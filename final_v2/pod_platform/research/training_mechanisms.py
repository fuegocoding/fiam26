"""Small, fixed-direction research baskets to reduce selection variance."""

from __future__ import annotations

import numpy as np

from .training_data import market_return_series
from .training_search import (
    candidate,
    make_weights,
    objective,
    risk_designs,
    smooth_signal,
    summarize,
)
from .training_tactical import monthly_net, panel_schedule, replay_daily

MECHANISMS = {
    "fundamentals": {
        "earnings_information": {"niq_su": 1, "saleq_su": 1, "niq_at_chg1": 1},
        "cash_profitability": {"cop_at": 1, "ocf_at": 1, "gp_at": 1},
        "quality_resilience": {
            "qmj_prof": 1,
            "earnings_variability": -1,
            "ni_ivol": -1,
            "cash_at": 1,
        },
        "improving_operations": {
            "sale_gr1": 1,
            "saleq_gr1": 1,
            "ocf_at_chg1": 1,
        },
    },
    "events": {
        "net_repurchases": {"eqnetis_at": -1, "chcsho_12m": -1, "eqnpo_12m": 1},
        "cash_earnings_quality": {"oaccruals_at": -1, "taccruals_at": -1},
        "investment_discipline": {
            "at_gr1": -1,
            "inv_gr1a": -1,
            "capex_abn": -1,
        },
        "financing_discipline": {
            "netis_at": -1,
            "dbnetis_at": -1,
            "eqnetis_at": -1,
        },
    },
    "valuation": {
        "cash_earnings_yield": {"ocf_me": 1, "fcf_me": 1, "ni_me": 1},
        "enterprise_yield": {"ebitda_mev": 1, "ebit_bev": 1, "sale_bev": 1},
        "asset_value": {"be_me": 1, "at_me": 1, "bev_mev": 1},
        "quality_at_value": {
            "ocf_me": 1,
            "ni_me": 1,
            "qmj_prof": 1,
            "cash_at": 1,
        },
    },
    "dynamics": {
        "medium_term_trend": {"ret_6_1": 1},
        "persistent_trend": {"ret_12_1": 1, "resff3_12_1": 1, "ret_6_1": 1},
        "defensive_risk": {
            "ivol_capm_252d": -1,
            "qmj_safety": 1,
            "rvol_21d": -1,
        },
        "trend_after_reversal": {"ret_12_1": 1, "ret_1_0": -1},
        "liquidity_stability": {
            "dolvol_var_126d": -1,
            "turnover_126d": -1,
            "bidaskhl_21d": -1,
        },
    },
}


def search_mechanisms(panel, market, settings):
    """Register directions before scoring, keeping parameter count small."""
    designs = risk_designs(panel, settings)
    trials = []
    for pod, mechanisms in MECHANISMS.items():
        for name, features in mechanisms.items():
            values = np.asarray(
                [
                    panel.features[feature] * sign
                    for feature, sign in features.items()
                ]
            )
            count = np.isfinite(values).sum(axis=0)
            signal = np.divide(
                np.nansum(values, axis=0),
                count,
                out=np.full(values.shape[1:], np.nan),
                where=count >= max(1, len(features) // 2),
            )
            for smoothing in settings["mechanism_smoothing_months"]:
                weights = make_weights(
                    smooth_signal(signal, smoothing), designs, settings
                )
                for frequency in settings["mechanism_review_months"]:
                    recipe = {
                        "mechanism": name,
                        "inputs": features,
                        "smoothing_months": smoothing,
                        "rebalance_months": frequency,
                        "scaling": "cross_sectional_rank",
                        "direction": "fixed_by_economic_hypothesis",
                    }
                    trial = candidate(
                        f"{pod}:mechanism_{name}:smooth{smoothing}:review{frequency}m",
                        pod,
                        recipe,
                        weights,
                        panel,
                        settings,
                    )
                    schedule = panel_schedule(weights, panel, market, frequency)
                    book = replay_daily(market, schedule, settings)
                    trial.net = monthly_net(book["net"], market, panel)
                    # Rescore using the same daily engine as faster strategies.
                    trial.metrics = summarize(
                        trial.net, market_return_series(panel)
                    )
                    trial.score = objective(trial.metrics)
                    trials.append(trial)
    return trials


def selection_universe(candidates, settings):
    """Keep the broad battery diagnostic when selecting a smaller rule family."""
    if settings.get("selection_family") == "mechanisms":
        return [
            c
            for c in candidates
            if "mechanism" in c.recipe
            or "model" in c.recipe
            or ("family" in c.recipe and c.recipe["expected_return_sign"] == 1)
        ]
    return candidates
