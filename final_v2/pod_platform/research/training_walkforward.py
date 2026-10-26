"""Annual frozen selections inside development, followed by exact daily replay."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .desk_metrics import compute_desk_metrics
from .training_allocation import allocation_search
from .training_mechanisms import selection_universe
from .training_search import (
    candidate,
    make_weights,
    objective,
    risk_designs,
    select_strategies,
    summarize,
)
from .training_tactical import (
    combined_schedule,
    monthly_net,
    panel_schedule,
    replay_daily,
)


def walk_forward(candidates, panel, market, settings, progress=None):
    """Tune on settled prior data, then freeze strategy IDs for the next year.

    All development years can inform engineering; these are chronological
    diagnostics within training, not claims of an untouched final holdout.
    Ensembles chosen on all training results are excluded from fold selection.
    """
    lookup = {c.strategy_id: c for c in candidates}
    frozen = {}
    pod_frozen = {
        pod: {} for pod in ("fundamentals", "events", "valuation", "dynamics")
    }
    folds = []
    cache = {}
    designs = risk_designs(panel, settings)
    for year in settings["walk_forward_years"]:
        cutoff = pd.Timestamp(year - 1, 12, 31)
        train_mask = panel.settlement_dates <= cutoff
        if train_mask.sum() < 24:
            raise ValueError(
                "Walk-forward selection needs 24 settled training months"
            )
        prior = []
        for trial in selection_universe(candidates, settings):
            if "components" in trial.recipe:
                continue
            metrics = summarize(trial.net[train_mask])
            prior.append(
                replace(
                    trial,
                    net=trial.net[train_mask],
                    metrics=metrics,
                    score=objective(metrics),
                )
            )
        if settings.get("selection_family") != "mechanisms":
            for pod in ("fundamentals", "events", "valuation", "dynamics"):
                ranked = sorted(
                    [
                        s
                        for s in prior
                        if s.pod == pod and "feature" in s.recipe
                    ],
                    key=lambda s: s.score,
                    reverse=True,
                )
                unique, seen = [], set()
                for trial in ranked:
                    if trial.recipe["feature"] not in seen:
                        unique.append(trial)
                        seen.add(trial.recipe["feature"])
                for size in settings["ensemble_sizes"]:
                    components = unique[:size]
                    if len(components) < size:
                        continue
                    weights = make_weights(
                        np.mean([s.weights for s in components], axis=0),
                        designs,
                        settings,
                        construction_scaling=False,
                    )
                    full = candidate(
                        f"{pod}:frozen_{year}_ensemble{size}",
                        pod,
                        {
                            "components": [s.strategy_id for s in components],
                            "selected_asof": str(cutoff.date()),
                        },
                        weights,
                        panel,
                        settings,
                    )
                    lookup[full.strategy_id] = full
                    if settings.get("capacity_execution", False):
                        measured = replay_daily(
                            market,
                            panel_schedule(weights, panel, market),
                            settings,
                        )
                        full.net = monthly_net(measured["net"], market, panel)
                        full.metrics = summarize(full.net)
                        full.score = objective(full.metrics)
                    metrics = summarize(full.net[train_mask])
                    prior.append(
                        replace(
                            full,
                            net=full.net[train_mask],
                            metrics=metrics,
                            score=objective(metrics),
                        )
                    )
        chosen = select_strategies(prior, settings)
        chosen = {
            pod: [lookup[s.strategy_id] for s in strategies]
            for pod, strategies in chosen.items()
        }
        allocation, trials = allocation_search(
            chosen, panel, settings, selection_mask=train_mask, market=market
        )
        schedule = combined_schedule(
            chosen, allocation["allocations"], panel, market, settings, cache
        )
        frozen.update(
            {
                d: target
                for d, target in schedule.items()
                if market.dates[d].year == year
            }
        )
        if not any(chosen.values()):
            first_session = int(
                market.dates.searchsorted(pd.Timestamp(year, 1, 1))
            )
            frozen[first_session] = np.zeros(len(panel.securities))
        for pod, strategies in chosen.items():
            if not strategies:
                # A dropped pod must liquidate its prior year's standalone
                # diagnostic holdings, just as the combined CIO deallocates it.
                first_session = int(
                    market.dates.searchsorted(pd.Timestamp(year, 1, 1))
                )
                pod_frozen[pod][first_session] = np.zeros(len(panel.securities))
                continue
            pod_allocations = [
                {**r, "capital_weight": r["within_pod_weight"]}
                for r in allocation["allocations"]
                if r["pod"] == pod
            ]
            pod_schedule = combined_schedule(
                {pod: strategies},
                pod_allocations,
                panel,
                market,
                settings,
                cache,
            )
            pod_frozen[pod].update(
                {
                    d: target
                    for d, target in pod_schedule.items()
                    if market.dates[d].year == year
                }
            )
        folds.append(
            {
                "test_year": year,
                "selection_asof": str(cutoff.date()),
                "settled_training_months": int(train_mask.sum()),
                "latest_training_return": str(
                    panel.settlement_dates[train_mask][-1].date()
                ),
                "selected": {
                    p: [s.strategy_id for s in ss] for p, ss in chosen.items()
                },
                "selected_recipes": {
                    p: [
                        {
                            "strategy_id": s.strategy_id,
                            "recipe": {
                                key: value
                                for key, value in s.recipe.items()
                                if key
                                not in {
                                    "final_coefficients",
                                    "model_artifact",
                                    "final_fit_asof",
                                }
                            },
                        }
                        for s in ss
                    ]
                    for p, ss in chosen.items()
                },
                "allocation_policy": allocation["policy"],
                "allocation_trials": len(trials),
            }
        )
        if progress:
            progress(
                f"walk_forward_{year}", sum(len(v) for v in chosen.values())
            )
        cache.clear()
    book = replay_daily(
        market,
        frozen,
        {
            **settings,
            "execution_start_nav": settings.get(
                "fund_capital_usd", 1_000_000_000
            ),
        },
        keep_holdings=True,
    )
    begin = pd.Timestamp(min(settings["walk_forward_years"]), 1, 1)
    observed = market.dates >= begin
    daily = book["net"][observed]
    dates = market.dates[observed]
    monthly = (1 + pd.Series(daily, index=dates)).resample("ME").prod() - 1
    for fold in folds:
        values = daily[dates.year == fold["test_year"]]
        fold["daily_metrics"] = compute_desk_metrics(
            values, periods_per_year=252
        )
    pod_results = {}
    for pod, schedule in pod_frozen.items():
        values = replay_daily(market, schedule, settings)["net"][observed]
        months = (1 + pd.Series(values, index=dates)).resample("ME").prod() - 1
        pod_results[pod] = {
            "monthly_metrics": summarize(months.to_numpy()),
            "daily_metrics": compute_desk_metrics(values, periods_per_year=252),
            "daily_returns": values,
        }
    return {
        "folds": folds,
        "monthly_metrics": summarize(monthly.to_numpy()),
        "daily_metrics": compute_desk_metrics(daily, periods_per_year=252),
        "daily_returns": daily,
        "dates": dates,
        "schedule": frozen,
        "book": book,
        "pods": pod_results,
        "selection_bias": "chronological_development_diagnostic_not_untouched_holdout",
    }
