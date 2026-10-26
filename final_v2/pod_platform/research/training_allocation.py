"""Capital allocation from settled prior performance within and across pods."""

from __future__ import annotations

import numpy as np

from .training_data import market_return_series
from .training_search import measure_weights, objective, summarize
from .training_tactical import combined_schedule, monthly_net, replay_daily


def capped_simplex(values, cap):
    """Project onto nonnegative weights summing to one with a per-pod cap."""
    values = np.asarray(values, dtype=float)
    if cap <= 0 or cap > 1 or len(values) * cap < 1 - 1e-10:
        raise ValueError("Infeasible capital cap")
    if (
        (values >= 0).all()
        and values.max() <= cap
        and abs(values.sum() - 1) < 1e-10
    ):
        return values.copy()
    low, high = values.min() - 1, values.max()
    for _ in range(35):
        mid = (low + high) / 2
        if np.clip(values - mid, 0, cap).sum() > 1:
            low = mid
        else:
            high = mid
    return np.clip(values - (low + high) / 2, 0, cap)


def capital_weights(history, method, cap, risk_aversion):
    """Use shrunk moments and bounded nonnegative capital, without leverage."""
    count = history.shape[1]
    if len(history) < 3 or method == "equal":
        return capped_simplex(np.full(count, 1 / count), cap)
    volatility = np.maximum(history.std(axis=0, ddof=1), 1e-4)
    if method == "inverse_volatility":
        raw = 1 / volatility
        return capped_simplex(raw / raw.sum(), cap)
    mean = history.mean(axis=0) * len(history) / (len(history) + 12)
    if method == "performance":
        raw = np.maximum(mean, 0) / volatility**2
        if raw.sum() == 0:
            raw = 1 / volatility
        return capped_simplex(raw / raw.sum(), cap)
    if method != "mean_variance":
        raise ValueError("Unknown capital allocation method")
    covariance = np.atleast_2d(np.cov(history, rowvar=False))
    covariance = 0.5 * covariance + 0.5 * np.diag(np.diag(covariance))
    covariance += np.eye(count) * 1e-6
    step = 1 / max(risk_aversion * np.linalg.eigvalsh(covariance).max(), 1e-6)
    weights = np.full(count, 1 / count)
    for _ in range(40):
        next_weights = capped_simplex(
            weights + step * (mean - risk_aversion * covariance @ weights), cap
        )
        if np.max(np.abs(next_weights - weights)) < 1e-8:
            break
        weights = next_weights
    return next_weights


def allocate(selected, panel, settings, policy):
    """Allocate at formation using returns settled by that formation date."""
    pods = [name for name, strategies in selected.items() if strategies]
    if len(pods) < 1:
        values = np.zeros(len(panel.returns))
        metrics = summarize(values)
        return {
            "policy": policy,
            "metrics": metrics,
            "score": objective(metrics),
            "weights": np.zeros(panel.mask.shape),
            "returns": values,
            "turnover": values.copy(),
            "allocations": [],
        }
    combined = np.zeros_like(next(iter(selected[pods[0]])).weights)
    records = []
    for t, date in enumerate(panel.dates):
        known = np.flatnonzero(panel.settlement_dates <= date)
        known = known[-policy["lookback_months"] :]
        active = len(known) >= settings["allocation_min_history"]
        pod_histories, pod_targets, within_weights = [], [], []
        for pod in pods:
            strategies = selected[pod]
            returns = np.column_stack([s.net for s in strategies])
            method = policy["method"] if active else "equal"
            within = capital_weights(
                returns[known], method, 1.0, policy["risk_aversion"]
            )
            if active and settings.get("capital_profitability_gate", False):
                recent = returns[known][-settings["capital_signal_months"] :]
                within *= recent.mean(axis=0) > 0
                if within.sum() > 0:
                    within /= within.sum()
            within_weights.append(within)
            pod_histories.append(returns[known] @ within)
            pod_targets.append(
                sum(
                    weight * strategy.weights[t]
                    for weight, strategy in zip(within, strategies)
                )
            )
        history = np.column_stack(pod_histories)
        cap = max(settings["max_pod_capital"], 1 / len(pods))
        outer = capital_weights(
            history,
            policy["method"] if active else "equal",
            cap,
            policy["risk_aversion"],
        )
        if active and settings.get("capital_profitability_gate", False):
            recent = history[-settings["capital_signal_months"] :]
            outer *= recent.mean(axis=0) > 0
        combined[t] = sum(w * target for w, target in zip(outer, pod_targets))
        for pod, pod_capital, within in zip(pods, outer, within_weights):
            for strategy, strategy_capital in zip(selected[pod], within):
                records.append(
                    {
                        "formation_date": str(date.date()),
                        "pod": pod,
                        "strategy_id": strategy.strategy_id,
                        "pod_capital_weight": float(pod_capital),
                        "within_pod_weight": float(strategy_capital),
                        "capital_weight": float(pod_capital * strategy_capital),
                        "settled_history_months": len(known),
                        "latest_settled_return": (
                            str(panel.settlement_dates[known[-1]].date())
                            if len(known)
                            else None
                        ),
                    }
                )
    net, turnover = measure_weights(combined, panel, settings)
    metrics = summarize(net, market_return_series(panel))
    return {
        "policy": policy,
        "metrics": metrics,
        "score": objective(metrics),
        "weights": combined,
        "returns": net,
        "turnover": turnover,
        "allocations": records,
    }


def allocation_search(
    selected, panel, settings, progress=None, selection_mask=None, market=None
):
    """Compare explicit allocation policies on the authorized training period."""
    trials = []
    cache = {}
    for method in settings["allocation_methods"]:
        for lookback in settings["allocation_lookbacks"]:
            risk_values = (
                settings["risk_aversions"] if method == "mean_variance" else [5]
            )
            for risk in risk_values:
                trial = allocate(
                    selected,
                    panel,
                    settings,
                    {
                        "method": method,
                        "lookback_months": lookback,
                        "risk_aversion": risk,
                    },
                )
                if market is not None and settings.get(
                    "capacity_execution", False
                ):
                    schedule = combined_schedule(
                        selected,
                        trial["allocations"],
                        panel,
                        market,
                        settings,
                        cache,
                    )
                    book = replay_daily(
                        market,
                        schedule,
                        {
                            **settings,
                            "execution_start_nav": settings["fund_capital_usd"],
                        },
                    )
                    trial["returns"] = monthly_net(book["net"], market, panel)
                    trial["metrics"] = summarize(
                        trial["returns"], market_return_series(panel)
                    )
                    trial["score"] = objective(trial["metrics"])
                if selection_mask is not None:
                    trial["score"] = objective(
                        summarize(trial["returns"][selection_mask])
                    )
                trials.append(trial)
        if progress:
            progress(method, len(trials))
    return max(trials, key=lambda trial: trial["score"]), trials
