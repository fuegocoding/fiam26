"""Shrunk pod return models fitted only on settled, prior observations."""

from __future__ import annotations

import numpy as np

from .training_data import POD_FEATURES, market_return_series
from .training_search import (
    candidate,
    make_weights,
    objective,
    risk_designs,
    summarize,
)
from .training_tactical import monthly_net, panel_schedule, replay_daily


def rolling_ridge(
    panel, features, ridge, lookback, min_history, *, settings=None
):
    """Fit equally weighted monthly cross sections with explicit label maturity.

    Sufficient statistics retain only formation-neutralized ranks and actual
    realized returns. A formation cannot use a label settling after that date.
    Ridge penalties shrink correlated factors rather than selecting their signs
    individually. A final December 2020 fit may use returns settled that day.
    """
    designs = risk_designs(panel, settings)
    count, names = panel.mask.shape
    dimension = len(features)
    design_cube = np.stack(
        [np.nan_to_num(panel.features[f]) for f in features], axis=2
    )
    covariance = np.zeros((len(panel.returns), dimension, dimension))
    cross = np.zeros((len(panel.returns), dimension))
    for t, (mask, control, inverse) in enumerate(designs):
        values = design_cube[t, mask]
        design_cube[t, mask] = values - control @ (inverse @ values)
        if t >= len(panel.returns):
            continue
        valid = mask & panel.entry_mask[t]
        values = design_cube[t, valid]
        outcomes = panel.returns[t, valid]
        covariance[t] = values.T @ values / max(1, len(values))
        cross[t] = values.T @ outcomes / max(1, len(values))
    predictions = np.zeros((count, names))
    coefficients = []
    for t, date in enumerate(panel.dates):
        known = np.flatnonzero(panel.settlement_dates <= date)[-lookback:]
        if len(known) < min_history:
            coefficients.append(np.zeros(dimension).tolist())
            continue
        matrix = covariance[known].mean(axis=0) + np.eye(dimension) * ridge
        beta = np.linalg.solve(matrix, cross[known].mean(axis=0))
        predictions[t] = design_cube[t] @ beta
        coefficients.append(beta.tolist())
    return predictions, coefficients


def search_models(panel, market, settings, progress=None):
    """Run a small regularization grid; model fitting remains chronological."""
    result = []
    designs = risk_designs(panel, settings)
    for pod, features in POD_FEATURES.items():
        for ridge in settings["ridge_penalties"]:
            for lookback in settings["ridge_lookbacks"]:
                prediction, coefficients = rolling_ridge(
                    panel,
                    features,
                    ridge,
                    lookback,
                    settings["ridge_min_history"],
                    settings=settings,
                )
                weights = make_weights(prediction, designs, settings)
                for frequency in settings["model_review_months"]:
                    recipe = {
                        "model": "rolling_ridge",
                        "features": features,
                        "ridge_penalty": ridge,
                        "lookback_months": lookback,
                        "min_settled_months": settings["ridge_min_history"],
                        "rebalance_months": frequency,
                        "final_coefficients": dict(
                            zip(features, coefficients[-1])
                        ),
                        "scaling": "formation_ranks_residualized_against_configured_risk_controls",
                    }
                    trial = candidate(
                        f"{pod}:ridge{ridge}:lookback{lookback}:review{frequency}m",
                        pod,
                        recipe,
                        weights,
                        panel,
                        settings,
                    )
                    book = replay_daily(
                        market,
                        panel_schedule(weights, panel, market, frequency),
                        settings,
                    )
                    trial.net = monthly_net(book["net"], market, panel)
                    trial.metrics = summarize(
                        trial.net, market_return_series(panel)
                    )
                    trial.score = objective(trial.metrics)
                    result.append(trial)
        if progress:
            progress(pod + "_ridge", len(result))
    return result
