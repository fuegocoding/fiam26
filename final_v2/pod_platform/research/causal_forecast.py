"""A small trusted trainer; generated strategies never receive return labels."""

from __future__ import annotations

import numpy as np

from .pod_plans import construction_designs


def fit_ridge_forecast(
    panel,
    features,
    *,
    penalty=1.0,
    lookback=36,
    minimum_history=12,
    controls="sector_beta",
):
    """Fit month-weighted sufficient statistics using settled observations only.

    One cross-section is materialized at a time, so all supplied characteristics
    can be requested without building a full stock-month-feature cube. Missing
    centered ranks use median-rank imputation inside the trusted fit; a forecast
    requires at least65% observed features. Labels are clipped by fixed monthly
    2.5/97.5 percentile rules before they enter the training sufficient statistics.
    These choices are declared research policy, not learned performance claims.
    """
    if (
        not features
        or len(features) != len(set(features))
        or not set(features) <= set(panel.features)
    ):
        raise ValueError(
            "Use distinct available research features for the model"
        )
    if penalty <= 0 or lookback < minimum_history or minimum_history < 6:
        raise ValueError(
            "Positive regularization and sufficient settled history are required"
        )
    designs = construction_designs(panel, controls)
    dimension = len(features)
    covariance = np.zeros((len(panel.returns), dimension, dimension))
    cross = np.zeros((len(panel.returns), dimension))
    training_rows = np.zeros(len(panel.returns), dtype=int)
    forecasts = np.full(panel.mask.shape, np.nan, dtype=np.float32)
    states = []
    for t, date in enumerate(panel.dates):
        local = np.column_stack(
            [panel.features[name][t] for name in features]
        ).astype(float)
        observed = np.isfinite(local).mean(axis=1)
        mask = panel.mask[t] & (observed >= 0.65)
        matrix = designs[t].matrix[mask[designs[t].mask]]
        raw_values = np.nan_to_num(local[mask], nan=0.0)
        values = raw_values.copy()
        if len(values):
            values -= matrix @ (np.linalg.pinv(matrix) @ values)
        known = np.flatnonzero(panel.settlement_dates <= date)[-lookback:]
        known = known[training_rows[known] > 0]
        if len(known) >= minimum_history:
            coefficient = np.linalg.solve(
                covariance[known].mean(axis=0) + np.eye(dimension) * penalty,
                cross[known].mean(axis=0),
            )
            forecasts[t, mask] = values @ coefficient
            states.append(
                {
                    "formation_date": str(date.date()),
                    "last_label_settlement": str(
                        panel.settlement_dates[known[-1]].date()
                    ),
                    "settled_months": len(known),
                    "training_rows": int(training_rows[known].sum()),
                    "coefficients": coefficient.tolist(),
                }
            )
        else:
            states.append(
                {
                    "formation_date": str(date.date()),
                    "settled_months": len(known),
                    "training_rows": int(training_rows[known].sum()),
                    "coefficients": None,
                }
            )
        if t < len(panel.returns):
            label = panel.returns[t, mask]
            entries = panel.entry_mask[t, mask] & np.isfinite(label)
            x = raw_values[entries]
            y = label[entries]
            fit_controls = matrix[entries]
            if len(y) >= 2 and len(y) > np.linalg.matrix_rank(fit_controls) + 1:
                y = np.clip(y, *np.quantile(y, [0.025, 0.975]))
                inverse = np.linalg.pinv(fit_controls)
                x -= fit_controls @ (inverse @ x)
                y -= fit_controls @ (inverse @ y)
                covariance[t] = x.T @ x / len(y)
                cross[t] = x.T @ y / len(y)
                training_rows[t] = len(y)
    return forecasts, {
        "model": "month_weighted_causal_ridge",
        "features": features,
        "penalty": penalty,
        "lookback_months": lookback,
        "minimum_settled_months": minimum_history,
        "controls": controls,
        "label_clip_percentiles": [2.5, 97.5],
        "missing_rank_policy": "median rank within at least65% observed inputs",
        "training_cohort_policy": "Residualize features and clipped labels on the same finite, eligible label cohort; empty or control-saturated months do not count toward minimum history",
        "states": states,
    }
