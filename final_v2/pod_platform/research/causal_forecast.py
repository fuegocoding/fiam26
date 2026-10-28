"""A small trusted trainer; generated strategies never receive return labels."""

from __future__ import annotations

import numpy as np

from .pod_plans import construction_designs
from .forecast_labels import ISSUER_FOLD_POLICY, issuer_validation_masks, label_metadata, training_target


def fit_ridge_forecast(
    panel,
    features,
    *,
    penalty=1.0,
    lookback=36,
    minimum_history=12,
    controls="sector_beta",
    validation_folds=0,
):
    """Fit month-weighted sufficient statistics using settled observations only.

    One cross-section is materialized at a time, so all supplied characteristics
    can be requested without building a full stock-month-feature cube. Missing
    centered ranks use median-rank imputation inside the trusted fit; a forecast
    requires at least65% observed features. Labels are clipped by fixed monthly
    2.5/97.5 percentile rules before they enter the training sufficient statistics.
    These choices are declared research policy, not learned performance claims.
    """
    if validation_folds not in (0, 2):
        raise ValueError("Issuer validation requires zero or two folds")
    cohort = np.any(panel.mask, axis=0)
    folds_to_validate = (issuer_validation_masks(panel.securities, cohort=cohort)
        if validation_folds else ())
    arguments = dict(penalty=penalty, lookback=lookback,
        minimum_history=minimum_history, controls=controls)
    forecasts, state = _fit_ridge_forecast(panel, features, **arguments)
    state["issuer_oof_folds"] = validation_folds
    if validation_folds:
        oof = np.full(panel.mask.shape, np.nan, dtype=np.float32)
        folds = []
        for held_out in folds_to_validate:
            predicted, fold_state = _fit_ridge_forecast(panel, features,
                training_issuers=~held_out, **arguments)
            oof[:, held_out] = predicted[:, held_out]
            folds.append({"held_out_issuers": int(np.count_nonzero(held_out & cohort)),
                "training_issuers": int(np.count_nonzero(~held_out & cohort)),
                "excluded_training_label_rows": fold_state["excluded_training_label_rows"],
                "fitted_formations": sum(s["coefficients"] is not None for s in fold_state["states"])})
            del predicted, fold_state
        state["_issuer_oof_forecasts"] = oof
        state["issuer_oof_validation"] = {"fold_policy": ISSUER_FOLD_POLICY,
            "exclusion": "held-out issuers excluded from all historical training labels and label transformations",
            "folds": folds}
    return forecasts, state


def _fit_ridge_forecast(panel, features, *, penalty, lookback, minimum_history,
    controls, training_issuers=None):
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
    if training_issuers is None:
        training_issuers = np.ones(len(panel.securities), dtype=bool)
    dimension = len(features)
    covariance = np.zeros((len(panel.returns), dimension, dimension))
    cross = np.zeros((len(panel.returns), dimension))
    training_rows = np.zeros(len(panel.returns), dtype=int)
    materialized = np.zeros(len(panel.returns), dtype=bool)
    excluded_rows = 0
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
        for k in known:
            if materialized[k]:
                continue
            # Read a label only when its entire declared horizon has settled.
            historical = np.column_stack([panel.features[name][k] for name in features]).astype(float)
            eligible = panel.mask[k] & (np.isfinite(historical).mean(axis=1) >= 0.65)
            label = np.asarray(panel.returns[k], dtype=float)
            finite = eligible & panel.entry_mask[k] & np.isfinite(label)
            excluded_rows += int(np.count_nonzero(finite & ~training_issuers))
            finite &= training_issuers
            finite, y = training_target(panel, k, label, finite)
            fit_controls = designs[k].matrix[finite[designs[k].mask]]
            x = np.nan_to_num(historical[finite], nan=0.0)
            if len(y) >= 2 and len(y) > np.linalg.matrix_rank(fit_controls) + 1:
                y = np.clip(y, *np.quantile(y, [0.025, 0.975]))
                inverse = np.linalg.pinv(fit_controls)
                x -= fit_controls @ (inverse @ x)
                y -= fit_controls @ (inverse @ y)
                covariance[k] = x.T @ x / len(y)
                cross[k] = x.T @ y / len(y)
                training_rows[k] = len(y)
            materialized[k] = True
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
        "excluded_training_label_rows": excluded_rows,
        **label_metadata(panel),
        "states": states,
    }
