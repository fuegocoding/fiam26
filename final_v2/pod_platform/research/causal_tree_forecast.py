"""Agent-specified nonlinear forecasts, fitted monthly on settled cohort labels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_tree_forecast(panel, features, *, parameters, penalty, lookback, minimum_history=12):
    import xgboost as xgb

    if not features or len(features) != len(set(features)) or not set(features) <= set(panel.features):
        raise ValueError("Choose distinct available features for the nonlinear model")
    if penalty <= 0 or lookback < minimum_history:
        raise ValueError("Positive regularization and sufficient settled history are required")
    params = parameters.model_dump() if hasattr(parameters, "model_dump") else dict(parameters)
    forecasts = np.full(panel.mask.shape, np.nan, dtype=np.float32)
    states, samples = [], {}
    for t, date in enumerate(panel.dates):
        # An unsettled label is never materialized in the trainer.
        known = np.flatnonzero(panel.settlement_dates <= date)[-lookback:]
        for old in set(samples) - set(known):
            del samples[old]
        usable = []
        for k in known:
            k = int(k)
            if k not in samples:
                x = np.column_stack([panel.features[name][k] for name in features]).astype(np.float32)
                y = np.asarray(panel.returns[k], float)
                mask = panel.mask[k] & panel.entry_mask[k] & np.isfinite(y) & (np.isfinite(x).mean(axis=1) >= .65)
                x, y = x[mask], y[mask]
                if len(y) >= 5:
                    y = (pd.Series(y).rank(pct=True).to_numpy() - .5 if params["label"] == "rank"
                         else np.clip(y, *np.quantile(y, [.025, .975])))
                    samples[k] = (x, y.astype(np.float32))
                else:
                    samples[k] = None
            if samples[k] is not None:
                usable.append(k)
        state = {"formation_date": str(date.date()), "settled_months": len(usable),
                 "last_label_settlement": str(panel.settlement_dates[usable[-1]].date()) if usable else None,
                 "fitted": False, "training_rows": sum(len(samples[k][1]) for k in usable)}
        if len(usable) >= minimum_history:
            x = np.concatenate([samples[k][0] for k in usable])
            y = np.concatenate([samples[k][1] for k in usable])
            weights = np.concatenate([np.full(len(samples[k][1]), len(y) / len(usable) / len(samples[k][1])) for k in usable])
            data = xgb.DMatrix(x, label=y, weight=weights, feature_names=features, nthread=1)
            model = xgb.train({"objective": "reg:squarederror", "tree_method": "hist", "nthread": 1,
                "max_bin": 64, "seed": 0, "verbosity": 0, "max_depth": params["max_depth"],
                "eta": params["learning_rate"], "min_child_weight": params["min_child_weight"],
                "subsample": params["subsample"], "colsample_bytree": params["column_subsample"],
                "lambda": penalty}, data, num_boost_round=params["rounds"])
            current = np.column_stack([panel.features[name][t] for name in features]).astype(np.float32)
            mask = panel.mask[t] & (np.isfinite(current).mean(axis=1) >= .65)
            if np.any(mask):
                forecasts[t, mask] = model.predict(xgb.DMatrix(current[mask], feature_names=features, nthread=1))
            state.update(fitted=True, feature_gain=model.get_score(importance_type="gain"))
            del model, data, x, y, weights
        states.append(state)
    return forecasts, {"model": "monthly_causal_xgboost", "features": features, "parameters": params,
        "penalty": penalty, "lookback_months": lookback, "minimum_settled_months": minimum_history,
        "training_cohort_policy": "Only declared dated alpha cohort and observed settled labels; equal total weight per month",
        "missing_feature_policy": "Native missing-value splits within at least65% observed inputs",
        "states": states}
