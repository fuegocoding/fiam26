"""Agent-specified nonlinear forecasts, fitted monthly on settled cohort labels."""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from .forecast_labels import ISSUER_FOLD_POLICY, issuer_validation_masks, label_metadata, training_target


def _training_device(xgb, data, parameters, rounds, requested):
    """Choose the faster backend on settled training rows, never on outcomes."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("Tree device must be auto, cpu or cuda")
    timing = {"requested": requested, "selected": "cpu"}
    if requested == "cpu":
        return "cpu", timing
    if not xgb.build_info().get("USE_CUDA"):
        if requested == "cuda":
            raise ValueError("CUDA was requested but this XGBoost build has no CUDA support")
        return "cpu", {**timing, "reason": "xgboost_build_without_cuda"}
    probe_rounds = min(30, rounds)
    try:
        gpu_times = []
        # Repeated monthly fits share a CUDA context. The second probe avoids
        # charging its one-time initialization to every future monthly fit.
        for _ in range(2 if requested == "auto" else 1):
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                probe = xgb.train({**parameters, "device": "cuda"}, data,
                    num_boost_round=probe_rounds)
            actual = json.loads(probe.save_config())["learner"]["generic_param"]["device"]
            del probe
            if not actual.startswith("cuda"):
                if requested == "cuda":
                    raise ValueError("CUDA was requested but the NVIDIA device is unavailable")
                return "cpu", {**timing, "reason": "cuda_device_unavailable"}
            gpu_times.append(time.perf_counter() - started)
        timing.update(cuda_seconds=min(gpu_times), probe_rounds=probe_rounds)
    except xgb.core.XGBoostError as exc:
        if requested == "cuda":
            raise
        return "cpu", {**timing, "reason": "cuda_probe_failed", "detail": str(exc)[:200]}
    if requested == "cuda":
        return "cuda", {**timing, "selected": "cuda"}
    started = time.perf_counter()
    probe = xgb.train({**parameters, "device": "cpu"}, data, num_boost_round=probe_rounds)
    del probe
    timing["cpu_seconds"] = time.perf_counter() - started
    device = "cuda" if timing["cuda_seconds"] < timing["cpu_seconds"] * 0.90 else "cpu"
    return device, {**timing, "selected": device, "reason": "training_time_comparison"}


def fit_tree_forecast(panel, features, *, parameters, penalty, lookback, minimum_history=12,
    device="auto", validation_folds=0):
    """Fit causal forecasts and optionally forecasts for wholly unseen issuers."""
    if validation_folds not in (0, 2):
        raise ValueError("Issuer validation requires zero or two folds")
    cohort = np.any(panel.mask, axis=0)
    folds_to_validate = (issuer_validation_masks(panel.securities, cohort=cohort)
        if validation_folds else ())
    arguments = dict(parameters=parameters, penalty=penalty, lookback=lookback,
        minimum_history=minimum_history)
    forecasts, state = _fit_tree_forecast(panel, features, device=device, **arguments)
    state["issuer_oof_folds"] = validation_folds
    if validation_folds:
        oof = np.full(panel.mask.shape, np.nan, dtype=np.float32)
        folds = []
        # Reuse the measured backend rather than repeating the auto benchmark.
        fold_device = state["training_device"].get("selected") or device
        for held_out in folds_to_validate:
            predicted, fold_state = _fit_tree_forecast(panel, features,
                training_issuers=~held_out, device=fold_device, **arguments)
            oof[:, held_out] = predicted[:, held_out]
            folds.append({"held_out_issuers": int(np.count_nonzero(held_out & cohort)),
                "training_issuers": int(np.count_nonzero(~held_out & cohort)),
                "excluded_training_label_rows": fold_state["excluded_training_label_rows"],
                "fitted_formations": sum(s["fitted"] for s in fold_state["states"]),
                "training_device": fold_state["training_device"]})
            del predicted, fold_state
        state["_issuer_oof_forecasts"] = oof
        state["issuer_oof_validation"] = {"fold_policy": ISSUER_FOLD_POLICY,
            "exclusion": "held-out issuers excluded from all historical training labels and label transformations",
            "folds": folds}
    return forecasts, state


def _fit_tree_forecast(panel, features, *, parameters, penalty, lookback, minimum_history,
    device, training_issuers=None):
    import xgboost as xgb

    if not features or len(features) != len(set(features)) or not set(features) <= set(panel.features):
        raise ValueError("Choose distinct available features for the nonlinear model")
    if penalty <= 0 or lookback < minimum_history:
        raise ValueError("Positive regularization and sufficient settled history are required")
    params = parameters.model_dump() if hasattr(parameters, "model_dump") else dict(parameters)
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Tree device must be auto, cpu or cuda")
    if training_issuers is None:
        training_issuers = np.ones(len(panel.securities), dtype=bool)
    forecasts = np.full(panel.mask.shape, np.nan, dtype=np.float32)
    states, samples = [], {}
    excluded_rows = 0
    selected_device, device_evidence = None, {"requested": device, "selected": None}
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
                excluded_rows += int(np.count_nonzero(mask & ~training_issuers))
                mask &= training_issuers
                mask, y = training_target(panel, k, y, mask)
                x = x[mask]
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
            training_parameters = {"objective": "reg:squarederror", "tree_method": "hist", "nthread": 1,
                "max_bin": 64, "seed": 0, "verbosity": 0, "max_depth": params["max_depth"],
                "eta": params["learning_rate"], "min_child_weight": params["min_child_weight"],
                "subsample": params["subsample"], "colsample_bytree": params["column_subsample"],
                "lambda": penalty}
            if selected_device is None:
                selected_device, device_evidence = _training_device(xgb, data,
                    training_parameters, params["rounds"], device)
            try:
                model = xgb.train({**training_parameters, "device": selected_device},
                    data, num_boost_round=params["rounds"])
            except xgb.core.XGBoostError as exc:
                if device != "auto" or selected_device != "cuda":
                    raise
                selected_device = "cpu"
                device_evidence.update(selected="cpu", reason="cuda_training_failed",
                    detail=str(exc)[:200])
                model = xgb.train({**training_parameters, "device": "cpu"},
                    data, num_boost_round=params["rounds"])
            current = np.column_stack([panel.features[name][t] for name in features]).astype(np.float32)
            mask = panel.mask[t] & (np.isfinite(current).mean(axis=1) >= .65)
            if np.any(mask):
                forecasts[t, mask] = model.predict(xgb.DMatrix(current[mask], feature_names=features, nthread=1))
            state.update(fitted=True, device=selected_device, feature_gain=model.get_score(importance_type="gain"))
            del model, data, x, y, weights
        states.append(state)
    return forecasts, {"model": "monthly_causal_xgboost", "features": features, "parameters": params,
        "penalty": penalty, "lookback_months": lookback, "minimum_settled_months": minimum_history,
        "training_device": device_evidence,
        "excluded_training_label_rows": excluded_rows,
        **label_metadata(panel),
        "training_cohort_policy": "Only declared dated alpha cohort and observed settled labels; equal total weight per month",
        "missing_feature_policy": "Native missing-value splits within at least65% observed inputs",
        "states": states}
