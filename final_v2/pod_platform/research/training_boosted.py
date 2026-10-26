"""Shallow nonlinear pod forecasts with annual, strictly chronological fits."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import xgboost as xgb
except ImportError:  # The optional learner never becomes a silent substitute.
    xgb = None

from .training_data import POD_FEATURES, market_return_series
from .training_search import (
    candidate,
    make_weights,
    objective,
    risk_designs,
    summarize,
)
from .training_tactical import monthly_net, panel_schedule, replay_daily


def annual_boosted(
    panel, features, depth, lookback, settings, *, model_path=None
):
    """Use resolved labels, clipped training outcomes and no issuer identifiers.

    Hyperparameters are fixed before prediction. Annual December refits serve
    the next year. The first fit waits for the configured history; the final
    December fit uses all eligible development labels for the frozen model.
    """
    if xgb is None:
        raise RuntimeError("Boosted research requires the xgboost/scipy extras")
    designs = risk_designs(panel, settings)
    cube = np.stack(
        [np.nan_to_num(panel.features[f]) for f in features], axis=2
    )
    train_x, train_y = [], []
    for t, (mask, control, inverse) in enumerate(designs):
        value = cube[t, mask]
        cube[t, mask] = value - control @ (inverse @ value)
        if t >= len(panel.returns):
            continue
        entries = panel.entry_mask[t, mask]
        local = control[entries]
        outcome = panel.returns[t, mask][entries].astype(float)
        outcome -= local @ (np.linalg.pinv(local) @ outcome)
        train_x.append(cube[t, mask][entries].astype(np.float32))
        train_y.append(outcome.astype(np.float32))
    predictions = np.zeros(panel.mask.shape)
    model = None
    last_fit = None
    parameters = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": depth,
        "eta": settings["gbdt_learning_rate"],
        "min_child_weight": settings["gbdt_min_child_weight"],
        "lambda": settings["gbdt_l2"],
        "max_bin": 64,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "seed": 44,
        "nthread": 1,
        "verbosity": 0,
    }
    for t, date in enumerate(panel.dates):
        known = np.flatnonzero(panel.settlement_dates <= date)[-lookback:]
        if len(known) < settings["gbdt_min_history"]:
            continue
        refit = model is None or date.month == 12 or t == len(panel.dates) - 1
        if refit:
            matrix = np.concatenate([train_x[k] for k in known])
            outcome = np.concatenate([train_y[k] for k in known])
            if not len(outcome):
                continue
            low, high = np.quantile(outcome, [0.005, 0.995])
            outcome = np.clip(outcome, low, high)
            average = len(outcome) / len(known)
            weights = np.concatenate(
                [
                    np.full(len(train_y[k]), average / len(train_y[k]))
                    for k in known
                    if len(train_y[k])
                ]
            )
            model = xgb.train(
                parameters,
                xgb.DMatrix(matrix, label=outcome, weight=weights, nthread=1),
                num_boost_round=settings["gbdt_rounds"],
            )
            last_fit = str(date.date())
        predictions[t] = model.predict(
            xgb.DMatrix(cube[t].astype(np.float32), nthread=1)
        )
    if model is not None and model_path is not None:
        destination = Path(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(destination)
    return predictions, last_fit


def search_boosted(panel, market, settings, *, model_dir=None, progress=None):
    """Compare a predeclared shallow-tree grid with the same execution rules."""
    trials = []
    designs = risk_designs(panel, settings)
    for pod, features in POD_FEATURES.items():
        for depth in settings.get("gbdt_depths", []):
            for lookback in settings["gbdt_lookbacks"]:
                model_name = f"{pod}_depth{depth}_window{lookback}.json"
                model_path = Path(model_dir) / model_name if model_dir else None
                prediction, fitted = annual_boosted(
                    panel,
                    features,
                    depth,
                    lookback,
                    settings,
                    model_path=model_path,
                )
                weights = make_weights(prediction, designs, settings)
                for review in settings["gbdt_review_months"]:
                    recipe = {
                        "model": "annual_shallow_boosted_trees",
                        "features": features,
                        "depth": depth,
                        "lookback_months": lookback,
                        "rebalance_months": review,
                        "refit": "december_using_settled_labels",
                        "min_history": settings["gbdt_min_history"],
                        "rounds": settings["gbdt_rounds"],
                        "learning_rate": settings["gbdt_learning_rate"],
                        "l2": settings["gbdt_l2"],
                        "min_child_weight": settings["gbdt_min_child_weight"],
                        "final_fit_asof": fitted,
                        "model_artifact": str(model_path)
                        if model_path
                        else None,
                        "xgboost_version": xgb.__version__,
                    }
                    trial = candidate(
                        f"{pod}:boosted_depth{depth}:window{lookback}:review{review}m",
                        pod,
                        recipe,
                        weights,
                        panel,
                        settings,
                    )
                    book = replay_daily(
                        market,
                        panel_schedule(weights, panel, market, review),
                        settings,
                    )
                    trial.net = monthly_net(book["net"], market, panel)
                    trial.metrics = summarize(
                        trial.net, market_return_series(panel)
                    )
                    trial.score = objective(trial.metrics)
                    trials.append(trial)
        if progress:
            progress(pod + "_boosted", len(trials))
    return trials
