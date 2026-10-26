"""Trusted learned forecasts must not expose future labels to generated code."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pod_platform.research.causal_forecast import fit_ridge_forecast


def panel_fixture(dimension=4):
    rng = np.random.default_rng(61)
    dates = pd.date_range("2017-01-31", periods=20, freq="ME")
    names = 80
    features = {
        f"x{i}": rng.normal(size=(len(dates), names)).astype(np.float32)
        for i in range(dimension)
    }
    frame = pd.DataFrame(
        {
            "me": np.full(names, 20000.0),
            "gics": ["10"] * 40 + ["20"] * 40,
            "beta_60m": rng.uniform(0.5, 1.5, names),
        }
    )
    returns = features["x0"][:-1] * 0.02 + rng.normal(
        0, 0.03, (len(dates) - 1, names)
    )
    return SimpleNamespace(
        dates=dates,
        mask=np.ones((len(dates), names), bool),
        frames=[frame] * len(dates),
        features=features,
        returns=returns,
        entry_mask=np.ones_like(returns, bool),
        settlement_dates=dates[1:],
    )


def test_future_labels_cannot_change_past_model_forecasts_or_states():
    panel = panel_fixture()
    original, state = fit_ridge_forecast(
        panel, list(panel.features), minimum_history=6
    )
    panel.returns[13:] = 99.0
    changed, new_state = fit_ridge_forecast(
        panel, list(panel.features), minimum_history=6
    )
    np.testing.assert_array_equal(original[:14], changed[:14])
    assert state["states"][:14] == new_state["states"][:14]
    assert np.isfinite(original[6:]).any()
    assert not np.isfinite(original[:6]).any()
    for row in state["states"]:
        if row["coefficients"] is not None:
            assert row["last_label_settlement"] <= row["formation_date"]


def test_all_147_features_are_permitted_and_missing_cohorts_stay_unknown():
    panel = panel_fixture(147)
    for feature in panel.features.values():
        feature[:, 0] = np.nan
    forecast, state = fit_ridge_forecast(
        panel, list(panel.features), minimum_history=6
    )
    assert len(state["features"]) == 147
    assert np.isnan(forecast[:, 0]).all()
    assert np.isfinite(forecast[6:, 1:]).all()
    with pytest.raises(ValueError, match="available research"):
        fit_ridge_forecast(panel, ["ret_exc_lead1m"])


def test_empty_label_months_do_not_satisfy_minimum_training_history():
    panel = panel_fixture()
    panel.entry_mask[:13] = False
    forecast, state = fit_ridge_forecast(
        panel, list(panel.features), minimum_history=6
    )
    assert np.isnan(forecast[:19]).all()
    assert np.isfinite(forecast[19:]).any()
    assert state["states"][18]["settled_months"] == 5
    assert state["states"][19]["settled_months"] == 6
    assert state["states"][19]["training_rows"] == 6 * 80
