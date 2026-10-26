"""Causal timing and matched benchmarks for the directional research probe."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from diagnose_dynamics import directional_schedule, market_index


def _inputs():
    dates = pd.bdate_range("2019-01-01", periods=80)
    frame = pd.DataFrame(
        {
            "me": np.full(350, 20_000),
            "dolvol_126d": np.arange(350) + 100_000_000,
            "rvol_21d": np.full(350, 0.02),
        }
    )
    panel = SimpleNamespace(
        mask=np.ones((1, 350), dtype=bool),
        frames=[frame],
        securities=np.arange(350),
    )
    market = SimpleNamespace(
        dates=dates,
        formation=np.zeros(80, dtype=int),
        returns=np.full((80, 350), 0.001),
    )
    recipe = {
        "lookback": 10,
        "review": "weekly",
        "rule": "long_cash",
        "universe": "liquid",
    }
    return panel, market, recipe


def test_directional_orders_do_not_use_execution_or_future_returns():
    panel, market, recipe = _inputs()
    reference = market_index(panel, market)
    original = directional_schedule(panel, market, reference, recipe)
    market.returns[40:] = -0.1
    changed = directional_schedule(
        panel, market, market_index(panel, market), recipe
    )
    for day in original:
        if day <= 40:
            np.testing.assert_array_equal(original[day], changed[day])
    assert any(
        not np.array_equal(original[d], changed[d]) for d in original if d > 40
    )


def test_passive_comparator_has_identical_calendar_and_basket():
    panel, market, recipe = _inputs()
    reference = market_index(panel, market)
    active = directional_schedule(panel, market, reference, recipe)
    passive = directional_schedule(
        panel, market, reference, {**recipe, "rule": "passive"}
    )
    assert active.keys() == passive.keys()
    for day in active:
        np.testing.assert_array_equal(active[day], passive[day])
        assert np.count_nonzero(active[day]) == 300
        np.testing.assert_allclose(active[day].sum(), 0.4)
