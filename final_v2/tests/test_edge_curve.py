"""Test signal evidence across ranked coverage without specifying capital or book size."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pod_platform.application.edge_research import host_control, select_owned
from pod_platform.research.edge_curve import (
    freeze_prefix_choice,
    signal_prefixes,
    summarize_prefixes,
)


def _panel():
    rng = np.random.default_rng(211)
    dates = pd.date_range("2015-01-31", periods=72, freq="ME")
    values = rng.uniform(-1, 1, (72, 200))
    frames = [
        pd.DataFrame(
            {
                "me": np.full(200, 20_000.0),
                "gics": "10",
                "beta_60m": np.ones(200),
                "dolvol_126d": np.ones(200),
            }
        )
        for _ in dates
    ]
    return SimpleNamespace(
        dates=dates,
        frames=frames,
        mask=np.ones((72, 200), dtype=bool),
        features={"cash_at": values},
        returns=values[:-1] * 0.01 + rng.normal(0, 0.002, (71, 200)),
        entry_mask=np.ones((71, 200), dtype=bool),
        execution_dates=dates[:-1] + pd.Timedelta(days=1),
        settlement_dates=dates[1:] + pd.Timedelta(days=1),
    )


def test_edge_curve_examines_all_feasible_prefixes_without_a_target_count():
    panel = _panel()
    plan = host_control("test_signal", {"cash_at": 1}, 1, "dollar")
    prefixes, ic = signal_prefixes(plan, panel)
    first = prefixes.loc[prefixes.formation_date == panel.dates[0]]
    np.testing.assert_array_equal(first.names, np.arange(1, len(first) + 1) * 2)
    assert len(first) > 50
    assert ic.rank_ic.mean() > 0.9
    curve = summarize_prefixes(prefixes)
    assert curve.loc[curve.names == 50, "mean"].iloc[0] > 0
    assert curve.loc[curve.names == 100, "mean"].iloc[0] > 0


def test_prefix_choice_cannot_use_returns_that_have_not_settled():
    panel = _panel()
    plan = host_control("test_signal", {"cash_at": 1}, 1, "dollar")
    original, _ = signal_prefixes(plan, panel)
    cutoff = pd.Timestamp("2017-12-31")
    before = freeze_prefix_choice({"signal": original}, cutoff)
    assert before is not None
    panel.returns[panel.settlement_dates > cutoff] *= -100
    changed, _ = signal_prefixes(plan, panel)
    after = freeze_prefix_choice({"signal": changed}, cutoff)
    assert before == after


def test_ownership_uses_signal_edge_not_capital_or_implementation_returns():
    rng = np.random.default_rng(44)
    strong = rng.normal(0.02, 0.01, 36)
    weak = rng.normal(-0.01, 0.02, 36)
    trials = {
        "strong": {
            "gross_monthly": strong,
            "monthly": np.full(36, -0.9),
            "capital_usd": 1,
        },
        "weak": {
            "gross_monthly": weak,
            "monthly": np.ones(36),
            "capital_usd": 1e9,
        },
    }
    settings = {"max_strategies_per_pod": 1}
    assert select_owned(list(trials), trials, np.arange(36), settings) == [
        "strong"
    ]
    trials["strong"]["capital_usd"] = 1e12
    assert select_owned(list(trials), trials, np.arange(36), settings) == [
        "strong"
    ]
