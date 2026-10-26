"""Annual removal must close both pod diagnostics and the real fund schedule."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from pod_platform.research import training_walkforward
from pod_platform.research.training_search import Candidate
from pod_platform.research.training_tactical import DailyMarket


def test_a_dropped_rule_stops_earning_in_both_pod_and_fund(monkeypatch):
    dates = pd.date_range("2015-01-31", "2020-12-31", freq="ME")
    panel = SimpleNamespace(
        dates=dates, settlement_dates=dates[1:], securities=np.array([1, 2])
    )
    candidate = Candidate(
        "quality",
        "fundamentals",
        {},
        np.zeros((72, 2)),
        np.full(71, 0.01),
        np.zeros(71),
        {},
        1.0,
    )
    calendar = pd.bdate_range("2015-01-01", "2020-12-31")
    returns = np.tile([0.001, -0.001], (len(calendar), 1))
    market = DailyMarket(
        calendar,
        returns,
        np.ones_like(returns, dtype=bool),
        np.zeros(len(calendar), dtype=int),
        [],
    )
    pods = ("fundamentals", "events", "valuation", "dynamics")
    monkeypatch.setattr(training_walkforward, "risk_designs", lambda *args: [])
    monkeypatch.setattr(
        training_walkforward, "selection_universe", lambda c, s: c
    )

    def select(prior, settings):
        return {
            pod: [prior[0]]
            if pod == "fundamentals" and len(prior[0].net) < 50
            else []
            for pod in pods
        }

    monkeypatch.setattr(training_walkforward, "select_strategies", select)
    monkeypatch.setattr(
        training_walkforward,
        "allocation_search",
        lambda *args, **kwargs: ({"allocations": [], "policy": {}}, []),
    )

    def schedule(selected, *args):
        if not any(selected.values()):
            return {}
        return {
            int(calendar.searchsorted(pd.Timestamp(year, 1, 2))): np.array(
                [0.04, -0.04]
            )
            for year in (2018, 2019, 2020)
        }

    monkeypatch.setattr(training_walkforward, "combined_schedule", schedule)
    result = training_walkforward.walk_forward(
        [candidate],
        panel,
        market,
        {
            "walk_forward_years": [2018, 2019, 2020],
            "selection_family": "mechanisms",
            "cost_bps_one_way": 0,
            "borrow_bps_annual": 0,
            "strategy_capital_usd": 1000,
            "liquidate_at_end": False,
        },
    )
    days = result["dates"]
    before = days.year == 2019
    after_exit = days >= pd.Timestamp("2020-01-02")
    for values in (
        result["daily_returns"],
        result["pods"]["fundamentals"]["daily_returns"],
    ):
        assert values[before].sum() > 0
        np.testing.assert_array_equal(values[after_exit], 0)
    # Jan 1 close liquidation still earns Jan 1's carried return before exit.
    assert result["daily_returns"][days == pd.Timestamp("2020-01-01")][0] > 0
