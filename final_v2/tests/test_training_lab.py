"""Timing, normalization, turnover and chronological allocation regressions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pod_platform.research.training_allocation import allocate, capped_simplex
from pod_platform.research.training_boosted import annual_boosted
from pod_platform.research.training_data import interval_returns, read_panel
from pod_platform.research.training_models import rolling_ridge
from pod_platform.research.training_search import (
    make_weights,
    project_target,
    risk_designs,
    smooth_signal,
)
from pod_platform.research.training_tactical import (
    DailyMarket,
    combined_schedule,
    replay_daily,
    tactical_schedule,
    threshold_positions,
)


def _settings():
    return {
        "gross_limit": 2,
        "single_name_limit": 0.05,
        "cost_bps_one_way": 0,
        "borrow_bps_annual": 0,
        "allocation_min_history": 1,
        "max_pod_capital": 0.5,
    }


def test_panel_reader_never_requests_forward_labels_or_post_2020(monkeypatch):
    calls = []

    def reader(path, *, columns, filters):
        calls.append((columns, filters))
        return pd.DataFrame(
            {"permno": [1], "eom": [pd.Timestamp("2020-12-31")]}
        )

    monkeypatch.setattr(pd, "read_parquet", reader)
    read_panel("panel.parquet", {"start": "2011-01-01", "end": "2020-12-31"})
    columns, filters = calls[0]
    assert "ret_exc_lead1m" not in columns
    assert ("eom", "<=", pd.Timestamp("2020-12-31")) in filters
    with pytest.raises(ValueError, match="2020"):
        read_panel(
            "panel.parquet", {"start": "2015-01-01", "end": "2021-01-01"}
        )


def test_signal_smoothing_has_no_future_dependency():
    original = np.arange(24, dtype=float).reshape(6, 4)
    changed = original.copy()
    changed[4:] *= -1e9
    before = smooth_signal(original, 3)
    after = smooth_signal(changed, 3)
    np.testing.assert_array_equal(before[:4], after[:4])
    np.testing.assert_allclose(before[2], original[:3].mean(axis=0))


def test_rolling_model_cannot_fit_returns_before_their_settlement():
    rng = np.random.default_rng(25)
    count, names = 9, 40
    dates = pd.date_range("2017-01-31", periods=count, freq="ME")
    frames = [
        pd.DataFrame(
            {
                "gics": ["10"] * 20 + ["20"] * 20,
                "beta_60m": rng.uniform(0.5, 1.5, names),
                "me": rng.uniform(1000, 20_000, names),
            }
        )
        for _ in dates
    ]
    panel = SimpleNamespace(
        dates=dates,
        frames=frames,
        mask=np.ones((count, names), dtype=bool),
        features={
            "x": rng.normal(size=(count, names)),
            "y": rng.normal(size=(count, names)),
        },
        returns=rng.normal(0, 0.02, size=(count - 1, names)),
        settlement_dates=dates[1:],
        entry_mask=np.ones((count - 1, names), dtype=bool),
    )
    original, _ = rolling_ridge(panel, ["x", "y"], 1.0, 24, 2)
    panel.returns[5:] = 100
    changed, _ = rolling_ridge(panel, ["x", "y"], 1.0, 24, 2)
    np.testing.assert_array_equal(original[:6], changed[:6])
    assert np.count_nonzero(original[:2]) == 0
    assert np.count_nonzero(original[2:]) > 0


def test_risk_projection_removes_controls_and_preserves_scale_invariance():
    names = 200
    rng = np.random.default_rng(7)
    design = np.column_stack([np.ones(names), np.linspace(-1, 1, names)])
    mask = np.ones(names, dtype=bool)
    designs = [(mask, design, np.linalg.pinv(design))]
    signal = rng.normal(size=(1, names))
    first = make_weights(signal, designs, _settings())
    second = make_weights(signal * 100_000_000, designs, _settings())
    np.testing.assert_allclose(first, second, atol=1e-8)
    np.testing.assert_allclose(design.T @ first[0], [0, 0], atol=1e-7)
    assert np.abs(first).sum() == pytest.approx(2)
    assert np.abs(first).max() <= 0.05


def test_wider_research_universe_still_produces_at_most_500_neutral_positions():
    rng = np.random.default_rng(90)
    names = 1500
    sector = np.eye(10)[np.arange(names) % 10]
    controls = np.column_stack([sector, rng.normal(size=(names, 2))])
    weights = project_target(
        rng.normal(size=names),
        controls,
        {**_settings(), "universe_size": 500},
        normalize=True,
    )
    assert np.count_nonzero(weights) <= 500
    np.testing.assert_allclose(controls.T @ weights, 0, atol=1e-9)
    assert np.abs(weights).sum() <= 2 + 1e-8
    assert np.abs(weights).max() <= 0.05


def test_small_caps_can_be_long_but_cannot_become_shorts_during_neutralization():
    rng = np.random.default_rng(91)
    names = 600
    controls = np.column_stack([np.ones(names), rng.normal(size=(names, 2))])
    shortable = np.arange(names) >= 200
    values = rng.normal(size=names)
    values[:100] = 3
    weights = project_target(
        values,
        controls,
        {**_settings(), "universe_size": 500},
        normalize=True,
        shortable=shortable,
    )
    assert np.any(weights[:200] > 0)
    assert np.all(weights[:200] >= -1e-12)
    np.testing.assert_allclose(controls.T @ weights, 0, atol=1e-9)


def test_daily_execution_cannot_earn_information_or_execution_day_returns():
    dates = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04"])
    returns = np.array([[9.0, -0.9], [0.5, 0], [0.1, -0.1]])
    market = DailyMarket(
        dates, returns, np.ones((3, 2), dtype=bool), np.zeros(3), []
    )
    book = replay_daily(market, {1: np.array([1.0, -1.0])}, _settings())
    np.testing.assert_allclose(book["net"], [0, 0, 0.2])


def test_interval_returns_skip_execution_close_and_exit_at_missing_quote():
    dates = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-03",
            "2020-02-04",
            "2020-02-05",
            "2020-02-28",
            "2020-03-02",
        ]
    )
    daily = pd.DataFrame(
        {
            "permno": [1] * 6,
            "date": dates,
            "prc": [10, 20, 22, np.nan, 30, 40],
            "daily_return": [0, 1, 0.1, np.nan, 0.5, 0.5],
        }
    )
    execution, settlement, realized, exits, entry = interval_returns(
        daily,
        pd.to_datetime(["2020-01-31"]),
        np.array([1]),
        pd.Timestamp("2020-03-02"),
    )
    assert execution[0] == pd.Timestamp("2020-02-03")
    assert settlement[0] == pd.Timestamp("2020-03-02")
    assert realized[0, 0] == pytest.approx(0.1)
    assert exits[0, 0] and entry[0, 0]


def test_thresholds_hold_between_entry_and_exit_and_allow_direction_changes():
    previous = np.array([1, 1, -1, -1, 0, 0], dtype=float)
    score = np.array([0.4, 0.1, -0.4, -0.1, 0.7, -0.7])
    actual = threshold_positions(score, previous, 0.6, 0.2)
    np.testing.assert_array_equal(actual, [1, 0, -1, 0, 1, -1])
    with pytest.raises(ValueError, match="Thresholds"):
        threshold_positions(score, previous, 0.2, 0.6)


def test_daily_replay_charges_entry_exit_and_forced_exit_without_redistributing():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    tradable = np.ones((4, 2), dtype=bool)
    tradable[2:, 0] = False
    returns = np.zeros((4, 2))
    returns[2, 0] = np.nan
    returns[3, 0] = (
        100  # A later bogus observation cannot revive the exited book.
    )
    market = DailyMarket(dates, returns, tradable, np.zeros(4), [])
    settings = {**_settings(), "cost_bps_one_way": 10}
    book = replay_daily(market, {1: np.array([1.0, -1.0])}, settings)
    assert book["net"][1] < 0
    assert book["net"][2] < 0
    assert book["net"][3] < 0
    assert np.prod(1 + book["net"]) < 1


def test_capped_allocation_and_future_returns_cannot_change_earlier_capital():
    actual = capped_simplex([2, 0, 0, 0], 0.4)
    assert actual.sum() == pytest.approx(1)
    assert actual.max() <= 0.4 + 1e-8
    dates = pd.date_range("2018-01-31", periods=5, freq="ME")
    shape = (5, 2)
    panel = SimpleNamespace(
        dates=dates,
        mask=np.ones(shape, dtype=bool),
        settlement_dates=dates[1:],
        execution_dates=dates[:-1],
        returns=np.zeros((4, 2)),
        entry_mask=np.ones((4, 2), dtype=bool),
        exit_mask=np.zeros((4, 2), dtype=bool),
    )
    positive = np.array([0.01, 0.02, -0.01, 0.01])
    negative = -positive
    strategies = {
        "a": [
            SimpleNamespace(
                strategy_id="a", weights=np.zeros(shape), net=positive.copy()
            )
        ],
        "b": [
            SimpleNamespace(
                strategy_id="b", weights=np.zeros(shape), net=negative.copy()
            )
        ],
    }
    policy = {
        "method": "performance",
        "lookback_months": 12,
        "risk_aversion": 5,
    }
    original = allocate(strategies, panel, _settings(), policy)
    strategies["a"][0].net[2:] = 100
    strategies["b"][0].net[2:] = -0.99
    changed = allocate(strategies, panel, _settings(), policy)
    first = [
        r
        for r in original["allocations"]
        if r["formation_date"] <= "2018-03-31"
    ]
    second = [
        r for r in changed["allocations"] if r["formation_date"] <= "2018-03-31"
    ]
    assert first == second


def _capacity_fixture():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    frames = [
        pd.DataFrame(
            {
                "me": [100.0, 20_000.0],
                "dolvol_126d": [100.0, 100.0],
                "rvol_21d": [0.02, 0.02],
                "bidaskhl_21d": [0.0, 0.0],
            }
        )
    ]
    market = DailyMarket(
        dates,
        np.zeros((6, 2)),
        np.ones((6, 2), dtype=bool),
        np.zeros(6, dtype=int),
        [],
        frames,
    )
    settings = {
        **_settings(),
        "capacity_execution": True,
        "strategy_capital_usd": 10_000,
        "fund_capital_usd": 10_000,
        "large_cap_borrow_threshold_usd": 10_000_000_000,
        "large_cap_borrow_bps": 0,
        "other_borrow_bps": 0,
        "max_borrow_fraction_market_cap": 0.001,
        "max_borrow_adv_days": 5,
        "order_ttl_sessions": 10,
        "execution_no_trade_band": 0,
        "max_daily_participation": 0.1,
        "impact_eta": 0,
        "execution_gross_buffer": 1,
        "universe_size": 500,
    }
    return market, settings


def test_capacity_fills_over_days_and_only_filled_shares_earn_returns():
    market, settings = _capacity_fixture()
    market.returns[2, 0] = 0.1
    book = replay_daily(
        market, {1: np.array([0.04, -0.04])}, settings, keep_holdings=True
    )
    assert book["holdings"][1, 0] == pytest.approx(0.001)
    assert book["net"][2] == pytest.approx(0.0001)
    assert book["holdings"][3, 0] > book["holdings"][1, 0]
    assert book["execution"]["max_daily_participation"] <= 0.1 + 1e-10
    assert book["execution"]["pending_notional_at_end_usd"] > 0


def test_capacity_rejects_microcap_short_and_charges_market_impact():
    market, settings = _capacity_fixture()
    schedule = {1: np.array([-0.04, 0.04])}
    free = replay_daily(market, schedule, settings, keep_holdings=True)
    costly = replay_daily(market, schedule, {**settings, "impact_eta": 0.5})
    assert np.all(free["holdings"][:, 0] == 0)
    assert free["execution"]["rejected_ineligible_short_observations"] > 0
    assert costly["execution"]["transaction_cost_nav_fractions_sum"] > 0
    assert (
        costly["execution"]["final_nav_usd"]
        < free["execution"]["final_nav_usd"]
    )


def test_expired_unfilled_entry_never_becomes_a_holding():
    market, settings = _capacity_fixture()
    book = replay_daily(
        market,
        {1: np.array([0.04, -0.04])},
        {**settings, "order_ttl_sessions": 1},
        keep_holdings=True,
    )
    np.testing.assert_allclose(
        book["holdings"][2:], np.repeat(book["holdings"][1:2], 4, axis=0)
    )
    assert book["execution"]["expired_entry_order_observations"] > 0


def test_completed_order_does_not_force_daily_rebalancing_after_price_moves():
    market, settings = _capacity_fixture()
    market.frames[0]["dolvol_126d"] = 1_000_000
    market.returns[2, 0] = 0.1
    book = replay_daily(
        market, {1: np.array([0.04, -0.04])}, settings, keep_holdings=True
    )
    assert book["net"][2] == pytest.approx(0.004)
    assert book["holdings"][2, 0] == pytest.approx(0.044 / 1.004)
    assert book["turnover"][2] == pytest.approx(0, abs=1e-12)
    assert book["execution"]["final_nav_usd"] == pytest.approx(
        settings["strategy_capital_usd"] * np.prod(1 + book["net"])
    )


def test_missing_quote_exit_cancels_the_old_order_until_a_new_decision():
    market, settings = _capacity_fixture()
    market.frames[0]["dolvol_126d"] = 1_000_000
    market.tradable[2, 0] = False
    market.returns[2, 0] = np.nan
    market.returns[3, 0] = 0.2
    book = replay_daily(
        market, {1: np.array([0.04, -0.04])}, settings, keep_holdings=True
    )
    assert np.all(book["holdings"][2:, 0] == 0)
    assert book["net"][3] == 0


def test_empty_pods_produce_cash_without_manufacturing_a_strategy():
    panel = SimpleNamespace(
        returns=np.zeros((4, 2)), mask=np.ones((5, 2), dtype=bool)
    )
    result = allocate(
        {"a": [], "b": []}, panel, _settings(), {"method": "equal"}
    )
    assert not result["allocations"]
    assert not np.any(result["weights"])
    assert result["metrics"]["sharpe_annualized"] is None


def test_weekly_review_uses_last_session_before_a_friday_holiday(monkeypatch):
    from pod_platform.research import training_tactical

    dates = pd.to_datetime(["2020-12-23", "2020-12-24", "2020-12-28"])
    names = 20
    design = np.ones((names, 1))
    market = DailyMarket(
        dates,
        np.zeros((3, names)),
        np.ones((3, names), dtype=bool),
        np.zeros(3, dtype=int),
        [(np.ones(names, dtype=bool), design, np.linalg.pinv(design))],
    )
    panel = SimpleNamespace(
        mask=np.ones((1, names), dtype=bool), securities=np.arange(names)
    )
    scores = np.tile(np.linspace(-1, 1, names), (3, 1))
    monkeypatch.setattr(
        training_tactical, "tactical_signals", lambda *args: scores.copy()
    )
    recipe = {
        "family": "reversal",
        "lookback_sessions": 5,
        "expected_return_sign": 1,
        "review": "weekly",
        "entry_rank": 0,
        "exit_rank": 0,
    }
    schedule = tactical_schedule(panel, market, _settings(), recipe)
    assert set(schedule) == {2}


def test_entry_and_pending_fills_are_not_filtered_as_small_resizes():
    market, settings = _capacity_fixture()
    book = replay_daily(
        market,
        {1: np.array([0.04, -0.04])},
        {**settings, "execution_no_trade_band": 0.1},
        keep_holdings=True,
    )
    assert book["holdings"][1, 0] > 0
    assert book["holdings"][3, 0] > book["holdings"][1, 0]


def test_cio_nets_raw_targets_before_applying_fund_capacity_once():
    market, settings = _capacity_fixture()
    market.dates = pd.date_range("2020-01-31", periods=6, freq="B")
    market.frames[0]["beta_60m"] = 1.0
    market.frames[0]["gics"] = "10"
    design = np.ones((2, 1))
    market.designs = [(np.ones(2, dtype=bool), design, np.linalg.pinv(design))]
    panel = SimpleNamespace(
        dates=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        execution_dates=pd.to_datetime(["2020-02-03"]),
        securities=np.array([1, 2]),
        frames=market.frames,
    )
    weights = np.tile([0.04, -0.04], (2, 1))
    first = SimpleNamespace(strategy_id="first", recipe={}, weights=weights)
    second = SimpleNamespace(
        strategy_id="second", recipe={}, weights=weights.copy()
    )
    allocations = [
        {
            "formation_date": "2020-01-31",
            "strategy_id": identity,
            "capital_weight": 0.5,
        }
        for identity in ("first", "second")
    ]
    selected = {"a": [first], "b": [second]}
    original = combined_schedule(selected, allocations, panel, market, settings)
    changed = combined_schedule(
        selected,
        allocations,
        panel,
        market,
        {**settings, "strategy_capital_usd": 100_000_000},
    )
    np.testing.assert_allclose(original[1], [0.04, -0.04])
    np.testing.assert_array_equal(original[1], changed[1])
    second.weights = -weights
    offset = combined_schedule(selected, allocations, panel, market, settings)
    np.testing.assert_allclose(offset[1], 0, atol=1e-10)


def test_cio_targets_cannot_use_the_execution_days_return(monkeypatch):
    from pod_platform.research import training_tactical

    market, settings = _capacity_fixture()
    market.dates = pd.date_range("2020-01-31", periods=6, freq="B")
    market.frames[0]["beta_60m"] = 1.0
    market.frames[0]["gics"] = "10"
    design = np.ones((2, 1))
    market.designs = [(np.ones(2, dtype=bool), design, np.linalg.pinv(design))]
    panel = SimpleNamespace(
        dates=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        execution_dates=pd.to_datetime(["2020-02-03"]),
        securities=np.array([1, 2]),
        frames=market.frames,
    )
    first = SimpleNamespace(strategy_id="first")
    second = SimpleNamespace(strategy_id="second")
    schedules = {
        "first": {1: np.array([0.04, -0.04])},
        "second": {3: np.zeros(2)},
    }
    monkeypatch.setattr(
        training_tactical,
        "schedule_for",
        lambda rule, *args: schedules[rule.strategy_id],
    )
    allocations = [
        {
            "formation_date": "2020-01-31",
            "strategy_id": identity,
            "capital_weight": 0.5,
        }
        for identity in schedules
    ]
    selected = {"a": [first], "b": [second]}
    original = combined_schedule(selected, allocations, panel, market, settings)
    market.returns[3] = [0.5, -0.5]
    changed = combined_schedule(selected, allocations, panel, market, settings)
    np.testing.assert_array_equal(original[3], changed[3])
    market.returns[2] = [0.05, -0.05]
    earlier = combined_schedule(selected, allocations, panel, market, settings)
    assert not np.array_equal(original[3], earlier[3])


def test_risk_clipping_partial_exits_cannot_open_more_than_the_position_cap():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "me": [20_000.0] * 3,
            "dolvol_126d": [1_000_000, 1_000_000, 200],
            "rvol_21d": [0.02] * 3,
            "bidaskhl_21d": [0.0] * 3,
        }
    )
    market = DailyMarket(
        dates,
        np.zeros((6, 3)),
        np.ones((6, 3), dtype=bool),
        np.zeros(6, dtype=int),
        [],
        [frame],
    )
    _, settings = _capacity_fixture()
    settings = {
        **settings,
        "universe_size": 2,
        "execution_max_absolute_net": 0.01,
    }
    book = replay_daily(
        market,
        {1: np.array([0.03, -0.03, 0]), 2: np.array([0, -0.03, 0.03])},
        settings,
        keep_holdings=True,
    )
    counts = np.count_nonzero(np.abs(book["holdings"]) > 1e-10, axis=1)
    assert counts.max() <= 2


def test_optional_size_control_preserves_sector_and_market_beta_neutrality():
    rng = np.random.default_rng(502)
    names = 100
    frame = pd.DataFrame(
        {
            "gics": ["10"] * 50 + ["20"] * 50,
            "me": np.exp(rng.uniform(np.log(1000), np.log(20_000), names)),
            "beta_60m": rng.uniform(0.5, 1.5, names),
            "dolvol_126d": [20_000_000.0] * names,
        }
    )
    panel = SimpleNamespace(
        frames=[frame], mask=np.ones((1, names), dtype=bool)
    )
    signal = np.log(frame.me.to_numpy())[None, :]
    included = make_weights(signal, risk_designs(panel), _settings())
    settings = {**_settings(), "neutralize_size": False}
    allowed = make_weights(signal, risk_designs(panel, settings), settings)
    assert np.abs(included).sum() < 1e-8
    assert np.abs(allowed).sum() > 0
    assert abs(allowed[0] @ frame.beta_60m.to_numpy()) < 1e-7
    assert abs(allowed[0, :50].sum()) < 1e-7
    assert abs(allowed[0, 50:].sum()) < 1e-7


def test_cash_after_start_is_reported_as_a_position_floor_breach():
    market, settings = _capacity_fixture()
    market.frames[0]["dolvol_126d"] = 1_000_000
    book = replay_daily(
        market,
        {1: np.array([0.04, -0.04]), 2: np.zeros(2)},
        {**settings, "min_positions": 2},
    )
    assert book["execution"]["position_count_breach_sessions"] == 4


def test_cost_sizing_cannot_replace_short_candidates_with_other_long_candidates():
    values = np.concatenate(
        [np.linspace(0.5, 1.0, 500), np.linspace(-0.01, -0.1, 500)]
    )
    control = np.ones((1000, 1))
    weights = project_target(
        values, control, {**_settings(), "universe_size": 500}, normalize=True
    )
    assert np.count_nonzero(weights[:500] > 0) == 250
    assert np.count_nonzero(weights[500:] < 0) == 250
    assert not np.any(weights[:500] < 0)
    assert not np.any(weights[500:] > 0)
    assert abs(weights.sum()) < 1e-9


def test_boosted_forecasts_do_not_use_unsettled_future_returns():
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(512)
    count, names = 9, 80
    dates = pd.date_range("2017-01-31", periods=count, freq="ME")
    features = {
        "x": rng.uniform(-1, 1, (count, names)),
        "y": rng.uniform(-1, 1, (count, names)),
    }
    frames = [
        pd.DataFrame(
            {
                "gics": ["10"] * 40 + ["20"] * 40,
                "beta_60m": rng.uniform(0.5, 1.5, names),
                "me": rng.uniform(1000, 20_000, names),
            }
        )
        for _ in dates
    ]
    panel = SimpleNamespace(
        dates=dates,
        frames=frames,
        features=features,
        mask=np.ones((count, names), dtype=bool),
        returns=0.02 * features["x"][:-1]
        + rng.normal(0, 0.001, (count - 1, names)),
        settlement_dates=dates[1:],
        entry_mask=np.ones((count - 1, names), dtype=bool),
    )
    settings = {
        "gbdt_min_history": 2,
        "gbdt_rounds": 5,
        "gbdt_learning_rate": 0.2,
        "gbdt_min_child_weight": 2,
        "gbdt_l2": 2,
    }
    before, _ = annual_boosted(panel, ["x", "y"], 2, 24, settings)
    panel.returns[5:] = rng.normal(0, 10, (count - 6, names))
    after, _ = annual_boosted(panel, ["x", "y"], 2, 24, settings)
    np.testing.assert_array_equal(before[:6], after[:6])
    assert not np.any(before[:2])
    assert np.any(before[2:])


def test_tiny_targets_cannot_reappear_as_extra_positions_after_fees():
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    frame = pd.DataFrame(
        {
            "me": [20_000.0] * 3,
            "dolvol_126d": [1_000_000.0] * 3,
            "rvol_21d": [0.02] * 3,
            "bidaskhl_21d": [0.0] * 3,
        }
    )
    market = DailyMarket(
        dates,
        np.zeros((4, 3)),
        np.ones((4, 3), dtype=bool),
        np.zeros(4, dtype=int),
        [],
        [frame],
    )
    _, settings = _capacity_fixture()
    settings = {**settings, "universe_size": 2, "cost_bps_one_way": 10}
    book = replay_daily(
        market,
        {1: np.array([0.04, -0.04, 0.99999e-10])},
        settings,
        keep_holdings=True,
    )
    assert np.count_nonzero(book["holdings"][1]) == 2
    assert book["execution"]["maximum_positions"] <= 2


def test_an_observed_total_loss_is_not_erased_by_a_missing_execution_quote():
    market, settings = _capacity_fixture()
    market.frames[0]["dolvol_126d"] = 1_000_000
    market.returns[2, 0] = -1.0
    market.tradable[2, 0] = False
    book = replay_daily(market, {1: np.array([0.04, -0.04])}, settings)
    assert book["net"][2] == pytest.approx(-0.04)
