import numpy as np
import pandas as pd

from src.time_series_search import (
    account_selected_book, allocate_quotas, calendar_lag, constrain_weights,
    metrics, select_validation, summarize, weights_from_score, atomic_json,
    ridge_forecasts, SNAPSHOT,
    simulate,
)
from src.graph.walk_forward import split_masks


def test_calendar_lags_preserve_gaps_and_quarter_rollovers():
    f = pd.DataFrame({"permno": [1, 1, 1, 1],
        "eom": pd.to_datetime(["2020-01-31", "2020-03-31", "2020-04-30", "2020-05-31"]),
        "target_fpe": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-06-30", "2020-06-30"]),
        "estimate": [1., 2., 5., 6.]})
    lag = calendar_lag(f, ["estimate"], 1, same_quarter=True).estimate
    assert lag.iloc[:3].isna().all()
    assert lag.iloc[3] == 5


def test_label_announced_in_test_cannot_shape_training_or_validation():
    f = pd.to_datetime(["2018-11-30", "2020-11-30"])
    masks = split_masks(f, 1, 2021, label_dates=pd.to_datetime(["2019-01-01", "2021-01-01"]))
    assert not masks["train"].any()
    assert not masks["validation"].any()


def test_sector_seats_and_hold_union_obey_position_and_exposure_limits():
    n = 800
    sectors = np.tile(np.arange(4), n // 4)
    scores = np.arange(n, dtype=float)
    controls = np.column_stack([np.sin(scores), np.cos(scores)])
    w = weights_from_score(scores, np.ones(n, bool), sectors, controls, "raw")
    assert (w != 0).sum() == 100
    # A six-cohort union would otherwise exceed the competition ceiling.
    union = sum(np.roll(w, j * 100) for j in range(6))
    bounded = constrain_weights(union, sectors, vol=np.linspace(.05, .2, n))
    assert 100 <= (bounded != 0).sum() <= 500
    assert np.isclose(np.abs(bounded).sum(), 2)
    assert np.isclose(bounded.sum(), 0)
    for sector in range(4):
        assert np.isclose(bounded[sectors == sector].sum(), 0)
    assert (allocate_quotas([1, 10, 100], 50) <= [1, 10, 100]).all()


def sample_book(dates, value=.01):
    n = len(dates)
    return pd.DataFrame({"date": dates, "gross_excess": value, "net_excess": value,
        "cash": .002, "turnover": 0., "cost": 0., "n_long": 50, "n_short": 50,
        "gross_exposure": 2., "net_exposure": 0., "missing_held_returns": 0,
        "long_pnl": .02, "short_pnl": -.01, "max_position": .02,
        "market_excess": np.linspace(-.02, .02, n), "short_me": 100.,
        "turnover_uncertain": False})


def test_horizon_selection_never_reads_test_year_and_rejects_missing_validation():
    dates = pd.date_range("2019-01-31", "2021-12-31", freq="ME")
    a, b = sample_book(dates), sample_book(dates)
    a["net_excess"] = np.resize([.005, .015], len(a))
    b["net_excess"] = np.resize([.006, .014], len(b))
    a.loc[a.date.dt.year == 2021, "net_excess"] = 100
    selected, _ = select_validation({1: a, 3: b}, 2021)
    assert selected == 3
    b.loc[0, "net_excess"] = np.nan
    selected, _ = select_validation({1: a, 3: b}, 2021)
    assert selected == 1


def test_selected_path_costs_liquidation_and_does_not_zero_unknown_return():
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    b = sample_book(dates)
    b.loc[1:, ["n_long", "n_short", "gross_exposure"]] = 0
    b.loc[1:, "gross_excess"] = 0
    class M:
        ids = np.array([1, 2])
        months = dates
        returns = np.zeros((3, 2))
        forward_returns = np.zeros((3, 2))
    w = np.array([[1, -1], [0, 0], [0, 0]], dtype=float)
    result = account_selected_book(M(), b, w, 25)
    assert np.isclose(result.loc[0, "cost"], .005)
    assert result.loc[1, "cost"] > 0  # cash exit must incur a liquidation cost
    result.loc[1, "net_excess"] = np.nan
    s = metrics(result)
    assert s["missing_return_months"] == 1
    assert s["total_cagr"] is None
    assert s["max_drawdown"] is None


def test_cash_only_ir_is_undefined_and_report_handles_it(tmp_path):
    b = sample_book(pd.date_range("2021-01-31", periods=12, freq="ME"), value=0)
    b[["n_long", "n_short", "gross_exposure"]] = 0
    s = metrics(b)
    assert s["information_ratio"] is None
    (tmp_path / "results").mkdir()
    atomic_json(tmp_path / "results/01_large_mega_raw.json", {
        "experiment": "01", "name": "cash_only", "universe": "large_mega", "mode": "raw", "summary": s})
    summarize(tmp_path)
    assert "undefined" in (tmp_path / "report.md").read_text()


def test_late_validation_labels_do_not_remove_historical_model_forecasts(tmp_path):
    rng = np.random.default_rng(11)
    index = pd.MultiIndex.from_product([pd.date_range("2015-01-31", "2026-07-31", freq="ME"), range(20)], names=["eom", "permno"])
    p = index.to_frame(index=False)
    for col in SNAPSHOT:
        p[col] = rng.normal(size=len(p))
    p["target_surprise_p"] = rng.normal(size=len(p))
    p["target_ann"] = p.eom + pd.Timedelta(days=75)
    forecasts = ridge_forecasts(p, "31", tmp_path)
    # These labels were announced in the test year and could not tune the 2021 model.
    # Their December-2020 formation rows must still be forecast, even though they are test rows.
    # November formation predicts December holding month (validation), label arrives in February.
    last_validation = forecasts[(forecasts.fold == 2021) & (forecasts.eom == pd.Timestamp("2020-11-30"))]
    assert len(last_validation) == 20
    assert (last_validation.window == "validation").all()
    assert last_validation.prediction.notna().all()


def test_pnl_uses_provided_forward_outcome_without_requiring_next_characteristic_row():
    class M:
        ids = np.arange(100)
        months = pd.date_range("2020-01-31", periods=3, freq="ME")
        returns = np.full((3, 100), np.nan)  # no next-month feature observations
        forward_returns = np.tile(np.linspace(-.1, .1, 100), (3, 1))
        eligible = np.ones((3, 100), bool)
        sectors = np.zeros((3, 100), int)
        controls = np.zeros((3, 100, 1))
        quarters = np.zeros((3, 100))
        beta = np.ones((3, 100))
        market = np.zeros(3)
        cash = np.zeros(3)
        wide_me = np.ones((3, 100)) * 100
    m = M()
    scores = np.tile(np.arange(100, dtype=float), (3, 1))
    b = simulate(m, scores, 1, "raw", "01", 25)
    assert (b.missing_held_returns == 0).all()
    assert b.gross_excess.notna().all()
    # Unknown outcomes cannot silently disappear from the selected book.
    m.forward_returns = m.forward_returns.copy()
    m.forward_returns[0, 0] = np.nan
    b = simulate(m, scores, 1, "raw", "01", 25)
    assert b.loc[0, "missing_held_returns"] == 1
    assert pd.isna(b.loc[0, "gross_excess"])


def test_turnover_drifts_positions_with_total_stock_returns_and_cash_nav():
    dates = pd.date_range('2021-01-31',periods=2,freq='ME')
    b=sample_book(dates,value=.2)
    b['cash']=.01
    class M:
        ids=np.array([1,2])
        months=pd.date_range('2020-12-31',periods=3,freq='ME')
        forward_returns=np.tile([.1,-.1],(3,1))
    w=np.array([[1.,-1.],[1.,-1.]])
    result=account_selected_book(M(),b,w,25)
    prior=np.array([1.11,-.91])/(1+.01+.2-.005)
    assert np.isclose(result.turnover.iloc[1],np.abs(w[1]-prior).sum())
