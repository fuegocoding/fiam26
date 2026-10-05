"""Validation selection and explicit liquidation regression checks."""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "final"))
import evaluate
import optimizer
import risk


def test_liquidation_prices_and_exit_are_in_missing_month_only():
    months = pd.period_range("2021-01", "2021-02", freq="M")
    u = pd.DataFrame(
        [
            {
                "return_month": m,
                "permno": 1,
                "ret_exc_lead1m": np.nan if i == 0 else 0.1,
                "prc": 10.0,
                "exit_price": 8.0,
                "exit_price_date": pd.Timestamp("2021-01-15"),
                "size_grp": "large",
                "sector": "45",
                "beta_60m": 1.0,
            }
            for i, m in enumerate(months)
        ]
    )
    market = pd.DataFrame({"rf": 0.001, "spx": 0.01}, index=months)
    r = evaluate.evaluate({months[0]: {1: 0.5}, months[1]: {1: 0.5}}, u, market)
    assert np.isclose(r.pnl.iloc[0], 0.5 * (-0.2 - 0.001))
    assert np.isclose(r.traded.iloc[0], 0.5 + 0.4)
    # The closed position must be bought afresh, with no fictitious drift.
    assert np.isclose(r.traded.iloc[1], 0.5)
    exit = r.attrs["liquidations"].iloc[0]
    assert exit.return_month == months[0] and exit.exit_price == 8
    assert np.isclose(exit.exit_cost, 0.4 * 0.001)

def test_missing_return_without_quote_fails_instead_of_filling_zero():
    with pytest.raises(ValueError, match="Liquidation requires"):
        evaluate.position_returns(pd.DataFrame({"ret_exc_lead1m": [np.nan]}), 0.001)

def test_quote_lookup_ignores_later_price(tmp_path):
    columns = ["prc", "dolvol_126d", "me", "beta_60m", "be_me", "ret_12_1"]
    frame = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "eom": pd.to_datetime(["2020-12-31", "2021-01-31", "2021-02-28"]),
            "ret_exc_lead1m": [np.nan, 0.01, 0.02],
            "size_grp": ["large"] * 3,
            "gics": ["451010"] * 3,
            **{c: [10.0, 8.0, 1000.0] for c in columns},
        }
    )
    frame["eom"] = frame.eom.astype("datetime64[us]")
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path)
    u = evaluate.load_universe(path)
    jan = u.loc[u.return_month.eq(pd.Period("2021-01", "M"))].iloc[0]
    assert jan.exit_price == 8 and jan.exit_price_date == pd.Timestamp("2021-01-31")
    frame.loc[2, "prc"] = -99999.0
    frame.to_parquet(path)
    updated = evaluate.load_universe(path)
    jan_after = updated.loc[updated.return_month.eq(pd.Period("2021-01", "M"))].iloc[0]
    assert jan_after.exit_price == jan.exit_price

def test_liquidated_holding_is_not_kept_in_next_retention_buffer(tmp_path, monkeypatch):
    import timing

    months = pd.to_datetime(["2021-01-31", "2021-02-28"])
    predictions = pd.DataFrame(
        [
            {
                "return_month": m,
                "formation_date": m - pd.offsets.MonthEnd(1),
                "timing_policy": timing.PREDICTION_POLICY,
                "permno": p,
                "final_score": float(p),
                "ticker": f"T{p}",
                "company_name": f"C{p}",
                "ticker_name_reference_date": m - pd.offsets.MonthEnd(1),
                "ticker_name_source": "test",
                "ticker_name_status": "verified",
                "prc": 10.0,
                "size_grp": "large",
                "dolvol_126d": 1e8,
                "sector": "45",
                "beta_60m": 1.0,
            }
            for m in months
            for p in (1, 2)
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "permno": p,
                "eom": m,
                "gics": "451010",
                "size_grp": "large",
                "ret_exc": 0.01,
                "rmax1_21d": 0.02,
                "ret_exc_lead1m": np.nan
                if p == 1 and m == pd.Timestamp("2020-12-31")
                else 0.01,
            }
            for m in pd.to_datetime(["2020-11-30", "2020-12-31", "2021-01-31"])
            for p in (1, 2)
        ]
    )
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)
    captured = []

    def book(current, nav, prev_long, prev_short):
        captured.append((set(prev_long), set(prev_short)))
        return {1: 0.5, 2: -0.5}, {}

    monkeypatch.setattr(optimizer, "_book", book)
    monkeypatch.setattr(
        risk,
        "protect",
        lambda weights, *args: (
            weights,
            pd.DataFrame({"risk_multiplier": 1.0}, index=list(weights)),
            {},
        ),
    )
    optimizer.optimize(predictions, path)
    assert captured == [(set(), set()), (set(), {2})]

def test_drawdown_includes_starting_nav():
    months = pd.period_range("2021-01", periods=2, freq="M")
    r = pd.DataFrame(
        {
            "total": [-0.1, 0.05],
            "rf": 0.0,
            "active": [-0.1 - 0.04 / 12, 0.05 - 0.04 / 12],
            "traded": 0.0,
            "cost": 0.0,
            "borrow": 0.0,
            "spx": [0.01, 0.02],
        },
        index=months,
    )
    assert np.isclose(evaluate.metrics(r)["max_dd"], -0.1)

def test_prediction_guard_rejects_targets_past_requested_endpoint(monkeypatch):
    import config
    import timing

    monkeypatch.setattr(config, "LAST_TARGET", pd.Timestamp("2021-01-31"))
    p = pd.DataFrame(
        {
            "permno": [1],
            "formation_date": [pd.Timestamp("2021-01-31")],
            "return_month": [pd.Timestamp("2021-02-28")],
            "ticker": ["A"],
            "company_name": ["A Inc"],
            "ticker_name_reference_date": [pd.Timestamp("2021-01-31")],
            "timing_policy": [timing.PREDICTION_POLICY],
        }
    )
    with pytest.raises(ValueError, match="configured evaluation window"):
        timing.validate_predictions(p)
