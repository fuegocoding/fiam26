import numpy as np
import pandas as pd
import pytest

from src.timesfm_forecast import build_contexts, forecast_month, load_returns


def returns_panel():
    months = pd.date_range("2019-01-31", "2020-01-31", freq="ME")
    return pd.DataFrame({"permno": 1, "eom": months, "ret_exc": np.arange(13, dtype=float) / 100,
                         "ret_exc_lead1m": np.arange(13, dtype=float) / 100})


def test_unknown_future_target_does_not_remove_forecast_or_enter_context(tmp_path):
    p = returns_panel()
    p.loc[p.eom == "2019-12-31", "ret_exc_lead1m"] = np.nan
    p.loc[p.eom == "2020-01-31", "ret_exc"] = 999
    path = tmp_path / "returns.parquet"
    p.to_parquet(path)
    loaded = load_returns(path)
    feat, contexts = build_contexts(loaded, pd.Timestamp("2019-12-31"))
    assert len(feat) == 1
    assert feat.ret_exc_lead1m.isna().all()
    assert len(contexts[0]) == 12
    assert np.isclose(contexts[0][-1], .11)
    assert 999 not in contexts[0]


def test_calendar_gap_cannot_be_compressed_into_regular_monthly_history():
    p = returns_panel().loc[lambda f: f.eom.ne(pd.Timestamp("2019-06-30"))]
    feat, contexts = build_contexts(p, pd.Timestamp("2019-12-31"))
    assert feat.empty
    assert contexts == []


def test_old_target_only_export_is_rejected(tmp_path):
    path = tmp_path / "old.parquet"
    returns_panel().drop(columns="ret_exc").to_parquet(path)
    with pytest.raises(ValueError, match="full formation universe"):
        load_returns(path)


def test_forecast_output_includes_stock_with_unknown_future_outcome():
    p = returns_panel()
    p.loc[p.eom == "2019-12-31", "ret_exc_lead1m"] = np.nan
    class Output:
        forecast = np.array([.01])
        quantiles = np.array([np.linspace(-.1, .1, 9)])
    class Forecaster:
        def predict_batch(self, contexts, **kwargs):
            assert len(contexts[0]) == 12
            return [Output()]
    forecast = forecast_month(Forecaster(), p, pd.Timestamp("2019-12-31"))
    assert len(forecast) == 1
    assert forecast.ret_exc_lead1m.isna().all()
    assert forecast.tfm_forecast.iloc[0] == pytest.approx(.01)
