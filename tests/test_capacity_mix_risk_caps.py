"""Synthetic checks for the risk caps and the selection-noise helper."""
import numpy as np
import pandas as pd
import pytest

from src.capacity_mix.risk_caps import apply_caps
from src.capacity_mix.sanity import selection_noise, sharpe


def _targets() -> pd.DataFrame:
    month = pd.Period("2021-01", "M")
    return pd.DataFrame({
        "target_month": [month] * 6,
        "permno": [1, 2, 3, 4, 5, 6],
        "target_weight": [0.05, 0.01, -0.04, -0.01, 0.03, -0.03],
        "size_grp": ["large", "large", "micro", "micro", "nano", "small"],
        "me": [50_000.0, 50_000.0, 100.0, 100.0, 20.0, 2_000.0],  # $ millions
        "sig": [0.25, 1.00, 0.50, 2.00, 1.00, 0.50],  # annualised volatility
    })


def test_name_cap_clips_each_side_and_leaves_smaller_weights():
    capped = apply_caps(_targets(), nav=100e6, name=(0.02, 0.015))
    assert capped["target_weight"].tolist() == [0.02, 0.01, -0.015, -0.01, 0.02, -0.015]


def test_vol_scaling_only_shrinks_names_more_volatile_than_the_reference():
    capped = apply_caps(_targets(), nav=100e6, vol_ref=0.5, vol=True)
    expected = np.array([0.05, 0.005, -0.04, -0.0025, 0.015, -0.03])
    assert np.allclose(capped["target_weight"], expected)


def test_bucket_cap_limits_micro_gross_and_leaves_other_names_alone():
    targets = _targets()
    capped = apply_caps(targets, nav=100e6, bucket=0.04)
    micro = capped["size_grp"].isin(["nano", "micro"])
    assert capped.loc[micro, "target_weight"].abs().sum() == pytest.approx(0.04)
    assert (capped.loc[~micro, "target_weight"] == targets.loc[~micro, "target_weight"]).all()
    assert np.sign(capped["target_weight"]).tolist() == np.sign(targets["target_weight"]).tolist()


def test_bucket_cap_is_a_no_op_when_the_group_is_already_inside_the_budget():
    targets = _targets()
    capped = apply_caps(targets, nav=100e6, bucket=1.0)
    pd.testing.assert_series_equal(capped["target_weight"], targets["target_weight"])


def test_market_cap_share_limits_positions_by_company_size():
    capped = apply_caps(_targets(), nav=100e6, mcap=(0.02, 0.01))
    # micro short: 1% of $100m market cap = $1m = 1% of a $100m NAV; nano long: 2% of $20m = $0.4m = 0.4% of NAV
    assert capped.loc[capped["permno"] == 3, "target_weight"].iloc[0] == pytest.approx(-0.01)
    assert capped.loc[capped["permno"] == 5, "target_weight"].iloc[0] == pytest.approx(0.004)


def test_missing_market_cap_does_not_erase_or_nan_the_position():
    targets = _targets()
    targets.loc[0, "me"] = np.nan
    capped = apply_caps(targets, nav=100e6, mcap=(0.02, 0.01))
    assert capped.loc[0, "target_weight"] == pytest.approx(0.05)
    assert capped["target_weight"].notna().all()


def test_caps_never_grow_a_position_flip_its_sign_or_change_the_inputs():
    targets = _targets()
    before = targets.copy()
    capped = apply_caps(targets, nav=100e6, vol_ref=0.5, name=(0.02, 0.015), vol=True, bucket=0.03, mcap=(0.02, 0.01))
    assert (capped["target_weight"].abs() <= targets["target_weight"].abs() + 1e-12).all()
    assert (np.sign(capped["target_weight"]) * np.sign(targets["target_weight"]) >= 0).all()
    pd.testing.assert_frame_equal(targets, before)


def test_selection_noise_grows_with_trials_and_shrinks_with_history():
    short = selection_noise(68, trials=(10, 100))
    long_ = selection_noise(136, trials=(10, 100))
    assert short["expected_best_sharpe_from_noise"].is_monotonic_increasing
    assert short["sharpe_std_error"].iloc[0] == pytest.approx(1 / np.sqrt(68 / 12))
    assert (long_["sharpe_std_error"] < short["sharpe_std_error"]).all()


def test_sharpe_is_excess_of_the_risk_free_rate():
    index = pd.period_range("2021-01", periods=24, freq="M")
    market = pd.DataFrame({"benchmark": 0.04 / 12 + 0.002}, index=index)  # risk-free = 0.2% a month
    returns = pd.Series(0.01 + np.tile([0.01, -0.01], 12), index=index)
    excess = returns - 0.002
    assert sharpe(returns, market) == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(12))
