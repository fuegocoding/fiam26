"""Hand-checkable research metrics and incomplete-evidence boundaries."""

import json
import math
from statistics import NormalDist

import numpy as np
import pytest

from pod_platform.research.desk_metrics import compute_desk_metrics

RETURNS = [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03]


def test_psr_dsr_and_track_record_match_published_formulas():
    metrics = compute_desk_metrics(
        RETURNS,
        periods_per_year=12,
        trial_sharpes=[-0.3, -0.1, 0.1, 0.3],
        total_attempts=4,
        trial_periods_per_year=12,
        attempts_complete=True,
    )
    # This symmetric six-point fixture has SR=1/sqrt(14), skew=0,
    # Pearson kurtosis=303/175 and trial sample variance=1/15.
    normal = NormalDist()
    sr = 1 / math.sqrt(14)
    kurtosis = 303 / 175
    variance_factor = 1 + (kurtosis - 1) * sr**2 / 4
    gamma = 0.5772156649015329
    sr0 = math.sqrt(1 / 15) * (
        (1 - gamma) * normal.inv_cdf(3 / 4)
        + gamma * normal.inv_cdf(1 - 1 / (4 * math.e))
    )
    denominator = math.sqrt(variance_factor)
    expected_psr = normal.cdf(sr * math.sqrt(5) / denominator)
    expected_dsr = normal.cdf((sr - sr0) * math.sqrt(5) / denominator)
    assert metrics["sharpe_periodic"] == pytest.approx(sr)
    assert metrics["sharpe_annualized"] == pytest.approx(math.sqrt(6 / 7))
    assert metrics["skewness"] == pytest.approx(0, abs=1e-14)
    assert metrics["kurtosis"] == pytest.approx(kurtosis)
    assert metrics["trial_sharpe_variance"] == pytest.approx(1 / 15)
    assert metrics["expected_maximum_sharpe_periodic"] == pytest.approx(sr0)
    assert metrics["probabilistic_sharpe_ratio"] == pytest.approx(expected_psr)
    assert metrics["deflated_sharpe_ratio"] == pytest.approx(expected_dsr)
    assert metrics["deflated_sharpe_ratio"] == pytest.approx(0.4961041818)
    assert metrics["minimum_track_record_periods"] == 40
    assert metrics["deflated_sharpe_status"] == "available"


def test_compounding_drawdown_tail_and_halves_use_actual_periods():
    metrics = compute_desk_metrics(RETURNS, periods_per_year=12)
    assert metrics["periods"] == 6
    assert metrics["cumulative_net_return"] == pytest.approx(
        math.prod(1 + value for value in RETURNS) - 1
    )
    assert metrics["cagr"] == pytest.approx(0.0598394500545657)
    assert metrics["maximum_drawdown"] == pytest.approx(0.0298)
    assert metrics["return_quantile_05"] == pytest.approx(-0.0175)
    assert metrics["expected_shortfall_05_return"] == -0.02
    assert metrics["downside_deviation_annualized"] == pytest.approx(
        math.sqrt(0.001)
    )
    assert metrics["halves"]["first_mean_return"] == pytest.approx(-0.01)
    assert metrics["halves"]["second_mean_return"] == pytest.approx(0.02)
    assert metrics["halves"]["first_sharpe_annualized"] < 0
    assert metrics["halves"]["second_sharpe_annualized"] > 0


@pytest.mark.parametrize("cash_return", [0.0, 0.001])
def test_constant_cash_has_no_manufactured_sharpe(cash_return):
    metrics = compute_desk_metrics(
        [cash_return] * 24,
        periods_per_year=12,
        trial_sharpes=[-0.1, 0.1],
        total_attempts=2,
        trial_periods_per_year=12,
        attempts_complete=True,
    )
    for name in (
        "sharpe_periodic",
        "sharpe_annualized",
        "skewness",
        "kurtosis",
        "probabilistic_sharpe_ratio",
        "deflated_sharpe_ratio",
        "minimum_track_record_periods",
        "sharpe_uncertainty",
    ):
        assert metrics[name] is None
    assert metrics["volatility_periodic"] == 0
    assert metrics["maximum_drawdown"] == 0
    assert metrics["deflated_sharpe_status"] == "unavailable"
    assert metrics["deflated_sharpe_reason"] == (
        "insufficient_nonconstant_returns"
    )
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        ({}, "complete_attempt_history_required"),
        (
            {"trial_sharpes": [0.2], "total_attempts": 50},
            "complete_attempt_history_required",
        ),
        (
            {
                "trial_sharpes": [0.1, 0.2],
                "total_attempts": 3,
                "attempts_complete": True,
            },
            "missing_trial_sharpes",
        ),
        (
            {
                "trial_sharpes": [0.2],
                "total_attempts": 1,
                "attempts_complete": True,
            },
            "at_least_two_observed_trials_required",
        ),
        (
            {
                "trial_sharpes": [0.1, 0.2],
                "total_attempts": 2,
                "attempts_complete": True,
                "trial_periods_per_year": 252,
            },
            "matching_explicit_trial_frequency_required",
        ),
    ],
)
def test_missing_evidence_never_infers_trial_variance(evidence, reason):
    metrics = compute_desk_metrics(RETURNS, periods_per_year=12, **evidence)
    assert metrics["probabilistic_sharpe_ratio"] is not None
    assert metrics["deflated_sharpe_ratio"] is None
    assert metrics["trial_sharpe_variance"] is None
    assert metrics["expected_maximum_sharpe_periodic"] is None
    assert metrics["deflated_sharpe_status"] == "unavailable"
    assert metrics["deflated_sharpe_reason"] == reason


def test_frequency_changes_annualization_but_not_selection_probability():
    evidence = {
        "trial_sharpes": [-0.3, -0.1, 0.1, 0.3],
        "total_attempts": 4,
        "attempts_complete": True,
    }
    monthly = compute_desk_metrics(
        RETURNS, periods_per_year=12, trial_periods_per_year=12, **evidence
    )
    daily = compute_desk_metrics(
        RETURNS, periods_per_year=252, trial_periods_per_year=252, **evidence
    )
    assert daily["sharpe_annualized"] == pytest.approx(
        monthly["sharpe_annualized"] * math.sqrt(21)
    )
    assert daily["deflated_sharpe_ratio"] == monthly["deflated_sharpe_ratio"]


def test_hac_uncertainty_detects_serial_dependence_in_same_distribution():
    # Ordered and shuffled fixtures have identical sample moments and Sharpe.
    values = 0.0001 + 0.01 * np.sin(np.arange(512) * 2 * math.pi / 64)
    shuffled = np.random.default_rng(731).permutation(values)
    ordered_metrics = compute_desk_metrics(
        values, periods_per_year=252, hac_lags=8
    )
    shuffled_metrics = compute_desk_metrics(
        shuffled, periods_per_year=252, hac_lags=8
    )
    ordered = ordered_metrics["sharpe_uncertainty"]
    independent = shuffled_metrics["sharpe_uncertainty"]
    assert ordered_metrics["sharpe_periodic"] == pytest.approx(
        shuffled_metrics["sharpe_periodic"]
    )
    assert ordered["iid_standard_error_periodic"] == pytest.approx(
        independent["iid_standard_error_periodic"]
    )
    assert ordered["hac_standard_error_periodic"] > (
        2 * independent["hac_standard_error_periodic"]
    )
    lag_zero = compute_desk_metrics(
        values, periods_per_year=252, hac_lags=0
    )["sharpe_uncertainty"]
    assert lag_zero["hac_standard_error_periodic"] == pytest.approx(
        lag_zero["iid_standard_error_periodic"]
    )


def test_nonpositive_skill_has_no_finite_minimum_track_record():
    metrics = compute_desk_metrics(
        [-value for value in RETURNS], periods_per_year=12
    )
    assert metrics["minimum_track_record_periods"] is None


def test_total_loss_is_a_full_drawdown_including_starting_capital():
    metrics = compute_desk_metrics([-1.0], periods_per_year=12)
    assert metrics["maximum_drawdown"] == 1
    assert metrics["cumulative_net_return"] == -1
    assert metrics["cagr"] == -1
    assert metrics["halves"]["first_mean_return"] is None
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    "returns", [[], [float("nan")], [float("inf")], [-1.1]]
)
def test_invalid_returns_are_rejected_instead_of_dropped(returns):
    with pytest.raises(ValueError):
        compute_desk_metrics(returns, periods_per_year=12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"periods_per_year": 0},
        {"periods_per_year": float("inf")},
        {"confidence": 0.5},
        {"hac_lags": 6},
        {"hac_lags": -1},
        {"trial_sharpes": [0.1, float("nan")]},
        {"trial_sharpes": [0.1, 0.2], "total_attempts": 1},
        {"attempts_complete": "yes"},
        {"trial_periods_per_year": float("nan")},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        compute_desk_metrics(RETURNS, **{"periods_per_year": 12, **kwargs})


def test_unmeasured_attempts_raise_the_deflation_hurdle():
    evidence = {
        "trial_sharpes": [-0.3, -0.1, 0.1, 0.3],
        "trial_periods_per_year": 12,
        "attempts_complete": True,
    }
    measured_only = compute_desk_metrics(
        RETURNS, periods_per_year=12, total_attempts=4, **evidence
    )
    with_failures = compute_desk_metrics(
        RETURNS,
        periods_per_year=12,
        total_attempts=10,
        unmeasured_attempts=6,
        **evidence,
    )
    assert with_failures["deflated_sharpe_status"] == "available"
    assert with_failures["trial_sharpe_variance"] == pytest.approx(
        measured_only["trial_sharpe_variance"]
    )
    assert (
        with_failures["expected_maximum_sharpe_periodic"]
        > measured_only["expected_maximum_sharpe_periodic"]
    )
    assert (
        with_failures["deflated_sharpe_ratio"]
        < measured_only["deflated_sharpe_ratio"]
    )
