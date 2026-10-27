"""Performance evidence from actual, equally spaced, net simple returns.

PSR and DSR follow Bailey and Lopez de Prado (2014), equations (1)-(2),
https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf. Minimum track
record inverts the PSR formula for a specified one-sided confidence target.
Sharpe ratios use a zero return hurdle and sample volatility. Trial Sharpe
ratios must use that same definition and periodic frequency, without
annualization. Moments are empirical central moments; kurtosis is Pearson
kurtosis (normal = 3). PSR/DSR assume independent return observations; the
separate Bartlett Newey-West estimate accounts for return dependence through
the Sharpe influence function. Neither estimate establishes future skill.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _sharpe(values: np.ndarray) -> float | None:
    if len(values) < 2 or float(np.ptp(values)) == 0.0:
        return None
    return _finite(float(np.mean(values) / np.std(values, ddof=1)))


def _growth(log_growth: float) -> float | None:
    if log_growth == -math.inf:
        return -1.0
    try:
        return _finite(math.expm1(log_growth))
    except OverflowError:
        return None


def _trial_evidence(
    trial_sharpes: Sequence[float] | None,
    total_attempts: int | None,
    trial_periods_per_year: float | None,
    periods_per_year: float,
    attempts_complete: bool,
    unmeasured_attempts: int = 0,
) -> dict:
    if (
        isinstance(unmeasured_attempts, bool)
        or not isinstance(unmeasured_attempts, int)
        or unmeasured_attempts < 0
    ):
        raise ValueError("unmeasured_attempts must be a nonnegative integer")
    trials = np.asarray(
        [] if trial_sharpes is None else trial_sharpes, dtype=float
    )
    if trials.ndim != 1 or not np.isfinite(trials).all():
        raise ValueError("trial_sharpes must be finite periodic Sharpe ratios")
    if not isinstance(attempts_complete, bool):
        raise ValueError("attempts_complete must be an explicit boolean")
    if trial_periods_per_year is not None and (
        not math.isfinite(trial_periods_per_year) or trial_periods_per_year <= 0
    ):
        raise ValueError("trial_periods_per_year must be finite and positive")
    if total_attempts is not None and (
        isinstance(total_attempts, bool)
        or not isinstance(total_attempts, int)
        or total_attempts < len(trials)
    ):
        raise ValueError("total_attempts must count every observed trial")
    evidence = {
        "observed_trials": len(trials),
        "total_attempts": total_attempts,
        "attempts_complete": attempts_complete,
        "trial_periods_per_year": trial_periods_per_year,
        "trial_sharpe_variance": None,
        "expected_maximum_sharpe_periodic": None,
        "deflated_sharpe_status": "unavailable",
    }
    if not attempts_complete or total_attempts is None:
        reason = "complete_attempt_history_required"
    elif total_attempts != len(trials) + unmeasured_attempts:
        reason = "missing_trial_sharpes"
    elif len(trials) < 2:
        reason = "at_least_two_observed_trials_required"
    elif trial_periods_per_year != periods_per_year:
        reason = "matching_explicit_trial_frequency_required"
    else:
        variance = float(np.var(trials, ddof=1))
        count = total_attempts
        # Count every attempt rather than infer a smaller effective count.
        expected_max = math.sqrt(variance) * (
            (1 - _EULER_GAMMA) * _NORMAL.inv_cdf(1 - 1 / count)
            + _EULER_GAMMA * _NORMAL.inv_cdf(1 - 1 / (count * math.e))
        )
        evidence.update(
            trial_sharpe_variance=variance,
            expected_maximum_sharpe_periodic=expected_max,
            deflated_sharpe_status="available",
        )
        reason = None
    evidence["deflated_sharpe_reason"] = reason
    return evidence


def compute_desk_metrics(
    net_returns: Sequence[float],
    *,
    periods_per_year: float,
    trial_sharpes: Sequence[float] | None = None,
    total_attempts: int | None = None,
    trial_periods_per_year: float | None = None,
    attempts_complete: bool = False,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    hac_lags: int | None = None,
    unmeasured_attempts: int = 0,
) -> dict:
    """Summarizes supplied net returns without reconstructing missing history.

    Args:
        net_returns: Chronological, equally spaced simple returns.
            Missing periods must be resolved by the caller, never dropped.
        periods_per_year: Explicit number of observations in one year.
        trial_sharpes: All observed, nonannualized trial Sharpe ratios.
        total_attempts: Actual number of attempted variants, including failures.
        trial_periods_per_year: Explicit frequency of the trial Sharpe ratios.
        attempts_complete: Whether the entire attempt ledger is represented.
            Missing or undefined trial Sharpe ratios leave DSR unavailable.
        benchmark_sharpe: Periodic Sharpe threshold for PSR and track record.
        confidence: One-sided PSR target and two-sided HAC interval coverage.
        hac_lags: Bartlett lag count; defaults to floor(4*(T/100)^(2/9)).
        unmeasured_attempts: Attempts without a return book. They enlarge
            the DSR trial count but never the trial Sharpe variance.

    Returns:
        JSON-safe evidence dictionary. Drawdown is a positive loss fraction;
        lower-tail return metrics retain their signed returns. Annualized
        Sharpe uses square-root time scaling and is descriptive under serial
        dependence. PSR, DSR and minimum track record use the IID formula;
        dependence-aware uncertainty is separately labeled. The DSR threshold
        assumes zero skill and conservatively counts all attempts as trials.

    Raises:
        ValueError: Inputs are empty, nonfinite, or use invalid parameters.
    """
    values = np.asarray(net_returns, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < -1)
    ):
        raise ValueError("net_returns must be finite simple returns >= -1")
    if not math.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    if not math.isfinite(benchmark_sharpe) or not 0.5 < confidence < 1:
        raise ValueError("finite benchmark and 0.5 < confidence < 1 required")
    count = len(values)
    if hac_lags is None:
        hac_lags = min(count - 1, int(4 * (count / 100) ** (2 / 9)))
    if (
        isinstance(hac_lags, bool)
        or not isinstance(hac_lags, int)
        or not 0 <= hac_lags < count
    ):
        raise ValueError("hac_lags must be an integer between 0 and T-1")
    evidence = _trial_evidence(
        trial_sharpes,
        total_attempts,
        trial_periods_per_year,
        periods_per_year,
        attempts_complete,
        unmeasured_attempts,
    )
    mean = float(np.mean(values))
    volatility = float(np.std(values, ddof=1)) if count > 1 else None
    if float(np.ptp(values)) == 0:
        volatility = 0.0 if count > 1 else None
    sharpe = _sharpe(values)
    scale = math.sqrt(periods_per_year)
    with np.errstate(divide="ignore"):
        log_wealth = np.concatenate(([0.0], np.cumsum(np.log1p(values))))
    peak = np.maximum.accumulate(log_wealth)
    drawdown = float(np.max(-np.expm1(log_wealth - peak)))
    downside = float(np.sqrt(np.mean(np.minimum(values, 0) ** 2)))
    quantile = float(np.quantile(values, 0.05))
    split = count // 2
    first_sr = _sharpe(values[:split])
    second_sr = _sharpe(values[split:])
    result = {
        "periods": count,
        "periods_per_year": periods_per_year,
        "mean_periodic_net_return": mean,
        "annualized_arithmetic_net_return": _finite(mean * periods_per_year),
        "cumulative_net_return": _growth(float(log_wealth[-1])),
        "cagr": _growth(float(log_wealth[-1]) * periods_per_year / count),
        "volatility_periodic": volatility,
        "volatility_annualized": (
            _finite(volatility * scale) if volatility is not None else None
        ),
        "sharpe_periodic": sharpe,
        "sharpe_annualized": sharpe * scale if sharpe is not None else None,
        "skewness": None,
        "kurtosis": None,
        "maximum_drawdown": drawdown,
        "downside_deviation_annualized": _finite(downside * scale),
        "sortino_annualized": mean / downside * scale if downside else None,
        "return_quantile_05": quantile,
        "expected_shortfall_05_return": float(
            np.mean(values[values <= quantile])
        ),
        "worst_period_return": float(np.min(values)),
        "best_period_return": float(np.max(values)),
        "positive_period_fraction": float(np.mean(values > 0)),
        "probabilistic_sharpe_ratio": None,
        "deflated_sharpe_ratio": None,
        "benchmark_sharpe_periodic": benchmark_sharpe,
        "minimum_track_record_periods": None,
        "inference_assumption": "PSR/DSR/track_record_assume_IID_returns",
        "sharpe_uncertainty": None,
        "halves": {
            "first_periods": split,
            "second_periods": count - split,
            "first_sharpe_annualized": (
                first_sr * scale if first_sr is not None else None
            ),
            "second_sharpe_annualized": (
                second_sr * scale if second_sr is not None else None
            ),
            "first_mean_return": (
                float(np.mean(values[:split])) if split else None
            ),
            "second_mean_return": float(np.mean(values[split:])),
        },
        **evidence,
    }
    if sharpe is None or count < 3:
        if result["deflated_sharpe_status"] == "available":
            result.update(
                deflated_sharpe_status="unavailable",
                deflated_sharpe_reason="insufficient_nonconstant_returns",
            )
        return result

    standardized = (values - mean) / np.std(values)
    skew = float(np.mean(standardized**3))
    kurtosis = float(np.mean(standardized**4))
    denominator = 1 - skew * sharpe + (kurtosis - 1) * sharpe**2 / 4
    result.update(skewness=skew, kurtosis=kurtosis)
    if denominator <= 0:
        result.update(
            deflated_sharpe_status="unavailable",
            deflated_sharpe_reason="nonpositive_sharpe_sampling_variance",
        )
        return result
    iid_se = math.sqrt(denominator / (count - 1))
    result["probabilistic_sharpe_ratio"] = _NORMAL.cdf(
        (sharpe - benchmark_sharpe) / iid_se
    )
    if evidence["deflated_sharpe_status"] == "available":
        result["deflated_sharpe_ratio"] = _NORMAL.cdf(
            (sharpe - evidence["expected_maximum_sharpe_periodic"]) / iid_se
        )
    if sharpe > benchmark_sharpe:
        required = (
            1
            + denominator
            * (_NORMAL.inv_cdf(confidence) / (sharpe - benchmark_sharpe)) ** 2
        )
        result["minimum_track_record_periods"] = math.ceil(required)

    influence = standardized - sharpe * (standardized**2 - 1) / 2
    long_variance = float(np.mean(influence**2))
    for lag in range(1, hac_lags + 1):
        covariance = float(np.dot(influence[lag:], influence[:-lag]) / count)
        long_variance += 2 * (1 - lag / (hac_lags + 1)) * covariance
    hac_se = math.sqrt(max(0.0, long_variance) / (count - 1))
    width = _NORMAL.inv_cdf((1 + confidence) / 2) * hac_se
    result["sharpe_uncertainty"] = {
        "method": "Newey_West_Bartlett_Sharpe_influence",
        "lags": hac_lags,
        "confidence": confidence,
        "iid_standard_error_periodic": iid_se,
        "hac_standard_error_periodic": hac_se,
        "hac_standard_error_annualized": hac_se * scale,
        "hac_interval_annualized": [
            (sharpe - width) * scale,
            (sharpe + width) * scale,
        ],
    }
    return result
