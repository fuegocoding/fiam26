"""Daily TimesFM 3.0 (google/timesfm-3.0-pytorch) 21-trading-day trajectory
and quantile uncertainty forecaster on dsf_mini_from_2014.csv.

Strictly point-in-time: for each formation month-end t, only daily returns on or
before t are fed into the 252-trading-day context window to forecast the 21
trading days of holding month t+1.
"""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None
try:
    from timesfm3 import TimesFM3Forecaster
except ImportError:
    TimesFM3Forecaster = None


from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

MAX_CONTEXT_DAYS = 252
MIN_CONTEXT_DAYS = 63
HORIZON_DAYS = 21
BATCH_SIZE = 512


@dataclass(frozen=True)
class DailyTimesFMConfig:
    model_id: str = "google/timesfm-3.0-pytorch"
    max_context: int = 256
    horizon: int = HORIZON_DAYS
    min_context: int = MIN_CONTEXT_DAYS
    batch_size: int = BATCH_SIZE


def prepare_daily_excess_inputs(
    dsf_window: pd.DataFrame,
    spx_daily: pd.Series,
    eligible_permnos: Sequence[int],
    cutoff_date: pd.Timestamp,
    *,
    max_context: int = MAX_CONTEXT_DAYS,
    min_context: int = MIN_CONTEXT_DAYS,
) -> tuple[list[int], list[np.ndarray]]:
    """Extracts trailing market-adjusted daily log-return arrays strictly <= cutoff_date."""
    sub = dsf_window.loc[
        (dsf_window["day"] <= cutoff_date)
        & (dsf_window["permno"].isin(eligible_permnos))
    ]
    if sub.empty:
        return [], []
    px = sub.pivot(index="day", columns="permno", values="daily_return").tail(
        max_context
    )
    spx_aligned = spx_daily.reindex(px.index).fillna(0.0).to_numpy(float)

    valid_permnos: list[int] = []
    inputs: list[np.ndarray] = []
    for permno in eligible_permnos:
        if permno not in px.columns:
            continue
        col = px[permno].to_numpy(float)
        finite_mask = np.isfinite(col)
        if int(finite_mask.sum()) < min_context:
            continue
        # Subtract daily S&P 500 return on valid days and convert to clipped log returns
        excess = np.where(finite_mask, col - spx_aligned, np.nan)
        valid_vals = excess[np.isfinite(excess)][-max_context:]
        clipped = np.clip(valid_vals, -0.50, 0.50)
        log_ret = np.log1p(clipped).astype(np.float32)
        if np.std(log_ret) < 1e-8:
            continue
        valid_permnos.append(int(permno))
        inputs.append(log_ret)
    return valid_permnos, inputs


def summarize_trajectory_forecasts(
    permnos: Sequence[int],
    formation_eom: pd.Timestamp,
    return_month: pd.Timestamp,
    point_forecasts: np.ndarray,
    quantile_forecasts: np.ndarray,
    context_lengths: Sequence[int],
) -> pd.DataFrame:
    """Summarizes 21-day point and quantile forecasts into holding-month signals & risk metrics.

    point_forecasts shape: (N, H)
    quantile_forecasts shape: (N, H, Q) where Q has mean at idx 0, q10 at idx 1, q90 at idx -1
    """
    pf = np.asarray(point_forecasts, dtype=float)
    qf = np.asarray(quantile_forecasts, dtype=float)
    h = pf.shape[1]
    h5 = min(5, h)
    h10 = min(10, h)

    cum21_log = pf.sum(axis=1)
    cum5_log = pf[:, :h5].sum(axis=1)
    slope_log = pf[:, h10:].sum(axis=1) - pf[:, :h10].sum(axis=1)

    if qf.ndim == 3 and qf.shape[2] >= 2:
        q_low_sum = qf[:, :, 0].sum(axis=1)
        q_high_sum = qf[:, :, -1].sum(axis=1)
        q_spread = np.expm1(np.clip(q_high_sum, -2.0, 2.0)) - np.expm1(
            np.clip(q_low_sum, -2.0, 2.0)
        )
        upside_tail_5d = np.expm1(
            np.clip(qf[:, :h5, -1].sum(axis=1), -2.0, 2.0)
        )
    else:
        q_spread = np.zeros(len(permnos), dtype=float)
        upside_tail_5d = np.expm1(np.clip(cum5_log, -2.0, 2.0))

    return pd.DataFrame(
        {
            "permno": np.asarray(permnos, dtype="int64"),
            "eom": pd.Timestamp(formation_eom),
            "return_month": pd.Timestamp(return_month),
            "tfm_daily_cum21": np.expm1(np.clip(cum21_log, -2.0, 2.0)),
            "tfm_daily_cum5": np.expm1(np.clip(cum5_log, -2.0, 2.0)),
            "tfm_daily_path_slope": slope_log,
            "tfm_daily_q_spread": np.abs(q_spread),
            "tfm_daily_upside_tail_5d": upside_tail_5d,
            "tfm_daily_n_ctx": np.asarray(context_lengths, dtype="int32"),
        }
    )


def load_timesfm_3p0_forecaster(
    cfg: DailyTimesFMConfig = DailyTimesFMConfig(),
) -> Callable[[list[np.ndarray], int], tuple[np.ndarray, np.ndarray]]:
    """Loads google/timesfm-3.0-pytorch via TimesFM3Forecaster and returns a batched forecast callable."""
    if torch is None or TimesFM3Forecaster is None:
        raise ImportError("TimesFM inference requires torch and timesfm3")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    forecaster = TimesFM3Forecaster.from_pretrained(
        pretrained_model_name_or_path=cfg.model_id,
        per_core_batch_size=min(cfg.batch_size, 128),
    )

    def _forecast_fn(
        inputs: list[np.ndarray], horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        outputs = list(
            forecaster.predict_batch(
                contexts=inputs,
                horizon=horizon,
                return_quantiles=True,
                use_znorm=True,
            )
        )
        pf = np.stack(
            [np.asarray(o.forecast, dtype=np.float32) for o in outputs], axis=0
        )
        qf = np.stack(
            [np.asarray(o.quantiles, dtype=np.float32) for o in outputs], axis=0
        )
        return pf, qf

    return _forecast_fn
