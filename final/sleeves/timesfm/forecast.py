"""Zero-shot TimesFM 3 forecasts of each stock's next-month excess return."""

import pathlib
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import timesfm3

from sleeves import base

CHECKPOINT = "google/timesfm-3.0-pytorch"
BATCH_SIZE = 256
MIN_CONTEXT = 12
# Earlier targets give the graph model forecasts for its training years.
FIRST_TARGET = pd.Timestamp("2016-01-31")


def path(work_dir: pathlib.Path) -> pathlib.Path:
    """Location of the combined TimesFM forecasts."""
    return work_dir / "timesfm" / "forecasts.parquet"


def build(context: base.Context) -> pathlib.Path:
    """Forecasts month t+1 from each stock's contiguous returns through t.

    Finished months are kept under parts/ so an interrupted run resumes.
    """
    output = path(context.work_dir)
    if output.exists():
        return output
    parts_dir = output.parent / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    returns = pq.read_table(
        context.panel_path, columns=["permno", "eom", "ret_exc"]
    ).to_pandas()
    returns["eom"] = pd.to_datetime(returns.eom)
    if returns.duplicated(["permno", "eom"]).any():
        raise ValueError("Duplicate stock-months in the return panel")
    months = [
        month
        for month in pd.DatetimeIndex(returns.eom.unique()).sort_values()
        if FIRST_TARGET <= month + pd.offsets.MonthEnd(1) <= base.LAST_TARGET
    ]
    forecaster = None
    parts = []
    for month in months:
        part_path = parts_dir / f"{month:%Y-%m}.parquet"
        if part_path.exists():
            parts.append(pd.read_parquet(part_path))
            continue
        if forecaster is None:
            forecaster = timesfm3.TimesFM3Evaluator(
                timesfm3.ModelConfig(
                    checkpoint_path=CHECKPOINT,
                    per_core_batch_size=BATCH_SIZE,
                    device=context.device,
                )
            )
        started = time.time()
        part = _forecast_month(forecaster, returns, month)
        if part is None:
            continue
        part.to_parquet(part_path, index=False)
        parts.append(part)
        print(
            f"TimesFM {month:%Y-%m}: {len(part):,} forecasts,"
            f" {time.time() - started:.0f}s",
            flush=True,
        )
    pd.concat(parts, ignore_index=True).to_parquet(output, index=False)
    return output


def _contexts(
    returns: pd.DataFrame, month: pd.Timestamp
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Each stock's latest gap-free return run, kept if long enough."""
    current = returns[returns.eom == month]
    history = returns[returns.eom <= month]
    history = history[history.permno.isin(set(current.permno))]
    grouped = {
        permno: stock
        for permno, stock in history.sort_values("eom").groupby(
            "permno", sort=False
        )
    }
    contexts, kept = [], []
    for permno in current.permno.to_numpy():
        stock = grouped.get(permno)
        if stock is None:
            continue
        calendar = pd.date_range(stock.eom.min(), month, freq="ME")
        series = (
            stock.set_index("eom")
            .ret_exc.reindex(calendar)
            .to_numpy(dtype=np.float32)
        )
        gaps = np.flatnonzero(~np.isfinite(series))
        if len(gaps):
            series = series[gaps[-1] + 1 :]
        if len(series) < MIN_CONTEXT:
            continue
        contexts.append(series)
        kept.append(permno)
    return current[current.permno.isin(kept)], contexts


def _forecast_month(
    forecaster: timesfm3.TimesFM3Evaluator,
    returns: pd.DataFrame,
    month: pd.Timestamp,
) -> pd.DataFrame | None:
    current, contexts = _contexts(returns, month)
    if not contexts:
        return None
    outputs = list(
        forecaster.predict_batch(
            contexts,
            horizon=1,
            return_quantiles=True,
            use_symmetric_averaging=False,
        )
    )
    if len(outputs) != len(contexts):
        raise ValueError("TimesFM returned a different number of forecasts")
    median = np.array([o.forecast[0] for o in outputs], dtype=np.float32)
    quantiles = np.array([o.quantiles[0] for o in outputs], dtype=np.float32)
    out = current[["permno", "eom"]].reset_index(drop=True)
    out["target_month"] = out.eom + pd.offsets.MonthEnd(1)
    out["tfm_forecast"] = median
    out["tfm_spread"] = quantiles[:, -1] - quantiles[:, 0]
    out["tfm_n_ctx"] = np.array([len(c) for c in contexts], dtype=np.int32)
    return out
