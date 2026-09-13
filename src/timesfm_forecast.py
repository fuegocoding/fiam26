"""Zero-shot TimesFM 3.0 forecasts of next-month excess returns, per stock-month.

For every target month t in 2021-01..2026-08 we take the feature row at eom = t-1
(the row whose ret_exc_lead1m IS the month-t return) and forecast it with
TimesFM using only each stock's realized monthly excess-return history through
month t-1. A realized return of month m lives in the row at eom = m-1, so the
context for target month t is ret_exc_lead1m over rows with eom <= t-2. The
future target never enters the context.

Output: one row per (permno, eom) feature row with the TimesFM median forecast
and the p10/p90 quantiles, joinable to the XGBoost predictions on (permno, eom).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
PARTS = OUT / "timesfm_parts"
PARTS.mkdir(parents=True, exist_ok=True)

FIRST_TARGET = pd.Timestamp("2021-01-31")
LAST_TARGET = pd.Timestamp("2026-08-31")
CHECKPOINT = "google/timesfm-3.0-pytorch"
BATCH_SIZE = 128
MIN_CONTEXT = 12


def load_returns(path=None):
    if path:
        table = pd.read_parquet(path)
    else:
        table = pq.read_table(
            RAW / "chars_final_with_names.parquet",
            columns=["permno", "eom", "ret_exc_lead1m"],
        ).to_pandas()
    table["eom"] = pd.to_datetime(table["eom"])
    return table.dropna(subset=["ret_exc_lead1m"])


def target_months(returns, start, end):
    months = returns["eom"].sort_values().unique()
    feats = []
    for m in months:
        nxt = m + pd.offsets.MonthEnd(1)
        if start <= nxt <= end:
            feats.append(m)
    return feats


def build_contexts(returns, feat_month, sample_permnos=None):
    feat = returns[returns["eom"] == feat_month]
    if sample_permnos is not None:
        feat = feat[feat["permno"].isin(sample_permnos)]
    hist = returns[(returns["eom"] <= feat_month - pd.offsets.MonthEnd(1))]
    hist = hist[hist["permno"].isin(set(feat["permno"]))].sort_values("eom")
    grouped = {p: g for p, g in hist.groupby("permno", sort=False)["ret_exc_lead1m"]}
    contexts, keep = [], []
    for permno in feat["permno"].values:
        series = grouped.get(permno)
        if series is None or len(series) < MIN_CONTEXT:
            continue
        contexts.append(series.to_numpy(dtype=np.float32))
        keep.append(permno)
    return feat[feat["permno"].isin(keep)], contexts


def forecast_month(forecaster, returns, feat_month, sample_permnos=None):
    feat, contexts = build_contexts(returns, feat_month, sample_permnos)
    if not contexts:
        return None
    outputs = list(
        forecaster.predict_batch(contexts, horizon=1, return_quantiles=True)
    )
    med = np.array([o.forecast[0] for o in outputs], dtype=np.float32)
    q = np.array([o.quantiles[0] for o in outputs], dtype=np.float32)
    out = feat[["permno", "eom"]].reset_index(drop=True)
    out["target_month"] = out["eom"] + pd.offsets.MonthEnd(1)
    out["ret_exc_lead1m"] = feat["ret_exc_lead1m"].to_numpy()
    out["tfm_forecast"] = med
    out["tfm_q10"] = q[:, 0]
    out["tfm_q90"] = q[:, -1]
    out["tfm_spread"] = q[:, -1] - q[:, 0]
    out["tfm_n_ctx"] = np.array([len(c) for c in contexts], dtype=np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="limit to N random permnos (pilot)")
    ap.add_argument("--start", default=str(FIRST_TARGET.date()))
    ap.add_argument("--end", default=str(LAST_TARGET.date()))
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--data", default=None, help="optional returns parquet (permno, eom, ret_exc_lead1m)")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    returns = load_returns(args.data)
    print(
        f"returns panel: {len(returns):,} rows, {returns['permno'].nunique():,} permnos"
    )
    sample_permnos = None
    if args.sample:
        rng = np.random.default_rng(0)
        sample_permnos = set(
            rng.choice(returns["permno"].unique(), size=args.sample, replace=False)
        )

    from timesfm3 import TimesFM3Forecaster

    forecaster = TimesFM3Forecaster.from_pretrained(
        CHECKPOINT, per_core_batch_size=args.batch
    )
    print(f"loaded {CHECKPOINT} on {forecaster.device}, batch={args.batch}")

    months = target_months(returns, start, end)
    print(f"{len(months)} target months: {months[0].date()}..{months[-1].date()}")

    all_parts = []
    for i, feat_month in enumerate(months, 1):
        done = PARTS / f"tfm_{feat_month.date()}.parquet"
        if done.exists():
            all_parts.append(pd.read_parquet(done))
            continue
        t0 = time.time()
        part = forecast_month(forecaster, returns, feat_month, sample_permnos)
        if part is None:
            print(f"{feat_month.date()}: no eligible series, skipped")
            continue
        part.to_parquet(done, index=False)
        all_parts.append(part)
        n = len(part)
        print(
            f"{feat_month.date()}: {n:,} forecasts in {time.time() - t0:.0f}s "
            f"({n / max(time.time() - t0, 1e-9):.0f}/s)"
        )

    if all_parts:
        full = pd.concat(all_parts, ignore_index=True)
        suffix = "_sample" if args.sample else ""
        path = OUT / f"preds_timesfm{suffix}.parquet"
        full.to_parquet(path, index=False)
        print(f"saved {len(full):,} rows -> {path}")


if __name__ == "__main__":
    main()
