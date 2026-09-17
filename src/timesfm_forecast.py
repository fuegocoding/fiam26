"""Zero-shot TimesFM 3.0 forecasts of next-month excess returns, per stock-month.

For every target month t in 2021-01..2026-08 we take the feature row at eom = t-1
(the row whose ret_exc_lead1m IS the month-t return) and forecast it with
TimesFM using only each stock's realized monthly excess-return history through
month t-1. The current-month ret_exc is observed at that row's month-end;
ret_exc_lead1m is the following month's outcome. Context uses a contiguous
ret_exc history through t-1. The future target never enters context or determines
forecast eligibility.

Output: one row per (permno, eom) feature row with the TimesFM median forecast
and the p10/p90 quantiles, joinable to the XGBoost predictions on (permno, eom).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "timesfm_safe"
PARTS = OUT / "timesfm_parts"

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
            columns=["permno", "eom", "ret_exc", "ret_exc_lead1m"],
        ).to_pandas()
    table["eom"] = pd.to_datetime(table["eom"])
    if "ret_exc" not in table:
        raise ValueError("Safe forecasting requires ret_exc (realized current-month returns) and the full formation universe; the old target-filtered returns_panel is unsuitable.")
    if table.duplicated(["permno", "eom"]).any():
        raise ValueError("Duplicate stock-months")
    if "ret_exc_lead1m" not in table:
        table["ret_exc_lead1m"] = np.nan
    return table


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
    hist = returns[returns["eom"] <= feat_month]
    hist = hist[hist["permno"].isin(set(feat["permno"]))].sort_values("eom")
    grouped = {p: g for p, g in hist.groupby("permno", sort=False)}
    contexts, keep = [], []
    for permno in feat["permno"].values:
        stock = grouped.get(permno)
        if stock is None:
            continue
        calendar = pd.date_range(stock.eom.min(), feat_month, freq="ME")
        series = stock.set_index("eom").ret_exc.reindex(calendar).to_numpy(dtype=np.float32)
        # Forecast a regular contiguous suffix rather than compressing calendar gaps.
        gaps = np.flatnonzero(~np.isfinite(series))
        if len(gaps):
            series = series[gaps[-1] + 1:]
        if len(series) < MIN_CONTEXT:
            continue
        contexts.append(series)
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
    ap.add_argument("--data", default=None, help="optional FULL returns parquet (permno, eom, ret_exc, optional ret_exc_lead1m)")
    ap.add_argument("--output-dir", type=Path, default=OUT, help="Separate directory for forecasts with safe universe/calendar handling")
    args = ap.parse_args()
    global PARTS
    PARTS = args.output_dir / "timesfm_parts"
    PARTS.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    returns = load_returns(args.data)
    source = Path(args.data) if args.data else RAW / "chars_final_with_names.parquet"
    policy = {"availability_version": 2, "source": str(source.resolve()),
              "source_size": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns,
              "checkpoint": CHECKPOINT, "sample": args.sample, "start": args.start, "end": args.end,
              "future_target_independent_universe": True, "calendar_policy": "contiguous_realized_suffix"}
    manifest = args.output_dir / "forecast_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != policy:
        raise ValueError("Forecast inputs/settings changed; use a new --output-dir rather than mixing cached parts.")
    if not manifest.exists() and any(PARTS.glob("*.parquet")):
        raise ValueError("Existing forecast parts have no safe availability manifest; use a new --output-dir.")
    manifest.write_text(json.dumps(policy, indent=2) + "\n")
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
        path = args.output_dir / f"preds_timesfm{suffix}.parquet"
        full.to_parquet(path, index=False)
        print(f"saved {len(full):,} rows -> {path}")


if __name__ == "__main__":
    main()
