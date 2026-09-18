"""TimesFM 3.0 zero-shot forecasts on alternative single-feature series.

Same protocol as src/timesfm_forecast.py (one series per stock, horizon 1,
median + p10/p90 + spread, MIN_CONTEXT=12, leakage-safe context) but the
context series is a command-line choice:

  exc        realized monthly excess returns via ret_exc_lead1m (control; must
             reproduce output/preds_timesfm_all.parquet up to float noise)
  resid_mkt  excess return minus the equal-weight market mean of its month
  resid_sec  excess return minus the equal-weight mean of its GICS
             sector-month (missing GICS -> market mean)
  dolvol     monthly log1p dollar-volume growth (consecutive months only)

Context rule: series values with month <= target month t - 1. For exc the
series month is eom+1 (a realized return of month m lives in the row at
eom=m-1), so the rule matches the original script exactly. For
contemporaneous columns (ret_exc, dolvol) the value of month m lives in the
row at eom=m.

Output: output/timesfm_alt/preds_<variant>.csv (+ .parquet when pyarrow
works), one row per feature row (permno, eom) that received a forecast, plus
attrs.csv (permno, target_month, size_grp, sector) so the local evaluation
needs no parquet reader. Per-(variant, month) part files under
output/timesfm_alt/parts/ act as the resume cache.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "timesfm_alt"
PARTS = OUT / "parts"

FIRST_TARGET = pd.Timestamp("2021-01-31")
LAST_TARGET = pd.Timestamp("2026-08-31")
CHECKPOINT = "google/timesfm-3.0-pytorch"
BATCH_SIZE = 256
MIN_CONTEXT = 12

ALL_VARIANTS = ("exc", "resid_mkt", "resid_sec", "dolvol")


def load_panel(path):
    table = pd.read_parquet(
        path,
        columns=["permno", "eom", "ret_exc_lead1m", "ret_exc", "gics", "dolvol", "size_grp"],
    )
    table["eom"] = pd.to_datetime(table["eom"])
    table["sector"] = table["gics"].astype("string").str.slice(0, 2)
    table.loc[table["gics"].isna(), "sector"] = None
    return table


def variant_series(panel, variant):
    if variant == "exc":
        t = panel.dropna(subset=["ret_exc_lead1m"])
        return pd.DataFrame({
            "permno": t["permno"].to_numpy(),
            "month": (t["eom"] + pd.offsets.MonthEnd(1)).to_numpy(),
            "value": t["ret_exc_lead1m"].to_numpy(dtype=np.float32),
        })
    if variant == "dolvol":
        t = panel[["permno", "eom", "dolvol"]].dropna(subset=["dolvol"]).sort_values(
            ["permno", "eom"]
        )
        t["logv"] = np.log1p(t["dolvol"].clip(lower=0))
        t["month_idx"] = t["eom"].dt.year * 12 + t["eom"].dt.month
        g = t.groupby("permno")
        t["prev_logv"] = g["logv"].shift(1)
        t["prev_idx"] = g["month_idx"].shift(1)
        consecutive = (t["month_idx"] - t["prev_idx"]) == 1
        t["value"] = (t["logv"] - t["prev_logv"]).where(consecutive)
        t = t.dropna(subset=["value"])
        return pd.DataFrame({
            "permno": t["permno"].to_numpy(),
            "month": t["eom"].to_numpy(),
            "value": t["value"].to_numpy(dtype=np.float32),
        })
    t = panel.dropna(subset=["ret_exc"])
    mkt = t.groupby("eom")["ret_exc"].transform("mean")
    if variant == "resid_mkt":
        value = t["ret_exc"] - mkt
    elif variant == "resid_sec":
        sec = t.groupby(["eom", "sector"])["ret_exc"].transform("mean")
        value = np.where(
            t["sector"].notna().to_numpy(),
            (t["ret_exc"] - sec).to_numpy(),
            (t["ret_exc"] - mkt).to_numpy(),
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    return pd.DataFrame({
        "permno": t["permno"].to_numpy(),
        "month": t["eom"].to_numpy(),
        "value": np.asarray(value, dtype=np.float32),
    })


def target_months(master, start, end):
    months = master["eom"].sort_values().unique()
    feats = []
    for m in months:
        nxt = m + pd.offsets.MonthEnd(1)
        if start <= nxt <= end:
            feats.append(m)
    return feats


def build_contexts(series, feat_month, permnos):
    hist = series[series["month"] <= feat_month]
    grouped = {p: v for p, v in hist.groupby("permno", sort=False)["value"]}
    contexts, keep = [], []
    for permno in permnos:
        values = grouped.get(permno)
        if values is None or len(values) < MIN_CONTEXT:
            continue
        contexts.append(values.to_numpy(dtype=np.float32))
        keep.append(permno)
    return keep, contexts


def forecast_month(forecaster, series, feat, feat_month):
    keep, contexts = build_contexts(series, feat_month, feat["permno"].to_numpy())
    if not contexts:
        return None
    outputs = list(forecaster.predict_batch(contexts, horizon=1, return_quantiles=True))
    med = np.array([o.forecast[0] for o in outputs], dtype=np.float32)
    q = np.array([o.quantiles[0] for o in outputs], dtype=np.float32)
    rows = feat[feat["permno"].isin(keep)].reset_index(drop=True)
    out = rows[["permno", "eom"]].copy()
    out["target_month"] = out["eom"] + pd.offsets.MonthEnd(1)
    out["ret_exc_lead1m"] = rows["ret_exc_lead1m"].to_numpy()
    out["tfm_forecast"] = med
    out["tfm_q10"] = q[:, 0]
    out["tfm_q90"] = q[:, -1]
    out["tfm_spread"] = q[:, -1] - q[:, 0]
    out["tfm_n_ctx"] = np.array([len(c) for c in contexts], dtype=np.int32)
    out["ctx_mean6"] = np.array(
        [float(np.mean(c[-6:])) for c in contexts], dtype=np.float32
    )
    return out


def save_part(frame, path):
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        frame.to_csv(path.with_suffix(".csv"), index=False)


def load_part(path):
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS),
                    help=f"comma-separated subset of {ALL_VARIANTS}")
    ap.add_argument("--start", default=str(FIRST_TARGET.date()))
    ap.add_argument("--end", default=str(LAST_TARGET.date()))
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--data", default=str(RAW / "chars_final_with_names.parquet"),
                    help="panel parquet with permno, eom, ret_exc_lead1m, "
                         "ret_exc, gics, dolvol, size_grp")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ALL_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; choose from {ALL_VARIANTS}")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    panel = load_panel(args.data)
    print(
        f"panel: {len(panel):,} rows, {panel['permno'].nunique():,} permnos, "
        f"variants: {variants}"
    )

    master = panel.dropna(subset=["ret_exc_lead1m"])[["permno", "eom", "ret_exc_lead1m"]]
    attrs = panel.dropna(subset=["ret_exc_lead1m"])[
        ["permno", "eom", "size_grp", "sector"]
    ].copy()
    attrs["target_month"] = attrs["eom"] + pd.offsets.MonthEnd(1)
    attrs = attrs[["permno", "target_month", "size_grp", "sector"]]
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    attrs.to_csv(OUT / "attrs.csv", index=False)

    series = {}
    for v in variants:
        s = variant_series(panel, v).sort_values(["permno", "month"])
        print(f"variant {v}: {len(s):,} series values, "
              f"{s['permno'].nunique():,} permnos, "
              f"{s['month'].min().date()}..{s['month'].max().date()}")
        series[v] = s

    months = target_months(master, start, end)
    print(f"{len(months)} target months: {months[0].date()}..{months[-1].date()}")

    from timesfm3 import TimesFM3Forecaster

    forecaster = TimesFM3Forecaster.from_pretrained(
        CHECKPOINT, per_core_batch_size=args.batch
    )
    print(f"loaded {CHECKPOINT} on {forecaster.device}, batch={args.batch}")

    for i, feat_month in enumerate(months, 1):
        feat = master[master["eom"] == feat_month]
        t0 = time.time()
        for v in variants:
            done = PARTS / f"tfm_{v}_{feat_month.date()}"
            part = load_part(done)
            if part is not None:
                continue
            part = forecast_month(forecaster, series[v], feat, feat_month)
            if part is None:
                continue
            save_part(part, done)
        print(f"{feat_month.date()}: {len(feat):,} feature rows in "
              f"{time.time() - t0:.0f}s")

    for v in variants:
        parts = []
        for feat_month in months:
            part = load_part(PARTS / f"tfm_{v}_{feat_month.date()}")
            if part is not None:
                parts.append(part)
        if not parts:
            print(f"variant {v}: no forecasts")
            continue
        full = pd.concat(parts, ignore_index=True)
        full.to_csv(OUT / f"preds_{v}.csv", index=False)
        try:
            full.to_parquet(OUT / f"preds_{v}.parquet", index=False)
        except Exception:
            pass
        print(f"variant {v}: saved {len(full):,} rows -> "
              f"{OUT / f'preds_{v}.csv'}")


if __name__ == "__main__":
    main()
