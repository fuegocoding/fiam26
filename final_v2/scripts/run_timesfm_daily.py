"""Runs Daily TimesFM 3.0 (google/timesfm-3.0-pytorch) 21-trading-day trajectory
and quantile uncertainty inference on dsf_mini_from_2014.csv.

Checkpoints per formation month under final_v2/artifacts/timesfm_daily/parts/YYYY-MM.parquet
and writes combined final_v2/artifacts/timesfm_daily/forecasts.parquet.

Usage:
  python final_v2/scripts/run_timesfm_daily.py --smoke-test
  python final_v2/scripts/run_timesfm_daily.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pod_platform.accounting.evaluation import load_dsf_daily, load_fred  # noqa: E402
from pod_platform.signals.timesfm_daily import (  # noqa: E402
    DailyTimesFMConfig,
    load_timesfm_3p0_forecaster,
    prepare_daily_excess_inputs,
    summarize_trajectory_forecasts,
)

REPO = ROOT.parent
PANEL = REPO / "final" / "data" / "chars_final_with_names.parquet"
EXTERNAL = REPO / "final" / "external_data" / "external.parquet"
DSF_PATH = REPO / "data" / "raw" / "final" / "dsf_mini_from_2014.csv"
OUT_DIR = ROOT / "artifacts" / "timesfm_daily"
PARTS_DIR = OUT_DIR / "parts"


def main() -> None:
    import torch

    smoke_test = "--smoke-test" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
    if not smoke_test and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required for full 68-month Daily TimesFM batch run. Attach a GPU or pass --smoke-test."
        )

    panel = pd.read_parquet(PANEL, columns=["permno", "eom", "prc", "dolvol_126d"])
    panel["eom_per"] = pd.to_datetime(panel["eom"]).dt.to_period("M")
    panel = panel.loc[
        (panel["prc"].abs() >= 5.0)
        & (panel["dolvol_126d"] > 0)
        & (panel["eom_per"] >= pd.Period("2020-12", "M"))
        & (panel["eom_per"] <= pd.Period("2026-07", "M"))
    ].copy()

    formation_periods = sorted(panel["eom_per"].unique())
    if smoke_test:
        formation_periods = formation_periods[:1]
        print(f"Smoke test mode: 1 formation month ({formation_periods[0]}), max 32 stocks", flush=True)

    all_permnos = set(panel["permno"].astype(int).unique())
    if smoke_test:
        all_permnos = set(sorted(all_permnos)[:32])

    print(f"Loading dsf_mini_from_2014.csv for {len(all_permnos):,} liquid permnos...", flush=True)
    dsf = load_dsf_daily(
        DSF_PATH,
        permnos=all_permnos,
        first_date=pd.Timestamp("2019-11-01"),
        last_date=pd.Timestamp("2026-08-31"),
    )
    fred = load_fred(EXTERNAL)
    spx = fred[fred["series"] == "SP500"].dropna(subset=["value"]).copy()
    spx["day"] = pd.to_datetime(spx["date"])
    spx_daily = spx.set_index("day")["value"].sort_index().pct_change().dropna()

    cfg = DailyTimesFMConfig()
    print(f"Loading {cfg.model_id}...", flush=True)
    forecaster = load_timesfm_3p0_forecaster(cfg)

    for per in formation_periods:
        part_path = PARTS_DIR / f"{per}.parquet"
        if not smoke_test and part_path.exists():
            continue
        t0 = time.time()
        eom_ts = per.to_timestamp("M")
        ret_ts = (per + 1).to_timestamp("M")
        eligible = sorted(
            panel.loc[panel["eom_per"] == per, "permno"].astype(int).unique().tolist()
        )
        if smoke_test:
            eligible = eligible[:32]
        start_win = (per - 14).to_timestamp("M")
        dsf_win = dsf.loc[(dsf["day"] > start_win) & (dsf["day"] <= eom_ts)]
        valid_permnos, inputs = prepare_daily_excess_inputs(
            dsf_win, spx_daily, eligible, eom_ts
        )
        if not inputs:
            continue
        pf, qf = forecaster(inputs, cfg.horizon)
        ctx_lens = [len(arr) for arr in inputs]
        df_month = summarize_trajectory_forecasts(
            valid_permnos, eom_ts, ret_ts, pf, qf, ctx_lens
        )
        if smoke_test:
            print(f"Smoke test complete in {time.time() - t0:.1f}s: {len(df_month)} stocks scored", flush=True)
            print(df_month.head(5).to_string(index=False), flush=True)
            return
        df_month.to_parquet(part_path, index=False)
        print(f"  {per}: {len(df_month):,} stocks in {time.time() - t0:.1f}s", flush=True)

    parts = sorted(PARTS_DIR.glob("*.parquet"))
    if parts:
        combined = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        combined.to_parquet(OUT_DIR / "forecasts.parquet", index=False)
        print(f"Saved {len(combined):,} daily TimesFM rows to {OUT_DIR / 'forecasts.parquet'}", flush=True)


if __name__ == "__main__":
    main()
