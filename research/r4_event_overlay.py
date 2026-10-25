"""R4 — 8-K event response and daily overlay (registered 2026-10-25).

Event dated t is actionable no earlier than the open of t+1; the book formed on
information through t first earns the close of t+1 to the close of t+2 (brief
Section 7 daily case 1). Reported: market response windows days 1..k (not
harvestable) and harvestable windows days 2..k+1 against a late baseline.
Retrospective replay (spec 2.4); register: docs/review/research_register_20261025.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research" / "out"
OUT.mkdir(parents=True, exist_ok=True)

FILINGS = ROOT / "data" / "raw" / "final" / "8k_20150101_20260831_identified.parquet"
DSF = ROOT / "data" / "raw" / "final" / "dsf_mini_from_2014.csv"


def load_daily() -> pd.DataFrame:
    dsf = pd.read_csv(DSF, usecols=["permno", "date", "daily_return"])
    dsf = dsf.rename(columns={"daily_return": "ret"})
    dsf["date"] = pd.to_datetime(dsf.date, format="%Y%m%d")
    dsf = dsf.dropna(subset=["ret"]).sort_values(["permno", "date"])
    mkt = dsf.groupby("date").ret.mean().rename("mkt")
    dsf = dsf.join(mkt, on="date")
    dsf["ar"] = dsf.ret - dsf.mkt
    dsf["cum"] = dsf.groupby("permno").ar.cumsum()
    return dsf


def load_events() -> pd.DataFrame:
    cols = ["filing_date", "permno", "items", "is_amendment", "ticker"]
    ev = pd.read_parquet(FILINGS, columns=cols)
    ev["filing_date"] = pd.to_datetime(ev.filing_date)
    ev = ev.loc[~ev.is_amendment.fillna(False) & ev.permno.notna()]
    ev["permno"] = ev.permno.astype(int)
    ev["items"] = ev["items"].astype(str)
    ev["is_earnings"] = ev["items"].str.contains("2.02", na=False)
    return ev.drop_duplicates(["permno", "filing_date"])


def event_study(dsf: pd.DataFrame, ev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = dsf.groupby("permno")["cum"]
    for k in (1, 5, 6, 10, 11, 21, 22, 42):
        dsf[f"fwd{k}"] = g.shift(-k) - dsf.cum

    merged = ev.merge(dsf, left_on=["permno", "filing_date"], right_on=["permno", "date"], how="inner")
    merged = merged.dropna(subset=["fwd42", "fwd1"])
    # response days t+1..t+k (not harvestable); harvestable days t+2..t+k+1
    merged["resp_5"] = merged.fwd5
    merged["resp_10"] = merged.fwd10
    merged["resp_21"] = merged.fwd21
    merged["harv_5"] = merged.fwd6 - merged.fwd1
    merged["harv_10"] = merged.fwd11 - merged.fwd1
    merged["harv_21"] = merged.fwd22 - merged.fwd1
    merged["baseline_21_42"] = merged.fwd42 - merged.fwd21
    merged["month"] = merged.filing_date.dt.to_period("M").dt.to_timestamp("M")
    return merged, {}


def cluster_t(frame: pd.DataFrame, col: str) -> dict:
    monthly = frame.groupby("month")[col].mean().dropna()
    mean, sd = monthly.mean(), monthly.std(ddof=1)
    t = mean / (sd / np.sqrt(len(monthly))) if sd and sd > 0 else np.nan
    return {
        "events": int(len(frame)),
        "months": int(len(monthly)),
        "mean": float(mean),
        "t_stat_monthly_clustered": float(t),
    }


def main() -> None:
    dsf = load_daily()
    ev = load_events()
    merged, _ = event_study(dsf, ev)
    results: dict = {
        "sample": {
            "events_total": int(len(ev)),
            "events_with_windows": int(len(merged)),
            "permno": int(merged.permno.nunique()),
            "window": f"{merged.filing_date.min():%Y-%m} .. {merged.filing_date.max():%Y-%m}",
            "earnings_share": float(merged.is_earnings.mean()),
        }
    }

    for label, part in (
        ("all", merged),
        ("earnings_2_02", merged.loc[merged.is_earnings]),
        ("ex_earnings", merged.loc[~merged.is_earnings]),
    ):
        for col in ("resp_5", "resp_10", "resp_21", "harv_5", "harv_10", "harv_21", "baseline_21_42"):
            results[f"{label}__{col}"] = cluster_t(part, col)

    # Overlay test: direction = sign of the first reaction day (known at the
    # close of t+1, when the book is formed). Harvestable drift days 2..11.
    overlay = merged.dropna(subset=["fwd1", "harv_10"]).copy()
    overlay["signal"] = np.sign(overlay.fwd1)
    overlay = overlay.loc[overlay.signal.ne(0)]
    rows = []
    for month, part in overlay.groupby("month"):
        part = part.dropna(subset=["signal", "harv_10"])
        if len(part) < 20:
            continue
        rows.append(
            {
                "month": month,
                "n": len(part),
                "ic": part.signal.rank().corr(part.harv_10.rank()),
                "signed_drift": float((part.signal * part.harv_10).mean()),
            }
        )
    ovl = pd.DataFrame(rows)
    if len(ovl):
        mean, sd = ovl.signed_drift.mean(), ovl.signed_drift.std(ddof=1)
        results["overlay"] = {
            "months": int(len(ovl)),
            "mean_monthly_signed_drift": float(mean),
            "t_stat": float(mean / (sd / np.sqrt(len(ovl)))) if sd > 0 else None,
            "mean_ic": float(ovl.ic.mean()),
        }
        results["overlay_cost_screen"] = {
            "note": "two-sided 10bp assumed; a daily overlay turns over its book per event window",
            "rough_cost_per_window": 0.002,
            "net_monthly_signed_drift": float(mean - 0.002),
        }

    (OUT / "r4_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
