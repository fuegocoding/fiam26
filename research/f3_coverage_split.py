"""F3 — coverage split of the filing channel (registered 2026-10-25).

The filing_events sleeve counts EDGAR form-type events (`external.edgar`
index) mapped through a crosswalk that also reads the 8-K corpus. (a) does the
EDGAR source itself cover exited names; (b) is the sleeve's IC concentrated in
active filers. Register: docs/review/research_register_20261025.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import external  # noqa: E402

external.FILE = ROOT / "final" / "external_data" / "external.parquet"

OUT = ROOT / "research" / "out"
OUT.mkdir(parents=True, exist_ok=True)
PANEL = ROOT / "data" / "raw" / "final" / "chars_final_with_names.parquet"
FORECASTS = ROOT / "final" / "output" / "filing_events" / "forecasts.parquet"
FEATURES = ROOT / "final" / "output" / "filing_events" / "features.parquet"

NAMED = {
    "Sears": ["SEARS"],
    "Bon-Ton": ["BON TON", "BON-TON"],
    "Pier 1": ["PIER 1"],
    "GNC": ["GNC"],
    "Ascena": ["ASCENA"],
    "J.C. Penney": ["PENNEY"],
}


def monthly_ic(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    rows = []
    for eom, part in frame.groupby("eom"):
        part = part.dropna(subset=[score, target])
        if len(part) < 20:
            continue
        rows.append({"eom": eom, "n": len(part), "ic": part[score].rank().corr(part[target].rank())})
    return pd.DataFrame(rows)


def ic_stats(ic: pd.DataFrame) -> dict:
    if ic.empty:
        return {"months": 0}
    mean, sd = ic.ic.mean(), ic.ic.std(ddof=1)
    t = mean / (sd / np.sqrt(len(ic))) if sd and sd > 0 else np.nan
    return {
        "months": int(len(ic)),
        "mean_ic": float(mean),
        "t_stat": float(t),
        "hit_rate": float((ic.ic > 0).mean()),
    }


def main() -> None:
    results: dict = {}

    # (a) do the named delisters appear in the EDGAR index extract at all?
    edgar = external.read("edgar", columns=["form_type", "company_name", "cik", "date_filed"])
    edgar["company_name"] = edgar.company_name.astype(str).str.upper()
    results["a_named_delisters_in_edgar_index"] = {}
    for label, needles in NAMED.items():
        hit = edgar[edgar.company_name.str.contains("|".join(needles), na=False)]
        results["a_named_delisters_in_edgar_index"][label] = {
            "rows": int(len(hit)),
            "ciks": sorted({int(c) for c in hit.cik.dropna().unique()})[:5],
            "first_filed": str(hit.date_filed.min()) if len(hit) else None,
            "last_filed": str(hit.date_filed.max()) if len(hit) else None,
        }

    # (b) mapped event coverage: exited vs surviving panel names
    feats = pd.read_parquet(FEATURES)
    count_cols = [c for c in feats.columns if c not in ("permno", "return_month", "lockup_m")]
    feats["total_events"] = feats[count_cols].sum(axis=1)
    feats["return_month"] = pd.to_datetime(feats.return_month)
    active = feats.loc[feats.total_events.gt(0), ["permno", "return_month"]]

    panel = pd.read_parquet(PANEL, columns=["permno", "eom", "ret_exc_lead1m"])
    panel["eom"] = pd.to_datetime(panel.eom)
    life = panel.groupby("permno").eom.max()
    exited = set(life[life.between("2017-01-01", "2020-12-31")].index)
    survivors = set(life[life.ge("2026-07-01")].index)
    ever_active = set(active.permno.unique())
    results["b_mapped_event_coverage"] = {
        "exited_2017_2020": {
            "n": len(exited),
            "ever_active": sum(1 for p in exited if p in ever_active),
            "share": float(sum(1 for p in exited if p in ever_active) / len(exited)),
        },
        "survivors": {
            "n": len(survivors),
            "ever_active": sum(1 for p in survivors if p in ever_active),
            "share": float(sum(1 for p in survivors if p in ever_active) / len(survivors)),
        },
    }

    # (c) sleeve IC split by trailing-12m mapped activity
    preds = pd.read_parquet(FORECASTS)
    preds["eom"] = pd.to_datetime(preds.eom)
    data = preds.merge(
        panel[["permno", "eom", "ret_exc_lead1m"]], on=["permno", "eom"], how="inner"
    )
    keys = data[["permno", "eom"]].copy()
    keys["lag12"] = keys.eom - pd.DateOffset(months=12)
    merged = keys.merge(active.rename(columns={"return_month": "active_month"}), on="permno", how="left")
    merged = merged.loc[merged.active_month.between(merged.lag12, merged.eom)]
    flagged = merged.groupby(["permno", "eom"]).size().rename("n_active_months").reset_index()
    data = data.merge(flagged, on=["permno", "eom"], how="left")
    data["n_active_months"] = data.n_active_months.fillna(0)
    data["covered"] = data.n_active_months.gt(0)

    results["c_ic_all"] = ic_stats(monthly_ic(data, "pred_filing_events", "ret_exc_lead1m"))
    results["c_ic_covered"] = ic_stats(
        monthly_ic(data.loc[data.covered], "pred_filing_events", "ret_exc_lead1m")
    )
    results["c_ic_uncovered"] = ic_stats(
        monthly_ic(data.loc[~data.covered], "pred_filing_events", "ret_exc_lead1m")
    )
    results["c_share_covered"] = float(data.covered.mean())

    # does the sleeve beat the coverage flag itself? IC of coverage vs returns
    results["c_ic_coverage_flag_only"] = ic_stats(monthly_ic(data, "covered", "ret_exc_lead1m"))

    (OUT / "f3_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
