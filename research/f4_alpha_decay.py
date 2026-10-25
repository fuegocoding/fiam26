"""F4 — alpha decay profile of the blended score (registered 2026-10-25).

IC of the blended score at 1, 2 and 3 holding months; the empirical answer to
the rebalancing-frequency question (brief section 4). Register:
docs/review/research_register_20261025.md.
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
PRED = ROOT / "final" / "output" / "predictions.parquet"
PANEL = ROOT / "data" / "raw" / "final" / "chars_final_with_names.parquet"


def monthly_ic(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    rows = []
    for eom, part in frame.groupby("return_month"):
        part = part.dropna(subset=[score, target])
        if len(part) < 30:
            continue
        rows.append({"month": eom, "n": len(part), "ic": part[score].rank().corr(part[target].rank())})
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
    preds = pd.read_parquet(
        PRED,
        columns=["return_month", "permno", "final_score", "final_score_liquid", "sleeve_score", "graph_available"],
    )
    preds["return_month"] = pd.to_datetime(preds.return_month)

    panel = pd.read_parquet(PANEL, columns=["permno", "eom", "ret_exc"])
    panel["eom"] = pd.to_datetime(panel.eom)
    panel = panel.sort_values(["permno", "eom"])
    g = panel.groupby("permno")["ret_exc"]
    panel["f1"] = g.shift(0)  # this month's excess return
    panel["f2"] = panel.f1 + g.shift(-1)
    panel["f3"] = panel.f2 + g.shift(-2)
    panel["f6"] = panel.f3 + g.shift(-3) + g.shift(-4) + g.shift(-5)
    rets = panel[["permno", "eom", "f1", "f2", "f3", "f6"]].rename(columns={"eom": "return_month"})

    data = preds.merge(rets, on=["permno", "return_month"], how="inner")
    results: dict = {
        "sample": {"rows": int(len(data)), "window": f"{data.return_month.min():%Y-%m} .. {data.return_month.max():%Y-%m}"}
    }

    for score in ("final_score", "final_score_liquid", "sleeve_score"):
        for h, col in (("1m", "f1"), ("2m", "f2"), ("3m", "f3"), ("6m", "f6")):
            results[f"{score}__{h}"] = ic_stats(monthly_ic(data, score, col))

    # decay ratio: IC(3m) / IC(1m)
    for score in ("final_score", "final_score_liquid"):
        a = results[f"{score}__1m"]["mean_ic"]
        b = results[f"{score}__3m"]["mean_ic"]
        results[f"{score}__decay_ratio_3m_over_1m"] = float(b / a) if a else None

    # half-life style: is IC(2m) distinguishable from IC(1m)? paired monthly test
    rows = []
    for month, part in data.groupby("return_month"):
        part = part.dropna(subset=["final_score", "f1", "f3"])
        if len(part) < 30:
            continue
        rows.append(
            {
                "month": month,
                "ic1": part.final_score.rank().corr(part.f1.rank()),
                "ic3": part.final_score.rank().corr(part.f3.rank()),
            }
        )
    paired = pd.DataFrame(rows)
    paired["d"] = paired.ic3 - paired.ic1
    mean, sd = paired.d.mean(), paired.d.std(ddof=1)
    results["paired_ic3_minus_ic1"] = {
        "months": int(len(paired)),
        "mean_diff": float(mean),
        "t_stat": float(mean / (sd / np.sqrt(len(paired)))) if sd > 0 else None,
    }

    (OUT / "f4_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
