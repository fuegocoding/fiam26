"""F5 — trade triage / no-trade bands on the 175% book (registered 2026-10-25).

Fixed-rule screen for the brief's 5.1.4 trade triage: skip trades whose target
weight change is below a band; measure turnover, costs and net return against
the unbanded book. Drift is ignored (screen, not a backtest). Register:
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
HOLDINGS = ROOT / "final_v2" / "artifacts" / "tfm_daily_tilt_squeeze_175" / "holdings.csv"
MONTHLY = ROOT / "final_v2" / "artifacts" / "tfm_daily_tilt_squeeze_175" / "monthly.json"
PANEL = ROOT / "data" / "raw" / "final" / "chars_final_with_names.parquet"


def main() -> None:
    monthly = pd.DataFrame(json.loads(MONTHLY.read_text()))
    monthly["month"] = pd.to_datetime(monthly.month).dt.to_period("M")
    rf = monthly.set_index("month")["rf_monthly"]
    bench = monthly.set_index("month")["benchmark_monthly"]

    hold = pd.read_csv(HOLDINGS)
    hold["month"] = pd.to_datetime(hold.Date).dt.to_period("M")
    hold["w"] = hold.WEIGHT / 100.0
    w = hold.pivot_table(index="month", columns="PERMNO", values="w").fillna(0.0)

    panel = pd.read_parquet(PANEL, columns=["permno", "eom", "ret_exc_lead1m"])
    panel["eom"] = pd.to_datetime(panel.eom)
    # formation eom maps to the following holding month
    panel["month"] = (panel.eom + pd.offsets.MonthEnd(1)).dt.to_period("M")
    r = panel.pivot_table(index="month", columns="permno", values="ret_exc_lead1m")
    r = r.reindex(index=w.index).reindex(columns=w.columns)
    r_total = r.add(rf.reindex(w.index), axis=0)

    # one-way cost per unit traded from the run itself
    cost_per_unit = float(monthly.cost.sum() / monthly.turnover.sum()) if monthly.turnover.sum() else 0.0

    months = list(w.index)
    results: dict = {
        "config": {
            "months": len(months),
            "names": int(w.shape[1]),
            "cost_per_one_way_unit": cost_per_unit,
            "note": "drift ignored; screen, not a full backtest",
        }
    }

    def run(band: float) -> dict:
        held = np.zeros(w.shape[1])
        rets, turns = [], []
        for i, month in enumerate(months):
            target = w.iloc[i].to_numpy()
            delta = target - held
            trade = np.abs(delta) >= band
            new = np.where(trade, target, held)
            turnover = float(np.abs(new - held).sum() / 2.0)
            held = new
            name_ret = r_total.iloc[i].to_numpy()
            gross = float(np.nansum(held * name_ret))
            net = gross - turnover * cost_per_unit
            rets.append(net)
            turns.append(turnover)
        s = pd.Series(rets, index=pd.Index(months))
        active = s - bench.reindex(months)
        sd = active.std(ddof=1)
        ir = float(active.mean() / sd * np.sqrt(12)) if sd and sd > 0 else None
        return {
            "band_nav": band,
            "ann_turnover": float(np.mean(turns) * 12),
            "ann_cost": float(np.mean(turns) * cost_per_unit * 12),
            "ann_net_return": float(s.mean() * 12),
            "ir_vs_benchmark": ir,
        }

    for band in (0.0, 0.0005, 0.001, 0.0025, 0.005, 0.01):
        results[f"band_{band}"] = run(band)

    (OUT / "f5_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
