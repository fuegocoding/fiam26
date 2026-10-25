"""F6 — slower-trading variant (registered 2026-10-25).

Same saved target books, different trading policies: trade to target every k
months (k = 1, 2, 3), holding drifted weights in between. Accounting matches
the run: gross = sum(w * excess) + rf on capital, minus borrow and costs.
Drift uses total returns. Register:
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
PANEL = ROOT / "data" / "raw" / "final" / "chars_final_with_names.parquet"
RUNS = ("tfm_daily_tilt_squeeze_175", "risk_book_graph_gate")


def load_inputs(run: str):
    art = ROOT / "final_v2" / "artifacts" / run
    monthly = pd.DataFrame(json.loads((art / "monthly.json").read_text()))
    monthly["month"] = pd.to_datetime(monthly.month).dt.to_period("M")
    monthly = monthly.set_index("month")

    hold = pd.read_csv(art / "holdings.csv")
    hold["month"] = pd.to_datetime(hold.Date).dt.to_period("M")
    hold["w"] = hold.WEIGHT / 100.0
    w = hold.pivot_table(index="month", columns="PERMNO", values="w").fillna(0.0)

    panel = pd.read_parquet(PANEL, columns=["permno", "eom", "ret_exc_lead1m"])
    panel["eom"] = pd.to_datetime(panel.eom)
    panel["month"] = (panel.eom + pd.offsets.MonthEnd(1)).dt.to_period("M")
    r = panel.pivot_table(index="month", columns="permno", values="ret_exc_lead1m")
    r = r.reindex(index=w.index).reindex(columns=w.columns).fillna(0.0)
    return w, r, monthly


def simulate(w: pd.DataFrame, r: pd.DataFrame, monthly: pd.DataFrame, step: int, cost_per_unit: float) -> dict:
    months = list(w.index)
    rf = monthly.reindex(months)["rf_monthly"].to_numpy()
    borrow = monthly.reindex(months)["borrow"].to_numpy()
    bench = monthly.reindex(months)["benchmark_monthly"].to_numpy()
    held = np.zeros(w.shape[1])
    rets, turns = [], []
    for i, month in enumerate(months):
        target = w.iloc[i].to_numpy()
        new = target if i % step == 0 else held
        turnover = float(np.abs(new - held).sum() / 2.0)
        held = new
        exc = r.iloc[i].to_numpy()
        total = exc + rf[i]
        r_p = float((held * exc).sum()) + rf[i] - borrow[i] - turnover * cost_per_unit
        rets.append(r_p)
        turns.append(turnover)
        nav_growth = 1.0 + float((held * total).sum())
        held = held * (1.0 + total) / nav_growth if nav_growth != 0 else held * (1.0 + total)
    s = pd.Series(rets)
    active = s - pd.Series(bench)
    sd = active.std(ddof=1)
    ir = float(active.mean() / sd * np.sqrt(12)) if sd > 0 else None
    return {
        "step_months": step,
        "ann_turnover": float(np.mean(turns) * 12),
        "ann_cost": float(np.mean(turns) * cost_per_unit * 12),
        "ann_net_return": float(s.mean() * 12),
        "ann_vol": float(s.std(ddof=1) * np.sqrt(12)),
        "ir_vs_benchmark": ir,
    }


def main() -> None:
    results: dict = {}
    for run in RUNS:
        w, r, monthly = load_inputs(run)
        cost_per_unit = float(monthly.cost.sum() / monthly.turnover.sum()) if monthly.turnover.sum() else 0.0016
        entry = {"cost_per_one_way_unit": cost_per_unit, "months": int(len(w))}
        net = monthly["net_return"]
        active = net - monthly["benchmark_monthly"]
        sd = active.std(ddof=1)
        entry["reported_book"] = {
            "ann_net_return": float(net.mean() * 12),
            "ir_vs_benchmark": float(active.mean() / sd * np.sqrt(12)) if sd > 0 else None,
            "ann_turnover": float(monthly.turnover.mean() * 12),
        }
        sims = {}
        for step in (1, 2, 3):
            sims[step] = simulate(w, r, monthly, step, cost_per_unit)
            entry[f"sim_step_{step}"] = sims[step]
        # reconciliation check of the step-1 simulation against the book
        exc = r.to_numpy()
        held_path = []
        s1 = []
        for i in range(len(w)):
            s1.append(
                float((w.iloc[i].to_numpy() * exc[i]).sum())
                + monthly.rf_monthly.iloc[i]
                - monthly.borrow.iloc[i]
            )
        entry["recon_corr_step1_vs_reported_net"] = float(np.corrcoef(s1, net.to_numpy())[0, 1])
        results[run] = entry

    (OUT / "f6_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
