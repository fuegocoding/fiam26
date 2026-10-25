"""R1 — valuation gap in the liquid universe (registered 2026-10-25).

Tests the registered hypothesis that the per-year selected pricing models'
residual gap (model-implied minus actual log market equity over assets) carries
cross-sectional return information outside micro caps. Retrospective replay of
a previously-seen window (spec 2.4); see docs/review/research_register_20261025.md.
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
VAL = ROOT / "final" / "output" / "gap_micro_best"


def load_gap() -> pd.DataFrame:
    selection = json.loads((VAL / "selection.json").read_text())["folds"]
    best = {int(y): f["best_pricing_model"] for y, f in selection.items()}
    xs = pd.read_parquet(VAL / "xs_predictions.parquet")
    pooled = pd.read_parquet(VAL / "pooled_predictions.parquet")
    xs["eom"] = pd.to_datetime(xs["eom"]).dt.to_period("M").dt.to_timestamp("M")
    pooled["eom"] = pd.to_datetime(pooled["eom"]).dt.to_period("M").dt.to_timestamp("M")
    pooled = pooled.merge(xs[["permno", "eom", "y"]], on=["permno", "eom"], how="left")
    pooled["M9"] = pooled[["M6", "M8"]].mean(axis=1, skipna=False)
    xs = xs.copy()
    xs["return_year"] = (xs["eom"] + pd.offsets.MonthEnd(1)).dt.year
    rows = []
    for year, model in sorted(best.items()):
        if model in ("M6", "M8", "M9"):
            part = pooled.loc[pooled.fold_year.eq(year)]
        else:
            part = xs.loc[xs.return_year.eq(year)]
        gap = part.loc[:, ["permno", "eom", "y"]].copy()
        gap["gap"] = part[model] - part["y"]
        gap["fold_year"] = year
        rows.append(gap)
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["gap"])


def monthly_ic(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    rows = []
    for eom, part in frame.groupby("eom"):
        part = part.dropna(subset=[score, target])
        if len(part) < 30:
            continue
        rows.append(
            {
                "eom": eom,
                "n": len(part),
                "ic": part[score].rank().corr(part[target].rank()),
            }
        )
    return pd.DataFrame(rows)


def ic_stats(ic: pd.DataFrame) -> dict:
    if ic.empty:
        return {"months": 0}
    mean = ic.ic.mean()
    sd = ic.ic.std(ddof=1)
    t = mean / (sd / np.sqrt(len(ic))) if sd > 0 else np.nan
    return {
        "months": int(len(ic)),
        "mean_ic": float(mean),
        "t_stat": float(t),
        "hit_rate": float((ic.ic > 0).mean()),
    }


def main() -> None:
    gap = load_gap()
    panel = pd.read_parquet(
        PANEL,
        columns=[
            "permno",
            "eom",
            "size_grp",
            "prc",
            "me",
            "dolvol",
            "gics",
            "ticker",
            "ret_exc_lead1m",
            "be_me",
            "bev_mev",
        ],
    )
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel["sector"] = panel.gics.astype("string").str[:2]
    panel["tradable"] = (
        (panel.prc.abs() >= 5)
        & (panel.groupby("eom").dolvol.rank(pct=True) >= 0.2)
        & panel.sector.notna()
    )
    gap["eom"] = pd.to_datetime(gap["eom"])
    data = panel.merge(gap[["permno", "eom", "gap"]], on=["permno", "eom"], how="inner")
    data = data.loc[np.isfinite(data.ret_exc_lead1m) & np.isfinite(data.gap)]
    data = data.loc[data.tradable]
    data["liquid"] = data.size_grp.isin(["small", "large", "mega"])

    results = {"folds": json.loads((VAL / "selection.json").read_text())["folds"]}
    results["sample"] = {
        "rows": int(len(data)),
        "months": int(data.eom.nunique()),
        "window": f"{data.eom.min():%Y-%m} .. {data.eom.max():%Y-%m}",
        "liquid_rows": int(data.liquid.sum()),
        "micro_rows": int((~data.liquid).sum()),
    }

    for label, sub in (
        ("liquid", data.loc[data.liquid]),
        ("micro_small_excluded", data.loc[~data.liquid]),
    ):
        ic = monthly_ic(sub, "gap", "ret_exc_lead1m")
        results[f"ic_{label}"] = ic_stats(ic)
        ic["period"] = np.where(ic.eom < pd.Timestamp("2022-01-01"), "early_2018_2021", "late_2022_2026")
        for period, part in ic.groupby("period"):
            results[f"ic_{label}_{period}"] = ic_stats(part)

    liquid = data.loc[data.liquid].copy()
    # Incremental: does the gap survive controls for the panel's own value chars?
    for col in ("gap", "be_me", "bev_mev"):
        liquid[f"r_{col}"] = liquid.groupby("eom")[col].rank(pct=True)
    rows = []
    for eom, part in liquid.groupby("eom"):
        part = part.dropna(subset=["r_gap", "r_be_me", "r_bev_mev", "ret_exc_lead1m"])
        if len(part) < 50:
            continue
        x = part[["r_gap", "r_be_me", "r_bev_mev"]].to_numpy()
        x = np.column_stack([np.ones(len(x)), x])
        y = part.ret_exc_lead1m.to_numpy()
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            resid = y - x @ beta
            rank_gap = part.r_gap.rank().to_numpy()
            rank_resid = pd.Series(resid).rank().to_numpy()
            rows.append(
                {
                    "eom": eom,
                    "n": len(part),
                    "beta_gap": beta[1],
                    "partial_ic": float(np.corrcoef(rank_gap, rank_resid)[0, 1]),
                }
            )
        except np.linalg.LinAlgError:
            continue
    partial = pd.DataFrame(rows)
    results["incremental"] = {
        "months": int(len(partial)),
        "mean_beta_gap": float(partial.beta_gap.mean()) if len(partial) else None,
        "mean_partial_ic": float(partial.partial_ic.mean()) if len(partial) else None,
        "t_partial_ic": float(
            partial.partial_ic.mean()
            / (partial.partial_ic.std(ddof=1) / np.sqrt(len(partial)))
        )
        if len(partial) > 2
        else None,
    }

    (OUT / "r1_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
