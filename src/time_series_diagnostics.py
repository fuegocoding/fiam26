"""Predictive diagnostics for the frozen time-series menu, independent of portfolio selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.graph.walk_forward import hac_mean
from src.time_series_search import (
    BASELINES, CONTROLS, IDEAS, ROOT, TIMESFM_DIAGNOSTICS,
    rank_month, signal_columns, UNIVERSES,
)


def run(output):
    p = pd.read_parquet(output / "panel.parquet")
    p["holding_month"] = p.eom + pd.offsets.MonthEnd(1)
    p = p[p.holding_month.between("2021-01-31", "2026-08-31")].reset_index(drop=True)
    # Temporal features were computed on the full historical panel, before slicing test years.
    full = pd.read_parquet(output / "panel.parquet")
    signals = signal_columns(full)
    key = pd.MultiIndex.from_frame(p[["permno", "eom"]])
    full_key = pd.MultiIndex.from_frame(full[["permno", "eom"]])
    all_scores = {}
    for idea, _, _ in IDEAS + BASELINES:
        if idea in {"29", "30", "31", "32"}:
            f = pd.read_parquet(output / f"model_{idea}.parquet")
            f = f[f.window == "test"]
            if f.duplicated(["permno", "eom"]).any():
                raise ValueError("Duplicate annual model test forecasts")
            all_scores[idea] = f.set_index(["permno", "eom"]).prediction.reindex(key).to_numpy()
        else:
            all_scores[idea] = pd.Series(np.asarray(signals[idea]), index=full_key).reindex(key).to_numpy()
    controls = np.column_stack([rank_month(p, p[c]).fillna(0) for c in CONTROLS])
    monthly = []
    for universe, sizes in UNIVERSES.items():
        admissible = p.eligible & p.size_grp.isin(sizes)
        for date, group in p[admissible].groupby("holding_month", sort=True):
            ix = group.index.to_numpy()
            x = controls[ix]
            sectors = group.sector.astype(str).to_numpy()
            dummy = np.column_stack([(sectors == s).astype(float) for s in np.unique(sectors)])
            design = np.column_stack([x, dummy])
            actual = group.ret_exc_lead1m.to_numpy(float)
            for idea, name, _ in IDEAS + BASELINES:
                score = all_scores[idea][ix]
                available = np.isfinite(score)
                public = np.flatnonzero(available)
                if len(public) < 20:
                    continue
                # Residualization uses every eligible public score, including unknown outcomes.
                for mode in ["raw", "factor"]:
                    s = score.copy()
                    if mode == "factor":
                        s[available] -= design[available] @ np.linalg.lstsq(design[available], s[available], rcond=None)[0]
                    if np.std(s[available]) <= 1e-12:
                        continue  # Do not turn numerical residue of a fully controlled signal into a rank.
                    observed = available & np.isfinite(actual)
                    if observed.sum() < 20:
                        continue
                    sr = pd.Series(s[observed]).rank(pct=True).to_numpy() - .5
                    yr = pd.Series(actual[observed]).rank(pct=True).to_numpy() - .5
                    ic = float(np.corrcoef(sr, yr)[0, 1]) if sr.std() > 1e-12 and yr.std() > 1e-12 else np.nan
                    # Ex-post regression is a diagnostic, never a fitted forecasting/trading rule.
                    rhs = np.column_stack([sr, design[observed]])
                    slope = float(np.linalg.lstsq(rhs, actual[observed], rcond=None)[0][0])
                    monthly.append({"experiment": idea, "name": name, "universe": universe, "mode": mode,
                        "date": date, "rank_ic": ic, "controlled_return_slope": slope,
                        "eligible_stocks": len(group), "public_scores": int(available.sum()),
                        "observed_outcomes": int(observed.sum()), "unknown_outcomes": int((available & ~np.isfinite(actual)).sum()),
                        "timesfm_cache_diagnostic": idea in TIMESFM_DIAGNOSTICS,
                        "entry_signal_only": idea in {"33", "34", "35", "36"}})
    stats = pd.DataFrame(monthly)
    if stats.empty:
        raise ValueError("No predictive diagnostic observations")
    rows = []
    for (idea, name, universe, mode), group in stats.groupby(["experiment", "name", "universe", "mode"]):
        ic = hac_mean(group.rank_ic, maxlags=6)
        slope = hac_mean(group.controlled_return_slope, maxlags=6)
        def probability(stat):
            t = stat["hac_tstat"]
            return float(2 * student_t.sf(abs(t), max(1, stat["observations"]-1))) if t is not None else 1.
        yearly = {str(y): float(g.rank_ic.mean()) for y, g in group.groupby(group.date.dt.year)}
        rows.append({"experiment": idea, "name": name, "universe": universe, "mode": mode,
            "months": len(group), "mean_rank_ic": ic["mean"], "rank_ic_hac_t": ic["hac_tstat"],
            "controlled_slope": slope["mean"], "controlled_slope_hac_t": slope["hac_tstat"],
            "ic_approximate_p": probability(ic), "slope_approximate_p": probability(slope),
            "unknown_outcome_share": float(group.unknown_outcomes.sum() / group.public_scores.sum()),
            "minimum_yearly_ic": min(yearly.values()), "positive_ic_years": sum(v > 0 for v in yearly.values()),
            "yearly_ic": json.dumps(yearly), "timesfm_cache_diagnostic": idea in TIMESFM_DIAGNOSTICS,
            "entry_signal_only": idea in {"33", "34", "35", "36"}})
    summary = pd.DataFrame(rows)
    # Adjust IC and return-slope tests together rather than selecting whichever metric looks better.
    tests = [(i, col, summary.loc[i, col]) for i in summary.index for col in ["ic_approximate_p", "slope_approximate_p"]]
    tests.sort(key=lambda v: v[2])
    adjusted = np.minimum(1, np.maximum.accumulate(np.array([v[2] for v in tests]) * np.arange(len(tests), 0, -1)))
    for (i, col, _), value in zip(tests, adjusted):
        summary.loc[i, col.replace("approximate_p", "holm_p")] = value
    destination = output / "diagnostics"
    destination.mkdir(exist_ok=True)
    stats.to_parquet(destination / "monthly_signal_statistics.parquet", index=False)
    summary.sort_values("mean_rank_ic", ascending=False).to_csv(destination / "signal_diagnostics.csv", index=False)
    print(f"Wrote {len(summary)} predictive diagnostics to {destination}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "output/time_series_search")
    args = ap.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
