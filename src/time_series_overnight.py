"""Run the frozen 40-idea menu at three cost assumptions, then assemble a research memo."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.time_series_search import ROOT
from src.graph.walk_forward import hac_mean
from scipy.stats import t as student_t


def fmt(value, style=".2f"):
    return format(value, style) if value is not None and pd.notna(value) else "undefined"


def summarize_batch(base):
    frames, paired = [], []
    for cost in [10, 25, 50]:
        folder = base / f"time_series_cost{cost}"
        f = pd.read_csv(folder / "leaderboard.csv", dtype={"experiment": str})
        f["cost_bp"] = cost
        frames.append(f)
        path = folder / "paired_comparisons.csv"
        if path.exists():
            pair = pd.read_csv(path, dtype={"experiment": str, "baseline": str})
            pair["cost_bp"] = cost
            paired.append(pair)
    all_results = pd.concat(frames, ignore_index=True)
    extra_folder = base / "time_series_followups"
    extra_path = extra_folder / "leaderboard.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path, dtype={"experiment": str})
        extra["timesfm_cache_diagnostic"] = False
        all_results = pd.concat([all_results, extra], ignore_index=True)
        extra_pairs = []
        for row in extra.itertuples():
            baseline = {"37": "31", "38": "B2", "39": "B1", "40": "B2"}[row.experiment]
            a = extra_folder / f"{row.experiment}_cost{row.cost_bp}_{row.universe}_{row.mode}_book.csv"
            b = base / f"time_series_cost{row.cost_bp}" / "books" / f"{baseline}_{row.universe}_{row.mode}.csv"
            if not b.exists():
                continue  # A resumed cost batch may not yet have reached this baseline.
            pair = pd.read_csv(a, usecols=["date", "net_excess"]).merge(
                pd.read_csv(b, usecols=["date", "net_excess"]), on="date", suffixes=("_idea", "_baseline")).dropna()
            stats = hac_mean(pair.net_excess_idea - pair.net_excess_baseline, maxlags=6)
            tstat = stats["hac_tstat"]
            extra_pairs.append({"experiment": row.experiment, "baseline": baseline, "cost_bp": row.cost_bp,
                "universe": row.universe, "mode": row.mode, "paired_months": len(pair),
                "incremental_annual_net": stats["mean"] * 12 if stats["mean"] is not None else None,
                "incremental_hac_t": tstat,
                "approximate_p": float(2 * student_t.sf(abs(tstat), max(1, len(pair)-1))) if tstat is not None else 1.})
        if extra_pairs:
            paired.append(pd.DataFrame(extra_pairs))
    nonlinear_folder = base / "time_series_nonlinear"
    nonlinear_path = nonlinear_folder / "leaderboard.csv"
    nonlinear_count = 0
    if nonlinear_path.exists():
        nonlinear = pd.read_csv(nonlinear_path, dtype={"experiment": str})
        nonlinear_count = len(nonlinear)
        all_results = pd.concat([all_results, nonlinear], ignore_index=True)
        nonlinear_pairs = []
        for row in nonlinear.itertuples():
            baseline = {"N29": "29", "N30": "N29", "N31": "31", "N32": "N31", "N39S": "39", "N39T": "N39S"}[row.experiment]
            a = nonlinear_folder / f"{row.experiment}_cost{row.cost_bp}_{row.universe}_{row.mode}_book.csv"
            if baseline.startswith("N"):
                b = nonlinear_folder / f"{baseline}_cost{row.cost_bp}_{row.universe}_{row.mode}_book.csv"
            elif baseline == "39":
                b = extra_folder / f"39_cost{row.cost_bp}_{row.universe}_{row.mode}_book.csv"
            else:
                b = base / f"time_series_cost{row.cost_bp}/books/{baseline}_{row.universe}_{row.mode}.csv"
            if not b.exists():
                continue
            pair = pd.read_csv(a, usecols=["date", "net_excess"]).merge(
                pd.read_csv(b, usecols=["date", "net_excess"]), on="date", suffixes=("_idea", "_baseline")).dropna()
            stats = hac_mean(pair.net_excess_idea - pair.net_excess_baseline, maxlags=6)
            tstat = stats["hac_tstat"]
            nonlinear_pairs.append({"experiment": row.experiment, "baseline": baseline, "cost_bp": row.cost_bp,
                "universe": row.universe, "mode": row.mode, "paired_months": len(pair),
                "incremental_annual_net": stats["mean"]*12 if stats["mean"] is not None else None,
                "incremental_hac_t": tstat,
                "approximate_p": float(2*student_t.sf(abs(tstat), max(1, len(pair)-1))) if tstat is not None else 1.})
        if nonlinear_pairs:
            paired.append(pd.DataFrame(nonlinear_pairs))
    full_folder = base / "time_series_full_characteristics"
    full_count = 0
    if (full_folder / "leaderboard.csv").exists():
        full = pd.read_csv(full_folder / "leaderboard.csv", dtype={"experiment": str})
        full_count = len(full)
        all_results = pd.concat([all_results, full], ignore_index=True)
        full_pairs = []
        for row in full.itertuples():
            baseline = {"F29": "N29", "F30": "F29", "F31": "N31", "F32": "F31"}[row.experiment]
            a = full_folder / f"{row.experiment}_cost{row.cost_bp}_{row.universe}_{row.mode}_book.csv"
            folder = full_folder if baseline.startswith("F") else nonlinear_folder
            b = folder / f"{baseline}_cost{row.cost_bp}_{row.universe}_{row.mode}_book.csv"
            if not b.exists():
                continue
            pair = pd.read_csv(a, usecols=["date", "net_excess"]).merge(
                pd.read_csv(b, usecols=["date", "net_excess"]), on="date", suffixes=("_idea", "_baseline")).dropna()
            stats = hac_mean(pair.net_excess_idea - pair.net_excess_baseline, maxlags=6)
            tstat = stats["hac_tstat"]
            full_pairs.append({"experiment": row.experiment, "baseline": baseline, "cost_bp": row.cost_bp,
                "universe": row.universe, "mode": row.mode, "paired_months": len(pair),
                "incremental_annual_net": stats["mean"]*12 if stats["mean"] is not None else None,
                "incremental_hac_t": tstat,
                "approximate_p": float(2*student_t.sf(abs(tstat), max(1, len(pair)-1))) if tstat is not None else 1.})
        if full_pairs:
            paired.append(pd.DataFrame(full_pairs))
    destination = base / "time_series_batch"
    destination.mkdir(exist_ok=True)
    all_results.to_csv(destination / "all_cost_results.csv", index=False)
    if paired:
        comparisons = pd.concat(paired, ignore_index=True).sort_values("approximate_p")
        comparisons["holm_across_costs_p"] = np.minimum(1, np.maximum.accumulate(
            comparisons.approximate_p.to_numpy() * np.arange(len(comparisons), 0, -1)))
        comparisons["complete_68_month_pair"] = comparisons.paired_months.eq(68)
        comparisons.to_csv(destination / "all_cost_paired_comparisons.csv", index=False)
    main = all_results[all_results.cost_bp == 25].sort_values("information_ratio", ascending=False)
    reliable = main[(main.missing_return_months == 0) & (main.position_compliant_share == 1)
                    & (main.beta_price_proxy.abs() <= .1) & ~main.timesfm_cache_diagnostic
                    & (main.information_ratio > 0)]
    reliable.to_csv(destination / "candidates_clearing_hurdle.csv", index=False)
    lines = ["# Overnight time-series search", "",
        "This is a retrospective hypothesis screen, not an untouched OOS study or a submission-ready strategy.", "",
        f"Completed {len(all_results)} experiment/universe/control/cost combinations: the 40-idea menu plus four baselines has {len(all_results)-nonlinear_count-full_count}/792 completed; compact nonlinear ablations have {nonlinear_count}/108 completed; full-characteristic ablations have {full_count}/72 completed.",
        f"Each combination compares 1/3/6-month holding cohorts on annual two-year validation windows, for {len(all_results)*3:,} logical holding configurations across the three costs.", "",
        "## Main result at 25 bp one way", "",
        f"{len(reliable)} combinations clear the hurdle and the mechanical implementation screen. "
        f"{int((main.active_months == 0).sum())} combinations remained in cash for all test years. "
        f"{int((main.missing_return_months > 0).sum())} combinations have unknown held outcomes.", "",
        "Cash-heavy results can reflect failed net validation or inability to assess a complete validation path. They do not establish that time series are useless.",
        "Any return-path gap prevents interpreting partial-month mean/IR as full-window performance. Read predictive diagnostics separately.", "",
        "## Highest observed IR at 25 bp (including rejected/diagnostic books)", "",
        "| ID | Universe | Mode | Net excess/year | IR | Beta | Active months | Position compliance | Unknown months | Cache diagnostic |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for _, row in main.head(15).iterrows():
        lines.append(f"| {row.experiment} {row['name']} | {row.universe} | {row['mode']} | {fmt(row.annual_net_excess, '.1%')} | {fmt(row.information_ratio)} | {fmt(row.beta_price_proxy)} | {row.active_months} | {fmt(row.position_compliant_share, '.0%')} | {row.missing_return_months} | {row.timesfm_cache_diagnostic} |")
    lines += ["", "## Snapshot versus trajectory models", "",
        "Predictive audits use the model's test forecasts independently of portfolio horizon selection. Surprise models predict surprise ranks; their IC is not a return metric.", "",
        "| Model | Target | Mean annual test rank IC | Mean annual return R² |",
        "|---|---|---:|---:|"]
    for idea in ["29", "30", "31", "32"]:
        audit = json.loads((base / "time_series_cost25" / f"model_{idea}_audit.json").read_text())
        ic = np.mean([r["test_rank_ic"] for r in audit])
        r2 = [r["r2_oos_zero"] for r in audit if r["r2_oos_zero"] is not None]
        lines.append(f"| {idea} | {audit[0]['target']} | {fmt(ic, '.4f')} | {fmt(np.mean(r2) if r2 else None, '.4f')} |")
    if extra_path.exists():
        lines += ["", "## Four alternative prediction targets", "",
            "MSE improvements below compare matched observations against the stated forecasting baseline, not against the stock-return hurdle. Large MSE reductions with negative IC can reflect outliers or a poor baseline.", "",
            "| ID | Target | Mean annual label IC | Matched MSE improvement | Years beating baseline |",
            "|---|---|---:|---:|---:|"]
        for idea in ["37", "38", "39", "40"]:
            audit = json.loads((extra_folder / f"model_{idea}_audit.json").read_text())
            count = sum(r["baseline_rows"] for r in audit)
            mse = sum(r["matched_test_mse"]*r["baseline_rows"] for r in audit)/count
            baseline_mse = sum(r["baseline_mse"]*r["baseline_rows"] for r in audit)/count
            wins = sum(r["matched_test_mse"] < r["baseline_mse"] for r in audit)
            lines.append(f"| {idea} | {audit[0]['target']} | {fmt(np.mean([r['test_rank_ic'] for r in audit]), '.4f')} | {fmt(1-mse/baseline_mse, '.1%')} | {wins}/6 |")
    diagnostics = base / "time_series_cost25/diagnostics/signal_diagnostics.csv"
    if diagnostics.exists():
        d = pd.read_csv(diagnostics, dtype={"experiment": str})
        clean_d = d[~d.timesfm_cache_diagnostic & ~d.entry_signal_only & (d['mode'] == "factor") & (d.universe != "micro")]
        clean_d = clean_d.sort_values("mean_rank_ic", ascending=False)
        lines += ["", "## Predictive signals beyond cash-heavy portfolio selection", "",
            "These statistics use observed stock outcomes; unknown outcomes are counted and may make the labeled sample selective. They are diagnostics, not tradable portfolio returns.", "",
            "| ID | Universe | Factor-adjusted return IC | IC HAC t | Joint diagnostic Holm p | Positive IC years | Unknown outcome share |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for _, row in clean_d.head(12).iterrows():
            lines.append(f"| {row.experiment} {row['name']} | {row.universe} | {fmt(row.mean_rank_ic, '.4f')} | {fmt(row.rank_ic_hac_t)} | {fmt(row.ic_holm_p, '.3f')} | {row.positive_ic_years} | {fmt(row.unknown_outcome_share, '.2%')} |")
    risk = base / "time_series_risk_audit/summary.csv"
    if risk.exists():
        r = pd.read_csv(risk)
        lines += ["", "## Does risk forecasting need histories?", "",
            "This retrospective attribution audit compares matched 68-month labeled samples. The calibrated EWMA baseline learns the log-square target's bias from training data; it avoids crediting a model merely for predicting log variance on a different scale.", "",
            "| Universe | Model | Risk IC | Log-square MSE |",
            "|---|---|---:|---:|"]
        for row in r[r.universe.isin(["large_mega", "investable"])].itertuples():
            lines.append(f"| {row.universe} | {row.model} | {fmt(row.rank_ic, '.4f')} | {fmt(row.mse, '.4f')} |")
    prediction = base / "time_series_prediction_audit/paired_predictive_comparisons.csv"
    if prediction.exists():
        a = pd.read_csv(prediction)
        lines += ["", "## Matched snapshot/trajectory attribution", "",
            "Positive improvement favors histories. These are retrospective diagnostics; the family includes all reported universes, target heads, and both IC/MSE tests.", "",
            "| Pair | Universe | Measure | Improvement | HAC t | Approximate Holm p | Positive years |",
            "|---|---|---|---:|---:|---:|---:|"]
        for row in a[a.universe.isin(["large_mega", "investable"]) & a.measure.eq("rank_ic")].itertuples():
            lines.append(f"| {row.pair} | {row.universe} | {row.measure} | {fmt(row.mean_improvement, '.4f')} | {fmt(row.hac_t)} | {fmt(row.holm_p, '.3f')} | {row.positive_years}/6 |")
    audit_path = destination / "book_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ["", "## Saved-holdings reconciliation", "",
            f"Independently audited {audit['books']} saved books; reconciliation errors: {len(audit['errors'])}. "
            "Checks cover month coverage, duplicates, public eligibility, position counts/exposures, missing outcomes, signed P&L, total-return position drift, and trading costs."]
    lines += ["", "## Limits and next decision", "",
        "The old TimesFM generator selected formation rows using future target availability. Tests 26/27/28/33/34/B3 are diagnostics until safe forecasts are regenerated. Core trajectory models 30/32 exclude that cache.",
        "The generator code is repaired and tested, but actual regeneration requires PyTorch, TimesFM, and its weights, which are absent locally.",
        "Trading costs are one-way assumptions; borrow fees and availability are absent. Alpha/beta use an S&P price-index excess-return proxy.",
        "Approximate Holm adjustments cover completed comparisons, including the three cost assumptions; they cannot undo historical discovery overlap or menu design.",
        "Prioritize a repeatable trajectory-versus-snapshot predictive improvement and a complete, implementable return path. A leaderboard winner alone is insufficient.", "",
        "Detailed outputs: all_cost_results.csv, all_cost_paired_comparisons.csv, candidates_clearing_hurdle.csv; per-cost folders contain validation choices, monthly books, holdings, and model audits."]
    report = destination / "summary.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"Batch memo: {report}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-output", type=Path, default=ROOT / "output")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--nonlinear", action="store_true", help="Also run six boosted-tree snapshot/trajectory ablations")
    ap.add_argument("--full-characteristics", action="store_true", help="Also run four full-147-characteristic controls, including nonlinear ablations")
    args = ap.parse_args()
    if not args.summarize_only:
        for cost in [25, 10, 50]:
            command = [sys.executable, "-u", "-m", "src.time_series_search", "--output",
                       str(args.base_output / f"time_series_cost{cost}")]
            if cost != 25:
                command += ["--cost-bp", str(cost)]
            subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-u", "-m", "src.time_series_diagnostics", "--output",
                        str(args.base_output / "time_series_cost25")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-u", "-m", "src.time_series_followups", "--base-output",
                        str(args.base_output)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-u", "-m", "src.time_series_risk_audit", "--base-output",
                        str(args.base_output)], cwd=ROOT, check=True)
        if args.nonlinear or args.full_characteristics:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / ".deps") + os.pathsep + env.get("PYTHONPATH", "")
            subprocess.run([sys.executable, "-u", "-m", "src.time_series_nonlinear", "--base-output",
                            str(args.base_output)], cwd=ROOT, env=env, check=True)
            if args.full_characteristics:
                subprocess.run([sys.executable, "-u", "-m", "src.time_series_full_characteristics", "--base-output",
                                str(args.base_output)], cwd=ROOT, env=env, check=True)
        subprocess.run([sys.executable, "-u", "-m", "src.time_series_prediction_audit", "--base-output",
                        str(args.base_output)], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "-u", "-m", "src.time_series_book_audit", "--base-output",
                        str(args.base_output)], cwd=ROOT, check=True)
    summarize_batch(args.base_output)


if __name__ == "__main__":
    main()
