"""Four additional, fixed time-series targets: current-consensus error, revisions, risk, dispersion."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph import expectations as ibes
from src.graph.walk_forward import split_masks
from src.time_series_search import (
    ROOT, SNAPSHOT, UNIVERSES, Market, account_selected_book, atomic_json,
    metrics, rank_month, ridge_solve, select_validation, simulate, temporal_features,
)

IDEAS = {
    "37": ("current_consensus_error", "target_error_vs_now_p", "Predict eventual actual EPS minus today's consensus, rather than final pre-announcement consensus."),
    "38": ("next_consensus_revision", "next_revision", "Predict the next month's same-quarter consensus revision."),
    "39": ("next_absolute_return_risk", "next_log_risk", "Predict log squared next-month return; use as an inverse-risk weight tilt on the momentum baseline."),
    "40": ("next_dispersion_change", "next_dispersion", "Predict next month's same-quarter dispersion change; condition current revision direction on expected resolution."),
}


def future_same_quarter(p):
    """Future observations are labels only; an exact quarter/month join prevents rollover artifacts."""
    keys = ["permno", "eom", "target_fpe"]
    nxt = p[keys + ["MEANEST", "STDEV", "STATPERS"]].copy()
    nxt.eom -= pd.offsets.MonthEnd(1)
    nxt = nxt.rename(columns={"MEANEST": "next_estimate", "STDEV": "next_std", "STATPERS": "next_snapshot"})
    return p[keys].merge(nxt, on=keys, how="left", validate="one_to_one")


def forecasts(p, idea, output):
    path = output / f"model_{idea}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    features = list(dict.fromkeys(SNAPSHOT + temporal_features(p) + ["fgap_p", "fgap_disp", "days_since_ann",
        "expected_report_next_month", "rev3_next_p", "exp_growth_p", "sue_vol8_p"]))
    x = (p[features].groupby(p.eom).rank(pct=True) - .5).fillna(0).to_numpy(float)
    x = np.column_stack([x, p[features].isna().to_numpy(float)])
    target = IDEAS[idea][1]
    y = p[target].to_numpy(float)
    label_dates = p.target_ann if idea == "37" else p.eom + pd.offsets.MonthEnd(1)
    audit, frames = [], []
    for year in range(2021, 2027):
        masks = split_masks(p.eom, 1, year, label_dates=label_dates)
        train, val = masks["train"] & np.isfinite(y), masks["validation"] & np.isfinite(y)
        if train.sum() < 500 or val.sum() < 100:
            raise ValueError(f"Insufficient admissible {target} labels in {year}")
        best = None
        for penalty in [.001, .01, .1, 1.]:
            coef = ridge_solve(x[train], y[train], penalty)
            loss = float(np.mean((y[val] - coef[0] - x[val] @ coef[1:]) ** 2))
            if best is None or loss < best[0]:
                best = loss, penalty, coef
        forecast_windows = split_masks(p.eom, 1, year)
        keep = forecast_windows["validation"] | forecast_windows["test"]
        f = p.loc[keep, ["permno", "eom"]].copy()
        f["prediction"] = best[2][0] + x[keep] @ best[2][1:]
        f["fold"] = year
        f["window"] = np.where(forecast_windows["test"][keep], "test", "validation")
        frames.append(f)
        test = forecast_windows["test"] & np.isfinite(y)
        pred = best[2][0] + x[test] @ best[2][1:]
        test_mse = float(np.mean((y[test] - pred) ** 2))
        # Risk persistence baseline is log trailing EWMA variance; others use no-change/seasonal forecasts.
        if idea == "39":
            baseline = np.log(p.loc[test, "ewma_vol"].to_numpy() ** 2 + .0001)
        elif idea == "37":
            baseline = p.loc[test, "fgap_p"].to_numpy(float)
        else:
            baseline = np.zeros(test.sum())
        observed = np.isfinite(baseline)
        baseline_mse = float(np.mean((y[test][observed] - baseline[observed]) ** 2))
        audit.append({"fold": year, "target": target, "features": features, "penalty": best[1],
            "train_rows": int(train.sum()), "validation_rows": int(val.sum()), "test_rows": int(test.sum()),
            "test_rank_ic": ibes.monthly_rank_ic(p.loc[test, "eom"], y[test], pred), "test_mse": test_mse,
            "matched_test_mse": float(np.mean((y[test][observed]-pred[observed])**2)), "baseline_mse": baseline_mse,
            "baseline_rows": int(observed.sum()), "target_is_stock_return": False})
        logging.info("Follow-up %s fold %s fitted", idea, year)
    f = pd.concat(frames, ignore_index=True)
    f.to_parquet(path, index=False)
    atomic_json(output / f"model_{idea}_audit.json", audit)
    return f


def run(base, costs):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    source = base / "time_series_cost25"
    destination = base / "time_series_followups"
    destination.mkdir(exist_ok=True)
    digest = hashlib.sha256((Path(__file__).read_text() + (source / "manifest.json").read_text()).encode()).hexdigest()
    manifest = destination / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["fingerprint"] != digest:
        raise ValueError("Follow-up code/source changed; use a different --base-output.")
    atomic_json(manifest, {"fingerprint": digest, "experiments": IDEAS, "costs": costs,
        "holding_menu": [1, 3, 6], "timing": "Core challenge splits; exact future-month/quarter labels; labels never determine who is forecast."})
    p = pd.read_parquet(source / "panel.parquet")
    nxt = future_same_quarter(p)
    p["next_revision"] = (nxt.next_estimate - p.MEANEST) / p.PRICE
    p["next_dispersion"] = (nxt.next_std - p.STDEV) / p.PRICE
    p["next_log_risk"] = np.log(p.ret_exc_lead1m ** 2 + .0001)
    models = {idea: forecasts(p, idea, destination) for idea in IDEAS}
    m = Market(p)
    m.wide_me = m.wide(p.me)
    original_eligibility = m.eligible.copy()
    old_vol = m.vol.copy()
    rows = []
    for idea, (name, _, description) in IDEAS.items():
        model = models[idea]
        for cost in costs:
            for universe, sizes in UNIVERSES.items():
                m.eligible = original_eligibility & m.wide(p.size_grp.isin(sizes).astype(float), fill=0).astype(bool)
                for mode in ["raw", "factor"]:
                    token = f"{idea}_cost{cost}_{universe}_{mode}"
                    result_file = destination / f"{token}.json"
                    if result_file.exists():
                        r = json.loads(result_file.read_text())
                        rows.append({"experiment": idea, "name": name, "cost_bp": cost, "universe": universe, "mode": mode, **r["summary"]})
                        continue
                    chosen_books, chosen_weights, choices = [], [], []
                    for year in range(2021, 2027):
                        current = model[model.fold == year].set_index(["permno", "eom"]).prediction
                        prediction = current.reindex(pd.MultiIndex.from_frame(p[["permno", "eom"]])).to_numpy()
                        if idea == "39":
                            # A log-risk forecast is a sizing input, not signed alpha or daily volatility/VaR.
                            m.vol = m.wide(np.exp(np.clip(prediction, -12, 4) / 2))
                            score = p.ret_12_1.to_numpy()
                            simulator_idea = "33"
                        elif idea == "40":
                            score = rank_month(p, p.rev3_p) * (1 - rank_month(p, prediction))
                            simulator_idea = idea
                        else:
                            score = prediction
                            simulator_idea = idea
                        wide = m.wide(score)
                        books = {hold: simulate(m, wide, hold, mode, simulator_idea, cost) for hold in [1, 3, 6]}
                        hold, val = select_validation(books, year)
                        template = next(iter(books.values()))
                        selected = (books[hold] if hold is not None else template).loc[template.date.dt.year == year].copy()
                        weights = books[hold].attrs["weights"][selected.index] if hold is not None else np.zeros((len(selected), len(m.ids)), np.float32)
                        selected.attrs = {}
                        if hold is None:
                            for col in ["gross_excess", "net_excess", "cost", "turnover", "n_long", "n_short", "gross_exposure",
                                        "net_exposure", "missing_held_returns", "long_pnl", "short_pnl", "max_position", "exante_beta"]:
                                selected[col] = 0
                            selected["short_me"] = np.nan
                            selected["turnover_uncertain"] = False
                        choices.append({"year": year, "chosen_hold": hold, "validation": val})
                        chosen_books.append(selected)
                        chosen_weights.append(weights)
                    m.vol = old_vol
                    book = pd.concat(chosen_books, ignore_index=True)
                    weights = np.concatenate(chosen_weights)
                    book = account_selected_book(m, book, weights, cost)
                    book.to_csv(destination / f"{token}_book.csv", index=False)
                    month_ix, stock_ix = np.nonzero(weights)
                    pd.DataFrame({"date": book.date.to_numpy()[month_ix], "permno": m.ids[stock_ix],
                        "weight": weights[month_ix, stock_ix]}).to_parquet(destination / f"{token}_holdings.parquet", index=False)
                    summary = metrics(book)
                    atomic_json(result_file, {"experiment": idea, "name": name, "description": description,
                        "cost_bp": cost, "universe": universe, "mode": mode, "summary": summary, "choices": choices})
                    rows.append({"experiment": idea, "name": name, "cost_bp": cost, "universe": universe, "mode": mode, **summary})
                    pd.DataFrame(rows).sort_values("information_ratio", ascending=False, na_position="last").to_csv(destination / "leaderboard.csv", index=False)
                    logging.info("Follow-up %s complete; IR=%s", token, summary["information_ratio"])
    print(f"All four follow-ups complete: {destination}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-output", type=Path, default=ROOT / "output")
    ap.add_argument("--costs", nargs="+", type=int, default=[10, 25, 50])
    args = ap.parse_args()
    run(args.base_output, args.costs)


if __name__ == "__main__":
    main()
