"""Compare directional trend rules with matched passive exposure in training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pod_platform.application.common import atomic_write_json, file_sha256
from pod_platform.application.configuration import config_path, load_config
from pod_platform.contracts.base import canonical_digest
from pod_platform.research.desk_metrics import compute_desk_metrics
from pod_platform.research.training_data import prepare_panel
from pod_platform.research.training_search import objective, summarize
from pod_platform.research.training_tactical import daily_market, replay_daily


def market_index(panel, market):
    """Use the previous session's known membership, including departed firms."""
    values = np.full(len(market.dates), np.nan)
    for day in range(1, len(values)):
        formation = market.formation[day - 1]
        if formation < 0:
            continue
        returns = market.returns[day, panel.mask[formation]]
        finite = returns[np.isfinite(returns) & (returns >= -1)]
        if len(finite):
            values[day] = finite.mean()
    return values


def directional_schedule(panel, market, index, recipe):
    """Form a 40% sleeve at close; trade the next session's close."""
    logs = pd.Series(np.log1p(index), index=market.dates)
    lookback = recipe["lookback"]
    trend = logs.rolling(lookback, min_periods=lookback).sum().to_numpy()
    schedule = {}
    for day, formation in enumerate(market.formation[:-1]):
        if formation < 0 or not np.isfinite(trend[day]):
            continue
        now, following = market.dates[day : day + 2]
        if recipe["review"] == "weekly":
            if now.isocalendar()[:2] == following.isocalendar()[:2]:
                continue
        elif now.to_period("M") == following.to_period("M"):
            continue
        exposure = 0.4
        if recipe["rule"] != "passive" and trend[day] <= 0:
            exposure = 0 if recipe["rule"] == "long_cash" else -0.4
        frame = panel.frames[formation]
        eligible = panel.mask[formation].copy()
        if exposure < 0:
            eligible &= frame.me.to_numpy() >= 500
        adv = frame.dolvol_126d.fillna(0).to_numpy()
        names = np.flatnonzero(eligible)
        names = names[np.argsort(-adv[names], kind="stable")[:500]]
        if recipe["universe"] == "low_volatility":
            sigma = frame.rvol_21d.fillna(1).to_numpy()
            names = names[np.argsort(sigma[names], kind="stable")]
        names = names[:300]
        target = np.zeros(len(panel.securities), dtype=np.float32)
        target[names] = exposure / max(1, len(names))
        schedule[day + 1] = target
    return schedule


def metrics(returns, reference):
    measured = compute_desk_metrics(returns, periods_per_year=252)
    valid = np.isfinite(reference)
    x = np.column_stack([np.ones(valid.sum()), reference[valid]])
    intercept, beta = np.linalg.lstsq(x, returns[valid], rcond=None)[0]
    return {
        **measured,
        "market_alpha_annualized": float(intercept * 252),
        "realized_market_beta": float(beta),
    }


def run_dynamics_diagnostic():
    config = load_config()
    settings = config["training_lab"]
    state = config_path(config["application"]["state_dir"])
    parent = Path(
        json.loads((state / "latest.json").read_text())["report"]
    ).parent
    experiment = json.loads((parent / "experiment.json").read_text())
    if settings != experiment["settings"]:
        raise ValueError(
            "Diagnostic must use the completed run's exact settings"
        )
    panel_path, daily_path = (
        config_path(settings[k]) for k in ("panel_path", "daily_path")
    )
    for path, key in (
        (panel_path, "panel_sha256"),
        (daily_path, "daily_sha256"),
    ):
        if file_sha256(path) != experiment[key]:
            raise ValueError("Diagnostic input differs from its parent run")
    cache_key = canonical_digest(
        {
            k: experiment[k]
            for k in ("panel_sha256", "daily_sha256", "settings", "runtime")
        }
    )[:16]
    panel = prepare_panel(
        panel_path, daily_path, settings, state / (cache_key + "_daily.parquet")
    )
    market = daily_market(panel, settings)
    reference = market_index(panel, market)
    # This is a separate pod diagnostic; its own cash periods do not satisfy
    # the combined fund's holdings floor. No admission is implied by returns.
    execution = {**settings, "execution_max_absolute_net": 0.49}
    recipes = [
        {
            "lookback": lookback,
            "review": review,
            "rule": rule,
            "universe": universe,
        }
        for lookback in (63, 126, 252)
        for review in ("weekly", "monthly")
        for rule in ("long_cash", "long_short", "passive")
        for universe in ("liquid", "low_volatility")
    ]
    output = parent / "directional_diagnostic"
    output.mkdir(exist_ok=True)
    source = Path(__file__).read_bytes()
    (output / "diagnose_dynamics.py").write_bytes(source)
    schedules, series, records = {}, {}, []
    observed = market.dates >= panel.execution_dates[0]
    late = market.dates >= pd.Timestamp("2018-01-01")
    for number, recipe in enumerate(recipes):
        identity = f"trend_{number:02d}"
        schedule = directional_schedule(panel, market, reference, recipe)
        book = replay_daily(market, schedule, execution)
        schedules[identity], series[identity] = schedule, book["net"]
        records.append(
            {
                "strategy_id": identity,
                "recipe": recipe,
                "training": metrics(book["net"][observed], reference[observed]),
                "late_development": metrics(book["net"][late], reference[late]),
                "execution": book["execution"],
            }
        )
        print(f"Directional diagnostic {number + 1}/{len(recipes)}", flush=True)
    folds, frozen = [], {}
    for year in settings["walk_forward_years"]:
        cutoff = pd.Timestamp(year - 1, 12, 31)
        candidates = []
        for record in records:
            if record["recipe"]["rule"] == "passive":
                continue
            prior = pd.Series(series[record["strategy_id"]], index=market.dates)
            prior = prior.loc[
                (prior.index >= panel.execution_dates[0])
                & (prior.index <= cutoff)
            ]
            months = (1 + prior).resample("ME").prod().to_numpy() - 1
            score = summarize(months)
            if score["cagr"] > 0:
                candidates.append((objective(score), record["strategy_id"]))
        selected = max(candidates)[1] if candidates else None
        if selected:
            frozen.update(
                {
                    d: w
                    for d, w in schedules[selected].items()
                    if market.dates[d].year == year
                }
            )
        else:
            day = int(market.dates.searchsorted(pd.Timestamp(year, 1, 1)))
            frozen[day] = np.zeros(len(panel.securities))
        folds.append(
            {
                "year": year,
                "selection_asof": str(cutoff.date()),
                "selected": selected,
            }
        )
    book = replay_daily(market, frozen, execution)
    summary = {
        "parent_run_hash": experiment["run_hash"],
        "scope": "exploratory_development_only_no_2021_plus_outcomes",
        "production_admission": False,
        "trial_count": len(recipes),
        "reference": "contemporaneous_equal_weight_supplied_universe_not_SP500",
        "source_sha256": file_sha256(Path(__file__)),
        "fund_constraints_evaluated": False,
        "cash_periods": "pod_may_be_flat_combined_fund_still_requires_200_names",
        "trials": records,
        "folds": folds,
        "chronological": metrics(book["net"][late], reference[late]),
        "execution": book["execution"],
    }
    atomic_write_json(output / "report.json", summary)
    pd.DataFrame(
        {
            "date": market.dates[observed],
            "reference": reference[observed],
            "frozen_selection": book["net"][observed],
            **{k: v[observed] for k, v in series.items()},
        }
    ).to_parquet(output / "daily_returns.parquet", index=False)
    return summary


if __name__ == "__main__":
    run_dynamics_diagnostic()
