"""Persistent pod strategies, unit-edge selection and continuous capital sizing."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts.base import canonical_digest
from ..research.edge_capacity import (
    allocate_continuously,
    current_capacity,
    measure_edge,
)
from ..research.pod_plans import PodPlans, StrategyPlan, plan_weights
from ..research.training_data import (
    POD_FEATURES,
    market_return_series,
    prepare_panel,
)
from ..research.training_search import objective, project_target, summarize
from ..research.training_tactical import (
    daily_market,
    monthly_net,
    panel_schedule,
    replay_daily,
)
from .common import atomic_write_json, file_sha256
from .configuration import config_path
from .research_access import additional_input_hashes, write_access_audit

HOST_CONTROLS = {
    "fundamentals": [
        (
            "cash_buffer",
            {"cash_at": 1},
            6,
            "sector_beta",
            1,
        ),
        (
            "earnings_quality",
            {"niq_su": 1, "saleq_su": 1, "qmj_prof": 1},
            3,
            "sector_beta",
        ),
    ],
    "events": [
        (
            "net_repurchases",
            {"eqnetis_at": -1, "chcsho_12m": -1, "eqnpo_12m": 1},
            3,
            "sector_beta",
        ),
        (
            "cash_earnings",
            {"oaccruals_at": -1, "taccruals_at": -1, "cash_at": 1},
            3,
            "sector_beta",
        ),
    ],
    "valuation": [
        (
            "cash_quality",
            {"ocf_me": 1, "fcf_me": 1, "qmj_prof": 1},
            3,
            "sector_beta",
        ),
        (
            "low_payout_relative_value",
            {"div12m_me": -1},
            1,
            "sector_beta",
            3,
        ),
    ],
    "dynamics": [
        (
            "residual_trend",
            {"ret_12_1": 1, "resff3_12_1": 1, "ret_6_1": 1},
            3,
            "sector",
        ),
        (
            "trend_after_reversal",
            {"ret_1_0": -1, "ret_12_1": 1},
            1,
            "sector_beta",
        ),
    ],
}


def host_control(name, features, review, controls, smoothing=None):
    return StrategyPlan.model_validate(
        {
            "name": name,
            "economic_mechanism": "A fixed signed characteristic basket is an economic research control; its validity must be measured, never inferred from its name.",
            "falsification": "Use only prior settled training outcomes to select a rule, then measure its subsequent pre-2021 unit returns, factor exposure and fixed-friction drag. Reject economic claims with negative or unstable evidence.",
            "terms": [
                {
                    "feature": f,
                    "other_feature": None,
                    "operation": "identity",
                    "sign": sign,
                }
                for f, sign in features.items()
            ],
            "review_months": review,
            "smoothing_months": smoothing
            if smoothing is not None
            else (3 if review > 1 else 1),
            "controls": controls,
            "entry_rank": 0,
            "exit_rank": 0,
            "expected_holding_months": review,
        }
    )


def source_digest():
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def select_owned(identities, trials, known, settings):
    """Own unit-edge rules independently of dollars and capacity curves."""
    ranked = sorted(
        identities,
        key=lambda i: objective(summarize(trials[i]["gross_monthly"][known])),
        reverse=True,
    )
    if not ranked:
        raise ValueError("A pod cannot have an empty research book")
    selected = [ranked[0]]
    if settings["max_strategies_per_pod"] == 1:
        return selected
    for identity in ranked[1:]:
        net = trials[identity]["gross_monthly"][known]
        if summarize(net)["cagr"] <= 0:
            continue
        correlations = [
            abs(
                float(
                    np.corrcoef(net, trials[other]["gross_monthly"][known])[
                        0, 1
                    ]
                )
            )
            for other in selected
        ]
        if any(
            not np.isfinite(c) or c > settings["max_strategy_correlation"]
            for c in correlations
        ):
            continue
        prior = np.mean(
            [trials[i]["gross_monthly"][known] for i in selected], axis=0
        )
        combined = np.mean(
            [trials[i]["gross_monthly"][known] for i in selected + [identity]],
            axis=0,
        )
        if (
            objective(summarize(combined))
            > objective(summarize(prior)) + settings["min_incremental_score"]
        ):
            selected.append(identity)
        if len(selected) >= settings["max_strategies_per_pod"]:
            break
    return selected


def run_edge_research(config):
    """No model calls: consume recorded proposals and test the exact fixed rules."""
    settings = config["training_lab"]
    policy = config["edge_research"]
    state = config_path(config["application"]["state_dir"])
    parent = Path(
        json.loads((state / "latest.json").read_text())["report"]
    ).parent
    experiment = json.loads((parent / "experiment.json").read_text())
    proposal_path = config_path(policy["proposal_path"])
    evidence = json.loads(proposal_path.read_text())
    if evidence["origin"] not in {"llm_proposal", "llm_proposal_review"}:
        raise ValueError(
            "Use actual recorded model proposals, never a substituted mock"
        )
    if set(evidence["proposal"]) != set(POD_FEATURES):
        raise ValueError("All four pods require their own strategy roster")
    fingerprint = {
        "code_sha256": source_digest(),
        "proposal_sha256": file_sha256(proposal_path),
        "settings": settings,
        "policy": policy,
        "panel_sha256": experiment["panel_sha256"],
        "daily_sha256": experiment["daily_sha256"],
        "runtime": experiment["runtime"],
        **additional_input_hashes(settings),
    }
    output = state / ("edge_" + canonical_digest(fingerprint)[:16])
    output.mkdir(exist_ok=True)
    atomic_write_json(output / "experiment.json", fingerprint)
    atomic_write_json(output / "model_proposals.json", evidence)
    snapshot = output / "source_snapshot"
    package_root = Path(__file__).parents[1]
    for path in package_root.rglob("*.py"):
        target = snapshot / "pod_platform" / path.relative_to(package_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ("run.py", "verify_edge_run.py", "pyproject.toml"):
        if (package_root.parent / name).exists():
            shutil.copy2(package_root.parent / name, snapshot / name)
    atomic_write_json(snapshot / "config.json", config)
    panel_path, daily_path = (
        config_path(settings[k]) for k in ("panel_path", "daily_path")
    )
    for path, key in (
        (panel_path, "panel_sha256"),
        (daily_path, "daily_sha256"),
    ):
        if file_sha256(path) != experiment[key]:
            raise ValueError("Training input changed")
    cache = (
        canonical_digest(
            {
                k: experiment[k]
                for k in ("panel_sha256", "daily_sha256", "settings", "runtime")
            }
        )[:16]
        + "_daily.parquet"
    )
    panel = prepare_panel(panel_path, daily_path, settings, state / cache)
    write_access_audit(config, output, panel.metadata)
    market = daily_market(panel, settings)
    market_monthly = market_return_series(panel)
    trials, rosters = {}, {}
    for pod in POD_FEATURES:
        proposal = PodPlans.model_validate(evidence["proposal"][pod])
        plans = [(p, "recorded_llm") for p in proposal.plans]
        room = 3 - len(plans)
        plans += [
            (host_control(*row), "host_economic_control")
            for row in HOST_CONTROLS[pod][:room]
        ]
        rosters[pod] = []
        for plan, origin in plans:
            identity = pod + ":" + plan.name
            weights = plan_weights(
                plan,
                panel,
                {
                    **settings,
                    "universe_size": policy["strategy_position_limit"],
                },
            )
            schedule = panel_schedule(
                weights, panel, market, plan.review_months
            )
            edge = measure_edge(market, schedule, settings)
            monthly = monthly_net(edge["net"], market, panel)
            gross_monthly = monthly_net(edge["gross"], market, panel)
            metrics = summarize(monthly, market_monthly)
            trials[identity] = {
                "pod": pod,
                "plan": plan,
                "origin": origin,
                "weights": weights,
                "schedule": schedule,
                "edge": edge,
                "monthly": monthly,
                "gross_monthly": gross_monthly,
                "metrics": metrics,
            }
            rosters[pod].append(
                {
                    "strategy_id": identity,
                    "origin": origin,
                    "plan": plan.model_dump(mode="json"),
                    "unit_implementation_metrics": metrics,
                    "signal_edge_metrics": summarize(
                        gross_monthly, market_monthly
                    ),
                    "capital_assumption": None,
                    "quality_status": "needs_chronological_evidence",
                    "falsification_policy": "host chronological protocol; model prose is not executable evidence",
                }
            )
            print(
                f"Unit-edge research {identity}: net CAGR {metrics['cagr']:.3%}",
                flush=True,
            )
        atomic_write_json(output / "research_candidates.json", rosters)
    allocations, scheduled = [], {}
    chronological_pods, unit_frames = {}, []
    late = market.dates >= pd.Timestamp("2018-01-01")
    for identity, trial in trials.items():
        unit_frames.append(
            pd.DataFrame(
                {
                    "date": market.dates,
                    "strategy_id": identity,
                    "gross_return": trial["edge"]["gross"],
                    "fixed_friction_return": trial["edge"]["net"],
                }
            )
        )
    pd.concat(unit_frames, ignore_index=True).to_parquet(
        output / "unit_daily_returns.parquet", index=False
    )
    for pod, roster in rosters.items():
        frozen, selections = {}, []
        for year in settings["walk_forward_years"]:
            cutoff = pd.Timestamp(year - 1, 12, 31)
            known = np.flatnonzero(panel.settlement_dates <= cutoff)
            selected = select_owned(
                [r["strategy_id"] for r in roster], trials, known, settings
            )[0]
            frozen.update(
                {
                    d: w
                    for d, w in trials[selected]["schedule"].items()
                    if market.dates[d].year == year
                }
            )
            selections.append(
                {
                    "year": year,
                    "selection_asof": str(cutoff.date()),
                    "strategy_id": selected,
                }
            )
        replay = measure_edge(market, frozen, settings)
        gross_months = (
            1 + pd.Series(replay["gross"][late], index=market.dates[late])
        ).resample("ME").prod() - 1
        net_months = (
            1 + pd.Series(replay["net"][late], index=market.dates[late])
        ).resample("ME").prod() - 1
        net_metrics = summarize(net_months.to_numpy())
        quality = (
            net_metrics["cagr"] > 0
            and (net_metrics["probabilistic_sharpe_ratio"] or 0) >= 0.95
            and net_metrics["halves"]["first_mean_return"] > 0
            and net_metrics["halves"]["second_mean_return"] > 0
        )
        chronological_pods[pod] = {
            "selections": selections,
            "gross_signal_metrics": summarize(gross_months.to_numpy()),
            "fixed_friction_metrics": net_metrics,
            "quality_evidence_passed": bool(quality),
            "capital_assumption": None,
        }
    atomic_write_json(
        output / "chronological_pod_evidence.json", chronological_pods
    )

    def decision(t, day, *, final=False):
        formation = panel.dates[t]
        known = np.flatnonzero(panel.settlement_dates <= formation)
        known = known[-policy["edge_lookback_months"] :]
        observed_days = (market.dates >= panel.execution_dates[0]) & (
            market.dates <= formation
        )

        def decide(nav, carried):
            identities, targets, expected, impact, caps = [], [], [], [], []
            ownership = {}
            for pod, roster in rosters.items():
                ownership[pod] = (
                    select_owned(
                        [r["strategy_id"] for r in roster],
                        trials,
                        known,
                        settings,
                    )
                    if len(known) >= 2
                    else [roster[0]["strategy_id"]]
                )
                ranked = []
                for row in roster:
                    trial = trials[row["strategy_id"]]
                    history = trial["gross_monthly"][known]
                    if len(history) < policy["minimum_edge_months"]:
                        continue
                    if (
                        np.count_nonzero(np.abs(history) > 1e-12)
                        < policy["minimum_edge_months"]
                    ):
                        continue
                    measured = summarize(history)
                    ranked.append((objective(measured), row["strategy_id"]))
                if not ranked:
                    continue
                identity = max(ranked)[1]
                trial = trials[identity]
                history = trial["gross_monthly"][known]
                # Shrink the expected unit edge, then subtract a declared
                # uncertainty allowance. No dollar amount enters this ranking.
                annual_mean = (
                    history.mean() * 12 * len(history) / (len(history) + 12)
                )
                fixed_drag = float(
                    np.mean(history - trial["monthly"][known]) * 12
                )
                uncertainty = history.std(ddof=1) * 12 / np.sqrt(len(history))
                estimate = float(
                    annual_mean
                    - fixed_drag
                    - policy["edge_uncertainty_haircut"] * uncertainty
                )
                if estimate <= 0:
                    continue
                if final:
                    target = (
                        trial["weights"][t]
                        if formation.month % trial["plan"].review_months == 0
                        else trial["edge"]["holdings"][-1]
                    )
                    prior = trial["edge"]["holdings"][-1]
                else:
                    prior = trial["edge"]["holdings"][day - 1]
                    target = trial["schedule"].get(day, prior)
                bound = current_capacity(
                    target,
                    np.zeros_like(target),
                    panel.frames[t],
                    settings,
                    nav * settings["max_pod_capital"],
                )
                coefficient = float(
                    trial["edge"]["impact_coefficient_daily"][
                        observed_days
                    ].mean()
                    * 252
                )
                identities.append(identity)
                targets.append(target)
                expected.append(estimate)
                impact.append(coefficient)
                caps.append(bound)
            dollars = allocate_continuously(expected, impact, caps, nav)
            target = np.zeros(len(panel.securities))
            for amount, weights in zip(dollars, targets):
                target += amount / nav * weights
            # Net all sleeves, then enforce the combined 500-name mandate.
            # Preserve declared beta/sector exposure; every rule is dollar neutral.
            mask = panel.mask[t]
            control = np.ones((int(mask.sum()), 1))
            reduced = project_target(
                target[mask],
                control,
                settings,
                shortable=panel.frames[t].me.to_numpy()[mask] >= 500,
            )
            target[:] = 0
            target[mask] = reduced
            record = {
                "formation_date": str(formation.date()),
                "decision_nav_usd": nav,
                "funded": [
                    {
                        "strategy_id": i,
                        "capital_usd": float(a),
                        "expected_unit_edge": float(e),
                        "impact_coefficient": float(k),
                        "hard_capacity_usd": float(c),
                    }
                    for i, a, e, k, c in zip(
                        identities, dollars, expected, impact, caps
                    )
                ],
                "cash_usd": float(nav - dollars.sum()),
                "final_frozen_decision": final,
                "owned_strategies": ownership,
            }
            allocations.append(record)
            return target

        return decide

    for t, day in enumerate(panel.execution_dates):
        index = int(market.dates.searchsorted(day))
        scheduled[index] = decision(t, index)
    execution_settings = {
        **settings,
        "execution_start_nav": settings["fund_capital_usd"],
    }
    fund = replay_daily(
        market, scheduled, execution_settings, keep_holdings=True
    )
    final_target = decision(
        len(panel.dates) - 1, len(market.dates), final=True
    )(fund["execution"]["final_nav_usd"], fund["holdings"][-1])
    owned = {
        pod: [
            r
            for r in roster
            if r["strategy_id"] in allocations[-1]["owned_strategies"][pod]
        ]
        for pod, roster in rosters.items()
    }
    for pod, rows in owned.items():
        for row in rows:
            row["pod_chronological_quality_passed"] = chronological_pods[pod][
                "quality_evidence_passed"
            ]
            row["quality_status"] = (
                "pod_evidence_supports_further_capacity_validation"
                if chronological_pods[pod]["quality_evidence_passed"]
                else "requires_further_research"
            )
    atomic_write_json(output / "pod_books.json", owned)
    pd.DataFrame({"date": market.dates, "net_return": fund["net"]}).to_csv(
        output / "daily_returns.csv", index=False
    )
    rows = []
    for day, target in fund["execution_targets"].items():
        for column in np.flatnonzero(np.abs(target) > 1e-10):
            rows.append(
                (
                    market.dates[day],
                    int(panel.securities[column]),
                    target[column],
                )
            )
    pd.DataFrame(
        rows, columns=["execution_date", "permno", "target_weight"]
    ).to_parquet(output / "fund_execution_targets.parquet", index=False)
    atomic_write_json(
        output / "fund_execution_dates.json",
        [str(market.dates[d].date()) for d in sorted(scheduled)],
    )
    atomic_write_json(output / "capital_allocations.json", allocations)
    atomic_write_json(
        output / "frozen_model.json",
        {
            "training_cutoff": settings["end"],
            "pod_books": owned,
            "capital_decision": allocations[-1],
            "targets": [
                {"permno": int(p), "weight": float(w)}
                for p, w in zip(panel.securities, final_target)
                if abs(w) > 1e-10
            ],
            "production_admission": False,
        },
    )
    chronological = market.dates >= pd.Timestamp("2018-01-01")
    all_months = monthly_net(fund["net"], market, panel)
    late_months = (
        1
        + pd.Series(
            fund["net"][chronological], index=market.dates[chronological]
        )
    ).resample("ME").prod() - 1
    report = {
        "status": "completed_edge_then_capacity_development",
        "output_dir": str(output),
        "data": panel.metadata,
        "source_and_input_fingerprints": fingerprint,
        "strategy_selection_uses_dollar_capital": False,
        "capital_grid_used": False,
        "allocation_method": "continuous_marginal_expected_edge_with_inventory_and_staged_liquidity_bounds",
        "model_calls_in_this_run": 0,
        "owned_strategy_counts": {p: len(r) for p, r in owned.items()},
        "all_pods_quality_evidence_passed": all(
            p["quality_evidence_passed"] for p in chronological_pods.values()
        ),
        "production_admission": False,
        "fund": {
            "metrics": summarize(all_months, market_monthly),
            "execution": fund["execution"],
        },
        "chronological_development_metrics": summarize(late_months.to_numpy()),
        "pods": owned,
        "pod_chronological_evidence": chronological_pods,
        "final_capital_decision": allocations[-1],
        "limitations": [
            "All pre-2021 periods are development; contemporary model plans are not historical LLM decisions.",
            "Borrow inventory and fees are disclosed proxies, not actual historical locate quotes.",
            "Impact estimation assumes one-session trades conservatively; actual replay stages fills and charges realized modeled impact.",
            "The CIO nets and compresses to 500 holdings; final fund returns can differ from standalone unit edges.",
            "Nonempty research books do not prove that their strategies possess alpha.",
        ],
    }
    atomic_write_json(output / "research_report.json", report)
    atomic_write_json(
        state / "latest_edge.json",
        {"report": str(output / "research_report.json")},
    )
    return report
