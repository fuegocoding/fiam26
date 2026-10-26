"""Explore the entire signal-concentration curve before implementation or funding."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts.base import canonical_digest
from ..research.edge_curve import (
    broad_positive_regions,
    freeze_prefix_choice,
    mean_evidence,
    signal_prefixes,
    summarize_prefixes,
)
from ..research.pod_plans import PodPlans
from ..research.training_data import POD_FEATURES, prepare_panel
from .common import atomic_write_json, file_sha256
from .configuration import config_path
from .edge_research import HOST_CONTROLS, host_control, source_digest
from .research_access import additional_input_hashes, write_access_audit


def run_edge_diagnostics(config):
    settings, policy = config["training_lab"], config["edge_research"]
    state = config_path(config["application"]["state_dir"])
    parent = Path(
        json.loads((state / "latest.json").read_text())["report"]
    ).parent
    experiment = json.loads((parent / "experiment.json").read_text())
    proposal_path = config_path(policy["proposal_path"])
    proposal = json.loads(proposal_path.read_text())
    fingerprint = {
        "source_sha256": source_digest(),
        "proposal_sha256": file_sha256(proposal_path),
        "panel_sha256": experiment["panel_sha256"],
        "daily_sha256": experiment["daily_sha256"],
        "stage": "all_ranked_prefixes_before_capital",
        "minimum_evidence_months": 24,
        "runtime": experiment["runtime"],
        "training_dates": [settings["start"], settings["end"]],
        **additional_input_hashes(settings),
    }
    output = state / ("edge_scan_" + canonical_digest(fingerprint)[:16])
    output.mkdir(exist_ok=True)
    atomic_write_json(output / "experiment.json", fingerprint)
    atomic_write_json(output / "model_proposals.json", proposal)
    snapshot = output / "source_snapshot"
    root = Path(__file__).parents[1]
    for source in root.rglob("*.py"):
        destination = snapshot / "pod_platform" / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for filename in ("run.py", "pyproject.toml", "plot_edge_scan.py"):
        if (root.parent / filename).exists():
            shutil.copy2(root.parent / filename, snapshot / filename)
    atomic_write_json(snapshot / "config.json", config)
    panel_path, daily_path = (
        config_path(settings[k]) for k in ("panel_path", "daily_path")
    )
    for path, key in (
        (panel_path, "panel_sha256"),
        (daily_path, "daily_sha256"),
    ):
        if file_sha256(path) != experiment[key]:
            raise ValueError(
                "The authorized supplied training snapshot changed"
            )
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
    access_audit = write_access_audit(config, output, panel.metadata)
    plan_reports, prefix_tables, curves, pod_reports = {}, {}, [], {}
    for pod in POD_FEATURES:
        proposed = PodPlans.model_validate(proposal["proposal"][pod])
        plans = [(p, "recorded_model_proposal") for p in proposed.plans]
        plans += [
            (host_control(*row), "host_economic_control")
            for row in HOST_CONTROLS[pod][: 3 - len(plans)]
        ]
        candidates, information_tables = {}, {}
        for plan, origin in plans:
            identity = pod + ":" + plan.name
            prefixes, ic = signal_prefixes(plan, panel)
            information_tables[identity] = ic
            prefixes["strategy_id"] = identity
            candidates[identity], prefix_tables[identity] = prefixes, prefixes
            curve = summarize_prefixes(prefixes)
            curve["strategy_id"] = identity
            curves.append(curve)
            regions = broad_positive_regions(curve)
            plan_reports[identity] = {
                "plan": plan.model_dump(mode="json"),
                "origin": origin,
                "rank_ic": mean_evidence(ic.rank_ic),
                "shortable_rank_ic": mean_evidence(ic.shortable_rank_ic),
                "pointwise_positive_regions": regions,
                "evaluated_prefix_counts": len(curve),
                "capital_assumption": None,
                "maximum_research_names": int(curve.names.max())
                if len(curve)
                else 0,
            }
            ic.to_csv(
                output / (identity.replace(":", "_") + "_rank_ic.csv"),
                index=False,
            )
            print(
                f"Edge curve {identity}: {len(curve)} concentration points, {len(regions)} positive regions",
                flush=True,
            )
        folds, observed, ic_folds, ic_observed, short_ic_observed = (
            [],
            [],
            [],
            [],
            [],
        )
        for year in settings["walk_forward_years"]:
            cutoff = pd.Timestamp(year - 1, 12, 31)
            ranked = []
            for identity, information in information_tables.items():
                prior_ic = mean_evidence(
                    information.loc[
                        information.settlement_date <= cutoff, "rank_ic"
                    ]
                )
                ranked.append(
                    (
                        prior_ic["lower_95"]
                        if prior_ic["lower_95"] is not None
                        else -np.inf,
                        identity,
                    )
                )
            signal_id = max(ranked)[1]
            information = information_tables[signal_id]
            following = information.loc[
                information.execution_date.dt.year == year
            ]
            ic_observed.extend(following.rank_ic.tolist())
            short_ic_observed.extend(following.shortable_rank_ic.tolist())
            ic_folds.append(
                {
                    "year": year,
                    "strategy_id": signal_id,
                    "selection_asof": str(cutoff.date()),
                    "selection_uses_capital_or_name_count": False,
                }
            )
            choice = freeze_prefix_choice(candidates, cutoff)
            folds.append({"year": year, "selection": choice})
            if choice is None:
                continue
            chosen = candidates[choice["strategy_id"]]
            measured = chosen.loc[
                (chosen.names == choice["names"])
                & (chosen.execution_date.dt.year == year)
            ]
            observed.append(measured)
        realized = (
            pd.concat(observed, ignore_index=True)
            if observed
            else pd.DataFrame(columns=["controlled_spread", "gross_spread"])
        )
        evidence = mean_evidence(realized.controlled_spread)
        predictive = mean_evidence(ic_observed)
        ranked = sorted(
            information_tables,
            key=lambda i: (
                plan_reports[i]["rank_ic"]["lower_95"]
                if plan_reports[i]["rank_ic"]["lower_95"] is not None
                else -np.inf
            ),
            reverse=True,
        )
        owned = [ranked[0]]
        for identity in ranked[1:]:
            if len(owned) >= settings["max_strategies_per_pod"]:
                break
            measured = plan_reports[identity]["rank_ic"]
            if (measured["lower_95"] or -np.inf) <= 0:
                continue
            complementary = True
            for other in owned:
                joined = information_tables[identity].merge(
                    information_tables[other],
                    on="formation_date",
                    suffixes=("_first", "_second"),
                )
                correlation = (
                    joined[["rank_ic_first", "rank_ic_second"]]
                    .corr()
                    .iloc[0, 1]
                )
                if (
                    not np.isfinite(correlation)
                    or abs(correlation) > settings["max_strategy_correlation"]
                ):
                    complementary = False
            if complementary:
                owned.append(identity)
            if len(owned) >= settings["max_strategies_per_pod"]:
                break
        pod_reports[pod] = {
            "owned_research_strategies": owned,
            "candidate_archive": list(candidates),
            "ownership_selection": "stable rank predictiveness and complementary monthly IC; no capital or selected book size",
            "folds": folds,
            "signal_selection_folds": ic_folds,
            "frozen_signal_rank_ic": predictive,
            "frozen_shortable_rank_ic": mean_evidence(short_ic_observed),
            "has_stable_predictive_information": bool(
                predictive["months"] >= 24
                and (predictive["lower_95"] or -np.inf) > 0
                and predictive["first_half_mean"] > 0
                and predictive["second_half_mean"] > 0
            ),
            "frozen_prefix_controlled_spread": evidence,
            "frozen_prefix_gross_spread": mean_evidence(realized.gross_spread),
            "consistent_chronological_diagnostic_edge": bool(
                evidence["months"] >= 24
                and (evidence["lower_95"] or -np.inf) > 0
                and evidence["first_half_mean"] > 0
                and evidence["second_half_mean"] > 0
            ),
        }
    pd.concat(prefix_tables.values(), ignore_index=True).to_parquet(
        output / "prefix_monthly_returns.parquet", index=False
    )
    pd.concat(curves, ignore_index=True).to_csv(
        output / "edge_curves.csv", index=False
    )
    report = {
        "status": "completed_signal_edge_diagnostics",
        "output_dir": str(output),
        "fingerprints": fingerprint,
        "data": panel.metadata,
        "research_access": access_audit,
        "capital_assumption": None,
        "fixed_name_count_required": False,
        "models_called": 0,
        "prefix_definition": "equal unit long-tail minus short-tail next-month returns; one unit per leg",
        "controlled_spread": "evaluation-only residual return against each plan's declared controls",
        "plans": plan_reports,
        "pods": pod_reports,
        "all_pods_have_consistent_diagnostic_edge": all(
            p["consistent_chronological_diagnostic_edge"]
            for p in pod_reports.values()
        ),
        "all_pods_have_stable_rank_predictiveness": all(
            p["has_stable_predictive_information"] for p in pod_reports.values()
        ),
        "production_admission": False,
        "limitations": [
            "All 2015-2020 outcomes are authorized development; no period is an untouched final holdout.",
            "Prefix intervals are pointwise and not corrected for this project's multiple searches.",
            "This stage measures one-month predictive strength before entry/exit, spread, borrow, impact or capital.",
            "A diagnostic research prefix can exceed the final fund's 500-name ceiling; deployment must satisfy 200-500 combined holdings.",
            "Positive controlled return spreads are not executable portfolio NAV or demonstrated after-cost alpha.",
        ],
    }
    atomic_write_json(output / "edge_scan_report.json", report)
    atomic_write_json(
        state / "latest_edge_scan.json",
        {"report": str(output / "edge_scan_report.json")},
    )
    return report
