"""One resumable path from evidence and agent code to executed portfolio evidence."""

from __future__ import annotations

import ast
import copy
import datetime as dt
import json
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import ValidationError

from ..prompts import PROMPTS_PATH, load_prompt
from ..agents.provider import (
    GenerationParams,
    LLMRequest,
    ProviderError,
)
from ..contracts.base import canonical_digest
from ..pods.charters import pod_spec
from ..papers import PaperService, PaperStore
from ..papers.transport import UrllibTransport
from ..research.campaign_metrics import competition_report
from ..research.causal_forecast import fit_ridge_forecast
from ..research.candidate_review import candidate_history, compare_refinement, review_candidate, supported_candidates
from ..research.research_loop import TopicResearch
from ..research.strategy_scope import scope_diagnostics, scoped_panel
from ..research.decision_filings import DecisionFilingFeatures
from ..research.edge_capacity import measure_edge
from ..research.edge_curve import (
    freeze_prefix_choice,
    mean_evidence,
    signal_prefixes,
    summarize_prefixes,
)
from ..research.feature_documentation import characteristic_definitions
from ..research.filing_access import FilingResearchStore
from ..research.filing_text import (
    TEXT_FEATURES,
    VERSION,
    custom_feature_arrays,
    custom_measurements,
    text_feature_arrays,
    text_measurements,
)
from ..research.forecast_construction import monthly_concentration
from ..research.input_permissions import FILING_RESEARCH_FEATURES
from ..research.joint_allocation import allocate_jointly
from ..research.mandate import DEFAULTS, validate_part2_settings
from ..research.submission import write_bundle
from ..research.sleeve_execution import SleeveExecutionBook
from ..research.strategy_code import (
    CodeBatch,
    CodeStrategy,
    ResearchTopic,
    TopicPlan,
    ProposalBatch,
    code_signals,
    compile_strategy,
    required_features,
    unit_schedule,
)
from ..research.training_data import POD_FEATURES, prepare_panel
from ..research.training_search import summarize
from ..research.training_tactical import (
    daily_market,
    monthly_net,
    replay_daily,
    unit_return_month_ends,
)
from .campaign_sources import (
    attach_alternative_features,
    discover_evidence,
    load_reporting_benchmarks,
    paper_relevant,
)
from .common import atomic_write_json, file_sha256, now_utc
from .configuration import config_path
from .common import source_digest
from .live_models import create_live_models
from .memory_control import (
    collect_research_memory,
    memory_sample,
    memory_watchdog,
    research_job_lock,
    spill_array,
    spill_schedule,
)
from .research_access import additional_input_hashes, write_access_audit


def select_roster(trials, pod, known, minimum=24, *, uncertainty_haircut=1.0):
    """Own up to three supported, distinct hypotheses; an empty book is valid."""
    return supported_candidates(trials, pod, known, minimum,
        uncertainty_haircut=uncertainty_haircut, maximum=3, maximum_correlation=.85)


def compact_response_schema(value):
    if isinstance(value, dict):
        return {k: compact_response_schema(v) for k, v in value.items()
            if k not in {"title", "description", "default"}}
    return [compact_response_schema(v) for v in value] if isinstance(value, list) else value


class ResearchCampaign:
    def __init__(self, config, *, providers=None, evidence_services=None):
        self.config = config
        self.settings = {
            **DEFAULTS,
            **config["training_lab"],
            "reject_ineligible_short_orders": True,
            "compact_storage": True,
        }
        for key in ("panel_path", "daily_path", "filings_path"):
            if key in self.settings:
                self.settings[key] = str(config_path(self.settings[key]))
        validate_part2_settings(self.settings)
        self.policy = config["research_campaign"]
        for key in ('earnings_calendar_path', 'expected_sessions_path'):
            if self.policy.get('submission', {}).get(key):
                self.policy['submission'][key] = str(config_path(self.policy['submission'][key]))
        self.evaluation_period = self.policy.get("evaluation_period")
        self.portfolio_evidence_scope = self.policy.get("portfolio_evidence_scope", "prior_settled")
        if self.portfolio_evidence_scope not in {"prior_settled", "retrospective_evaluation_period"}:
            raise ValueError("Unknown portfolio evidence scope")
        if self.evaluation_period:
            begin, end = (pd.Timestamp(self.evaluation_period[k]) for k in ("start", "end"))
            if begin > end or begin < pd.Timestamp(self.settings["start"]) or end > pd.Timestamp(self.settings["end"]):
                raise ValueError("Evaluation period must be ordered and inside the supplied data prefix")
        elif self.portfolio_evidence_scope != "prior_settled":
            raise ValueError("Retrospective evidence requires an explicit evaluation period")
        self.state = config_path(config["application"]["state_dir"])
        self.state.mkdir(parents=True, exist_ok=True)
        self.providers = providers
        self.evidence_services = evidence_services or {}
        self.trials, self.attempts = {}, []
        self.calls = 0
        self.custom_feature_specs = {}
        self.decision_filings = None
        self.feature_evidence = {}
        documentation = self.policy.get("feature_documentation_path")
        self.feature_definitions = (
            characteristic_definitions(config_path(documentation))
            if documentation and config_path(documentation).exists()
            else {}
        )

    def _save(self):
        atomic_write_json(self.output / "attempt_ledger.json", self.attempts)

    def _evaluate(
        self,
        strategy,
        origin,
        round_number,
        provenance=None,
        unavailable_features=(),
    ):
        if origin not in {"agent_written_code", "recorded_agent_code"}:
            raise ValueError("Only agent-proposed strategies may enter the research pool")
        variant = strategy.model_dump(mode="json")
        for key in ("name", "mechanism", "falsification", "evidence_refs", "research_topic", "parent_strategy_id"):
            variant.pop(key)
        variant["universe"].pop("rationale")
        try:
            variant["code"] = ast.dump(ast.parse(strategy.code), include_attributes=False)
        except SyntaxError:
            pass  # Record the syntax rejection through the normal evaluator.
        variant_id = canonical_digest(variant)
        duplicate = next((i for i, t in self.trials.items() if t.get("variant_id") == variant_id), None)
        if duplicate is not None:
            self.attempts.append({"pod": strategy.pod, "round": round_number,
                "status": "duplicate_variant", "strategy": strategy.model_dump(mode="json"),
                "duplicate_of": duplicate})
            self._save()
            return
        identity = (
            strategy.pod
            + ":"
            + strategy.name
            + ":"
            + canonical_digest(strategy.model_dump(mode="json"))[:10]
        )
        if identity in self.trials:
            return
        record = {
            "strategy_id": identity,
            "origin": origin,
            "round": round_number,
            "strategy": strategy.model_dump(mode="json"),
            "status": "registered_before_evaluation",
            "model_provenance": provenance,
        }
        self.attempts.append(record)
        self._save()
        try:
            dependencies = set(required_features(strategy.code))
            if strategy.forecast_model != "code":
                dependencies.update(
                    strategy.model_features
                )
            if dependencies & set(unavailable_features):
                raise ValueError(
                    "Strategy requires a rejected text measurement definition"
                )
            compile_strategy(strategy.code, set(self.panel.features) | (
                {"model_forecast"} if strategy.forecast_model != "code" else set()
            ))
            unknown_refs = set(strategy.evidence_refs) - self.evidence_ids
            if unknown_refs:
                raise ValueError(f"Unknown evidence_refs {sorted(unknown_refs)}; use an exact supplied evidence_id or [] for an original hypothesis")
            strategy_panel = scoped_panel(self.panel, strategy.universe)
            scope_report = scope_diagnostics(strategy_panel, strategy.universe,
                self._known_outcomes()[-strategy.evidence_lookback_months:])
            if scope_report["maximum_names"] == 0:
                raise ValueError(json.dumps({"reason": "empty_declared_universe", **scope_report}))
            model_state = None
            if strategy.forecast_model != "code":
                if "model_forecast" not in required_features(strategy.code):
                    raise ValueError(
                        "For ridge, use def score(features, history, daily): return features['model_forecast']; the host supplies this vector"
                    )
                model_inputs = strategy.model_features
                fit = fit_ridge_forecast
                extra = {}
                if strategy.forecast_model == "xgboost":
                    from ..research.causal_tree_forecast import fit_tree_forecast
                    fit = fit_tree_forecast
                    extra = {"parameters": strategy.model_parameters}
                else:
                    extra = {"controls": strategy.controls}
                predicted, model_state = fit(
                    strategy_panel,
                    model_inputs,
                    penalty=strategy.model_penalty,
                    lookback=strategy.model_lookback_months,
                    **extra,
                )
                strategy_panel = replace(
                    strategy_panel,
                    features={
                        **strategy_panel.features,
                        "model_forecast": predicted,
                    },
                )
            compile_strategy(strategy.code, set(strategy_panel.features))
            readiness = {}
            scores, decisions = code_signals(
                strategy, strategy_panel, self.market, readiness=readiness,
                decision_feature_provider=self.decision_filings,
            )
            stem = identity.replace(":", "_")
            spill_schedule(self.output / (stem + "_native_forecasts.npy"),
                {day: value for day, (_, value) in decisions.items()})
            atomic_write_json(self.output / (stem + "_native_forecasts.json"),
                {str(day): int(t) for day, (t, _) in decisions.items()})
            inception = min(
                (day for day in decisions if readiness.get(day)),
                default=len(self.market.dates),
            )
            if inception == len(self.market.dates):
                raise ValueError(json.dumps({"reason": "no_available_forecasts_in_declared_cohort_or_fitted_history", **scope_report}))
            if model_state is not None:
                ready = {
                    i
                    for i, row in enumerate(model_state["states"])
                    if row.get("fitted", row.get("coefficients") is not None)
                }
                inception = min(
                    (
                        day
                        for day, (t, score) in decisions.items()
                        if t in ready and readiness.get(day)
                    ),
                    default=len(self.market.dates),
                )
            evidence_available = unit_return_month_ends(self.panel) >= (
                self.market.dates[inception]
                if inception < len(self.market.dates)
                else pd.Timestamp.max
            )
            monthly_scores = np.full(self.panel.mask.shape, np.nan)
            # For daily/weekly rules the latest review preceding each monthly
            # formation is recomputed independently, never taken from a later day.
            for t, formation in enumerate(self.panel.dates[:-1]):
                execution = int(
                    self.market.dates.searchsorted(
                        self.panel.execution_dates[t]
                    )
                )
                eligible = [
                    (d, pair[1])
                    for d, pair in decisions.items()
                    if d <= execution and pair[0] <= t
                ]
                if eligible:
                    monthly_scores[t] = max(eligible, key=lambda x: x[0])[1]
            prefixes, information = signal_prefixes(
                strategy, strategy_panel, signal=monthly_scores
            )
            curves = summarize_prefixes(prefixes)
            if model_state is not None:
                atomic_write_json(
                    self.output / (stem + "_causal_model.json"), model_state
                )
            curves.to_csv(
                self.output / (stem + "_concentration.csv"), index=False
            )
            information.to_csv(
                self.output / (stem + "_information.csv"), index=False
            )
            prefixes.to_parquet(
                self.output / (stem + "_prefixes.parquet"), index=False
            )
            # Choose a supported region using only prior years. The fund mandate
            # limits construction to 500; it does not truncate research rankings.
            limits, schedule = {}, {}
            executable_curve = monthly_concentration(
                strategy, decisions, strategy_panel, self.market, self.settings
            )
            if executable_curve is not None:
                executable_curve.to_parquet(
                    self.output / (stem + "_executable_concentration.parquet"),
                    index=False,
                )
            for year in sorted(self.market.dates.year.unique()):
                limits[year] = (
                    freeze_prefix_choice(
                        {identity: executable_curve},
                        pd.Timestamp(year - 1, 12, 31),
                        self.policy["minimum_edge_months"],
                        lookback_months=strategy.evidence_lookback_months,
                    )
                    if executable_curve is not None
                    else None
                )
            designs_cache = {}
            for day, pair in decisions.items():
                year = self.market.dates[day].year
                choice = limits[year]
                maximum = (
                    choice["names"]
                    if choice
                    else self.settings["universe_size"]
                )
                designs_cache[day] = (pair, maximum)
            # Group by concentration to avoid rebuilding all construction matrices.
            for maximum in sorted({x[1] for x in designs_cache.values()}):
                group = {
                    d: pair
                    for d, (pair, count) in designs_cache.items()
                    if count == maximum
                }
                schedule.update(
                    unit_schedule(
                        strategy, group, strategy_panel, self.settings, maximum
                    )
                )
            schedule = spill_schedule(
                self.output / (stem + "_schedule.npy"), schedule
            )
            edge = measure_edge(
                self.market,
                schedule,
                self.settings,
                keep_holdings=False,
                snapshot_days=sorted(set([int(self.market.dates.searchsorted(date))
                               for date in self.panel.execution_dates]) | set(self._first_session_reviews())),
            )
            edge["prior_holdings"] = spill_schedule(
                self.output / (stem + "_cio_prior_holdings.npy"),
                {day: value.astype(np.float32)
                 for day, value in edge["prior_holdings"].items()},
            )
            atomic_write_json(self.output / (stem + "_cio_prior_dates.json"),
                [str(self.market.dates[day].date()) for day in sorted(edge["prior_holdings"])])
            atomic_write_json(
                self.output / (stem + "_schedule_dates.json"),
                [
                    str(self.market.dates[day].date())
                    for day in sorted(schedule)
                ],
            )
            unit_monthly = monthly_net(edge["returns"], self.market, self.panel)
            outcomes = unit_return_month_ends(self.panel)
            recent = np.flatnonzero((outcomes <= pd.Timestamp(self.settings["end"])) & evidence_available)[-strategy.evidence_lookback_months:]
            if not len(recent):
                raise ValueError("No settled forecast evidence is available inside the declared cohort and history window")
            first = outcomes[recent[0]].to_period("M").start_time if len(recent) else pd.Timestamp.max
            last = outcomes[recent[-1]] if len(recent) else pd.Timestamp.min
            daily_evidence = (self.market.dates >= first) & (self.market.dates <= last)
            information = information.loc[information.formation_date.isin(self.panel.dates[recent])
                & (information.settlement_date <= pd.Timestamp(self.settings["end"]))]
            report = {
                "universe": strategy.universe.model_dump(mode="json"),
                "cohort_coverage": scope_report,
                "evidence_lookback_months": strategy.evidence_lookback_months,
                "validation_window_months": strategy.validation_window_months,
                "strategy_evidence_inception": str(
                    self.market.dates[inception].date()
                )
                if inception < len(self.market.dates)
                else None,
                "post_inception_months": int(evidence_available.sum()),
                "feedback_window_start": str(first.date()) if len(recent) else None,
                "feedback_window_end": str(last.date()) if len(recent) else None,
                "warmup_policy": "Pre-fit ridge warmup stays in total returns but is excluded from strategy evidence and funding history; legitimate idle periods after inception remain",
                "unit_metrics": summarize(unit_monthly[recent]),
                "chronological_development_unit_metrics": summarize(
                    unit_monthly[recent[len(recent) // 2:]]
                ),
                "rank_ic": mean_evidence(information.rank_ic),
                "shortable_rank_ic": mean_evidence(
                    information.shortable_rank_ic
                ),
                "prior_only_concentration_choices": limits,
                "concentration_points": len(curves),
                "concentration_selection": "identical_constructor_native_monthly_retention"
                if executable_curve is not None
                else "predeclared_fund_ceiling_no_monthly_proxy_count_selection",
                "capital_assumption": None,
                "execution_eligible_short_rejections": edge[
                    "rejected_short_orders"
                ],
                "unit_attribution": {
                    "annualized_long_contribution": float(
                        np.mean(
                            edge["long_contribution"][
                                daily_evidence
                            ]
                        )
                        * 252
                    ),
                    "annualized_short_or_hedge_contribution": float(
                        np.mean(
                            edge["short_contribution"][
                                daily_evidence
                            ]
                        )
                        * 252
                    ),
                },
            }
            if not np.allclose(
                edge["long_contribution"] + edge["short_contribution"],
                edge["returns"],
                atol=1e-12,
            ):
                raise ValueError(
                    "Unit long/short contribution failed to reconcile"
                )
            pd.DataFrame(
                {"date": self.market.dates, "returns": edge["returns"]}
            ).to_parquet(
                self.output / (stem + "_unit_returns.parquet"), index=False
            )
            self.trials[identity] = {
                "variant_id": variant_id,
                "strategy": strategy,
                "origin": origin,
                "schedule": schedule,
                "edge": edge,
                "monthly": unit_monthly,
                "evidence_available": evidence_available,
                "report": report,
            }
            record.update(status="evaluated", evidence=report)
            atomic_write_json(self.output / (stem + ".json"), record)
            (self.output / (stem + ".py")).write_text(strategy.code)
            print(
                f"Campaign {identity}: unit CAGR {report['unit_metrics']['cagr']:.2%}",
                flush=True,
            )
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            SyntaxError,
            FloatingPointError,
        ) as exc:
            record.update(
                status="rejected",
                failure={"type": type(exc).__name__, "reason": str(exc)[:1200]},
            )
        self._save()
        collect_research_memory(
            [
                *self.panel.features.values(),
                *[trial["edge"]["holdings"] for trial in self.trials.values()],
            ]
        )
        atomic_write_json(self.output / "memory_usage.json", memory_sample())

    def _known_outcomes(self):
        return np.flatnonzero(unit_return_month_ends(self.panel) <= pd.Timestamp(self.settings["end"]))

    def _pool_health(self, pod):
        distinct = supported_candidates(self.trials, pod, self._known_outcomes(),
            self.policy["minimum_edge_months"],
            uncertainty_haircut=self.policy["edge_uncertainty_haircut"],
            maximum=max(1, len(self.trials)), maximum_correlation=.98)
        agent = [i for i in distinct if self.trials[i]["origin"] in {"agent_written_code", "recorded_agent_code"}]
        target = self.policy.get("target_supported_candidates_per_pod", 5)
        return {"pod": pod, "executable_candidates": sum(t["strategy"].pod == pod for t in self.trials.values()),
            "supported_distinct_candidates": distinct, "supported_count": len(distinct),
            "supported_agent_count": len(agent), "target_supported_count": target,
            "target_met": len(distinct) >= target}

    def _feature_feedback(self, pod):
        rows = []
        reference = (getattr(self, "research_context", {}).get("topic") or {}).get("best_strategy")
        trial = self.trials.get(reference)
        panel = scoped_panel(self.panel, trial["strategy"].universe) if trial is not None else self.panel
        known = candidate_history(trial, self._known_outcomes()) if trial is not None else self._known_outcomes()
        scope = panel.metadata.get("strategy_universe_digest", "initial_broad_diagnostic")
        period = tuple(known.tolist())
        # The host measures all available inputs; pod examples do not fence in
        # the research. Diagnostics never generate a strategy automatically.
        for name in sorted(self.panel.features):
            key = (name, scope, period)
            if key in self.feature_evidence:
                rows.append(self.feature_evidence[key])
                continue
            values, short_values, count = [], [], 0
            for t in known:
                feature, label = panel.features[name][t], panel.returns[t]
                valid = panel.mask[t] & panel.entry_mask[t] & np.isfinite(feature) & np.isfinite(label)
                count += int(valid.sum())
                for mask, result in ((valid, values), (valid & (panel.frames[t].me.to_numpy() >= 500), short_values)):
                    if mask.sum() < 5:
                        continue
                    x, y = (pd.Series(v[mask]).rank().to_numpy() for v in (feature, label))
                    if x.std() and y.std():
                        result.append(float(np.corrcoef(x, y)[0, 1]))
            information = mean_evidence(values)
            row = {"feature": name,
                "cohort": scope,
                "rank_ic": {k: information[k] for k in ("months", "mean", "hac_se", "first_half_mean", "second_half_mean")},
                "shortable_rank_ic": mean_evidence(short_values)["mean"], "stock_months": count,
                "meaning": (self.feature_definitions.get(name, {}).get("meaning") or "")[:160]}
            self.feature_evidence[key] = row
            rows.append(row)
        rows.sort(key=lambda row: abs(row["rank_ic"]["mean"] or 0), reverse=True)
        local = [r for r in rows if r["feature"] in POD_FEATURES[pod]
                 or r["feature"].startswith("filings_")]
        chosen = {}
        for i in range(6):
            for source in (rows, local):
                if i < len(source):
                    chosen[source[i]["feature"]] = source[i]
        return list(chosen.values())

    def _research_feedback(self, pod):
        evaluated, failures, lessons = [], [], []
        context = getattr(self, "research_context", {})
        topic = (context.get("topic") or {}).get("name")
        parent = (context.get("topic") or {}).get("best_strategy")
        for row in self.attempts:
            if row.get("pod", (row.get("strategy") or {}).get("pod")) != pod:
                continue
            metrics = row.get("evidence", {})
            trial = self.trials.get(row.get("strategy_id"))
            if topic and trial is not None and trial["strategy"].research_topic != topic:
                continue
            item = {
                    "id": row.get("strategy_id"),
                    "status": row["status"],
                    "strategy": row.get("strategy"),
                    "cagr": metrics.get("unit_metrics", {}).get("cagr"),
                    "sharpe": metrics.get("unit_metrics", {}).get(
                        "sharpe_annualized"
                    ),
                    "chronological_cagr": metrics.get(
                        "chronological_development_unit_metrics", {}
                    ).get("cagr"),
                    "rank_ic": metrics.get("rank_ic", {}).get("mean"),
                    "shortable_rank_ic": metrics.get(
                        "shortable_rank_ic", {}
                    ).get("mean"),
                    "unit_attribution": metrics.get("unit_attribution"),
                    "universe": metrics.get("universe"),
                    "cohort_coverage": metrics.get("cohort_coverage"),
                    "failure": {
                        k: v
                        for k, v in row.get("failure", {}).items()
                        if k in {"type", "reason", "message"}
                    }
                    if isinstance(row.get("failure"), dict)
                    else (
                        str(row.get("failure"))[:250]
                        if row.get("failure")
                        else None
                    ),
                }
            if trial is not None:
                if "execution_feedback" not in trial:
                    targets = list(trial["schedule"].values())
                    counts = [int(np.count_nonzero(np.abs(value) > 1e-10)) for value in targets]
                    trial["execution_feedback"] = {
                        "native_reviews": len(counts),
                        "mean_target_positions": float(np.mean(counts)) if counts else 0.0,
                        "cash_only_reviews": sum(count == 0 for count in counts),
                    }
                item["execution"] = trial["execution_feedback"]
                item["review"] = review_candidate(trial, self._known_outcomes(),
                    self.policy["minimum_edge_months"], self.policy["edge_uncertainty_haircut"])
                evaluated.append(item)
            elif row["status"] != "evaluated":
                item["duplicate_of"] = row.get("duplicate_of")
                failures.append(item)
                failure = item.get("failure")
                reason = failure.get("reason") if isinstance(failure, dict) else failure
                if reason:
                    try:
                        parsed = json.loads(reason)
                        if isinstance(parsed, list):
                            reason = "; ".join(dict.fromkeys(e.get("msg", str(e)) for e in parsed))
                    except (ValueError, TypeError):
                        pass
                    reason = reason[:350]
                if reason and reason not in lessons:
                    lessons.append(reason)
        recent = evaluated[-2:]
        evaluated.sort(key=lambda row: (row["id"] == parent,
            row["review"]["expected_annual_unit_edge"]
            if row["review"]["expected_annual_unit_edge"] is not None else -1e9), reverse=True)
        return {"best_measured": evaluated[:3], "recent_measured": recent, "recent_failures": failures[-3:],
                "correction_rules": lessons[-6:]}

    def _agent_round(self, number, pod, *, repair=0):
        planning = self.research_context["phase"] == "plan"
        system_prompt = load_prompt("campaign.planner" if planning else "campaign.programmer")
        response_schema = compact_response_schema((TopicPlan if planning else ProposalBatch).model_json_schema())
        feedback = self._research_feedback(pod)
        papers = [
            p
            for p in self.evidence["papers"]
            if p["namespace"] in {"all", "pod_" + pod}
        ]
        packet = {
            "task": load_prompt("campaign.repair_task" if repair else (
                "campaign.plan_task" if planning else "campaign.task")),
            "pod": pod,
            "mandate": pod_spec(pod).mission,
            "development_cutoff": self.settings["end"],
            "features": sorted(self.panel.features),
            "default_feature_coverage": {
                name: round(
                    self.panel.metadata.get("feature_coverage", {}).get(
                        name, 0.0
                    ),
                    3,
                )
                for name in [*POD_FEATURES[pod], *FILING_RESEARCH_FEATURES]
                if name in self.panel.features
            },
            "trusted_forecast_model": load_prompt("campaign.forecast_model"),
            "runtime_contract": load_prompt("campaign.runtime_contract"),
            "improvement_protocol": load_prompt("campaign.improvement_protocol"),
            "research_actions": load_prompt("campaign.research_actions"),
            "topic_protocol": load_prompt("campaign.topic_protocol"),
            "scope_contract": load_prompt("campaign.scope_contract"),
            "research_context": self.research_context,
            "research_priorities": self.policy.get("research_priorities", {}).get(pod, []),
            "topic_catalog": [
                {k: t[k] for k in ("name", "status", "best_score")}
                for t in self.research_loop.pods[pod]["topics"].values()
            ],
            "alternative_data_research": self._alternative_feedback(pod),
            "research_errors": [e for e in self.evidence.get("errors", []) if e.get("pod") == pod][-3:],
            "research_pool": self._pool_health(pod),
            "review_policy": {"minimum_settled_months": self.policy["minimum_edge_months"],
                "uncertainty_haircut": self.policy["edge_uncertainty_haircut"],
                "ownership_maximum_correlation": .85, "pool_duplicate_correlation": .98},
            "feature_diagnostics": self._feature_feedback(pod),
            "feature_definitions": {
                f: definition["meaning"][:160]
                for f, definition in self.feature_definitions.items()
                if f in self.panel.features
            },
            "papers": [
                {k: p[k] for k in ("evidence_id", "title", "version_date")}
                | {"excerpt": p["text"][:1600]}
                for p in papers[-2:]
            ],
            "required_strategy_fields": [
                "pod",
                "name",
                "mechanism",
                "falsification",
                "evidence_refs",
                "code",
                "controls",
                "review",
                "signal_update_months",
                "entry_rank",
                "exit_rank",
                "expected_holding_days",
                "weighting",
                "hedge_policy",
                "construction",
                "universe",
                "evidence_lookback_months",
                "validation_window_months",
                "research_topic",
                "forecast_model",
                "model_features",
                "model_penalty",
                "model_lookback_months",
            ],
            "filing_features": load_prompt("campaign.filing_features"),
            "feedback": feedback,
            "round": number,
            "repair_attempt": repair,
            "requested_candidates": 1 if repair else self.policy.get("candidates_per_request", 3),
        }
        params = GenerationParams.model_validate(
            {
                **self.config["models"]["generation"],
                "max_tokens": self.policy.get("code_max_tokens", 8192),
            }
        )
        # The system prompt and typed schema already carry these contracts.
        # Repeating them consumes provider context without adding evidence.
        for key in ("runtime_contract", "trusted_forecast_model", "required_strategy_fields"):
            packet.pop(key, None)
        if planning:
            for key in ("runtime_contract", "trusted_forecast_model", "required_strategy_fields", "filing_features"):
                packet.pop(key, None)
            for item in [*packet["feedback"]["best_measured"], *packet["feedback"]["recent_measured"]]:
                item["strategy"] = {k: v for k, v in (item["strategy"] or {}).items()
                    if k in {"name", "mechanism", "universe", "evidence_lookback_months"}}
        if not self.config["models"]["roles"]["lead"].get("structured_outputs", False):
            packet["response_schema"] = response_schema
        packet = self._bounded_packet(packet, params.max_tokens,
            response_schema=response_schema, system_prompt=system_prompt)
        request = LLMRequest(
            system_prompt,
            json.dumps(packet, default=str, sort_keys=True, separators=(",", ":")),
            params,
            role="lead",
            response_schema=response_schema,
        )
        cache = self.state / "campaign_model_cache"
        cache.mkdir(exist_ok=True)
        backend_identity = {
            "role": self.config["models"]["roles"]["lead"],
            "request_hash": request.request_hash,
        }
        path = cache / (canonical_digest(backend_identity) + ".json")
        tag = f"round_{number}_{pod}" + (f"_repair_{repair}" if repair else "")
        atomic_write_json(
            self.output / (tag + "_request.json"),
            {
                "system": system_prompt,
                "packet": packet,
                "request_hash": request.request_hash,
            },
        )
        cached = json.loads(path.read_text()) if path.exists() else None
        # Transient provider failures must not become permanent response caches.
        if cached and "text" in cached:
            result = cached
        else:
            if self.providers is None:
                self.providers = create_live_models(self.config["models"])
            before_calls = getattr(self.providers.lead, "calls", None)
            try:
                response = self.providers.lead.complete(request)
                result = {
                    "text": response.text,
                    "request_hash": request.request_hash,
                    "response_hash": response.response_hash,
                    "provider": response.provider,
                    "model": response.model,
                    "received_at": response.received_at.isoformat(),
                    "tokens": response.total_tokens,
                }
            except ProviderError as exc:
                result = {
                    "error": type(exc).__name__,
                    "reason": str(exc)[:600],
                    "diagnostic": getattr(exc, "diagnostic", {}),
                    "request_hash": request.request_hash,
                }
            finally:
                if before_calls is not None:
                    self.calls += self.providers.lead.calls - before_calls
                else:
                    self.calls += 1
            atomic_write_json(path, result)
        atomic_write_json(
            self.output / (tag + "_response.json"), result
        )
        if "text" not in result:
            failed_generation = result.get("diagnostic", {}).get(
                "failed_generation"
            )
            self.attempts.append(
                {
                    "pod": pod,
                    "round": number,
                    "status": "provider_rejected",
                    "failure": {
                        "message": result.get("diagnostic", {}).get(
                            "message", result.get("reason")
                        ),
                        "generation_excerpt": failed_generation[:1800]
                        if failed_generation
                        else None,
                    },
                }
            )
            self._save()
            return {
                "status": "invalid_batch" if failed_generation else "provider_failed",
                **{k: v for k, v in result.items() if k != "diagnostic"},
            }
        try:
            batch, invalid = CodeBatch.validate_partial(result["text"])
            for rejected in invalid:
                self.attempts.append({"pod": pod, "round": number,
                    "status": "rejected_strategy_schema", "strategy": rejected["strategy"],
                    "failure": {"type": "ValidationError", "reason": json.dumps(rejected["errors"])[:1200]}})
            self._save()
            if batch is None:
                return {"status": "all_proposals_rejected", "accepted": 0,
                        "rejected": len(invalid), "provider": result["provider"], "model": result["model"]}
            if len(batch.strategies) + len(invalid) > packet["requested_candidates"]:
                raise ValueError("Candidate batch exceeds requested size")
            context = self.research_context
            if context["phase"] == "plan":
                if batch.research_action != "plan":
                    raise ValueError(load_prompt("campaign.plan_required"))
                added = self.research_loop.add_topics(pod, batch.research_topics)
                if added < 2:
                    raise ValueError(load_prompt("campaign.distinct_topics_required"))
                return {"status": "topics_planned", "research_action": "plan",
                    "topics_added": added, "diagnosis": batch.diagnosis}
            if batch.research_action == "plan":
                raise ValueError(load_prompt("campaign.proposal_after_plan"))
            self.research_loop.add_topics(pod, batch.research_topics)
        except ValueError as exc:
            summary = str(exc)[:250]
            if isinstance(exc, ValidationError):
                summary = json.dumps(
                    exc.errors(
                        include_input=False,
                        include_url=False,
                        include_context=False,
                    )
                )[:1200]
            self.attempts.append(
                {
                    "round": number,
                    "pod": pod,
                    "status": "rejected_batch",
                    "failure": {"type": type(exc).__name__, "reason": summary},
                }
            )
            self._save()
            return {"status": "invalid_batch", "reason": summary}
        unavailable_features = ()
        if batch.research_queries:
            self._search_evidence(pod, batch.research_queries)
        if batch.alternative_searches or batch.alternative_requests:
            self._search_alternative(pod, batch.alternative_searches, batch.alternative_requests)
        if not batch.strategies:
            return {"status": "retained" if batch.research_action == "retain" else "evidence_requested",
                "accepted": 0, "rejected": len(invalid), "research_action": batch.research_action,
                "research_request_signature": canonical_digest({"papers": batch.research_queries,
                    "discover": [x.model_dump(mode="json") for x in batch.alternative_searches],
                    "acquire": [x.model_dump(mode="json") for x in batch.alternative_requests]}),
                "diagnosis": batch.diagnosis, "provider": result["provider"], "model": result["model"]}
        try:
            self._install_text_measurements(
                batch.filing_measurements, tag
            )
        except (ValueError, TypeError, KeyError) as exc:
            self.attempts.append(
                {
                    "pod": pod,
                    "round": number,
                    "status": "rejected_text_definition",
                    "failure": str(exc)[:1200],
                }
            )
            self._save()
            unavailable_features = tuple(
                s.name for s in batch.filing_measurements
            )
        before = set(self.trials)
        for strategy in batch.strategies:
            if strategy.pod != pod:
                self.attempts.append({"pod": pod, "round": number, "status": "rejected_wrong_pod",
                    "strategy": strategy.model_dump(mode="json"),
                    "failure": {"type": "ValueError", "reason": "A pod may propose only its own strategies"}})
                continue
            required = {"universe", "evidence_lookback_months", "validation_window_months", "research_topic"}
            if strategy.forecast_model != "code":
                required.update({"model_features", "model_penalty", "model_lookback_months"})
                if strategy.forecast_model == "xgboost":
                    required.add("model_parameters")
            if (not required <= strategy.model_fields_set or strategy.research_topic != context["topic"]["name"]
                or (context["phase"] in {"refine", "review"} and strategy.parent_strategy_id != context["topic"]["best_strategy"])):
                self.attempts.append({"pod": pod, "round": number,
                    "status": "rejected_missing_scope_or_topic", "strategy": strategy.model_dump(mode="json"),
                    "failure": {"type": "ValueError", "reason": load_prompt("campaign.explicit_scope_required")}})
                continue
            self._evaluate(
                strategy,
                "agent_written_code",
                number,
                unavailable_features=unavailable_features,
                provenance={
                    k: result[k]
                    for k in (
                        "provider",
                        "model",
                        "request_hash",
                        "response_hash",
                        "received_at",
                    )
                    if k in result
                },
            )
        accepted = len(set(self.trials) - before)
        return {
            "status": "evaluated" if accepted else "all_proposals_rejected",
            "accepted": accepted,
            "rejected": len(batch.strategies) + len(invalid) - accepted,
            "diagnosis": batch.diagnosis,
            "research_action": batch.research_action,
            "proposed": len(batch.strategies) + len(invalid),
            "new_strategy_ids": sorted(set(self.trials) - before),
            "provider": result["provider"],
            "model": result["model"],
        }

    def _search_evidence(self, pod, queries):
        """Execute a pod's research requests against its dated paper corpus."""
        application = self.config["application"]
        service = PaperService(PaperStore(config_path(application["papers_db"])),
            clock=now_utc, online=application.get("online_papers", False),
            transport=UrllibTransport() if application.get("online_papers") else None,
            timeout=15)
        cutoff = dt.datetime.combine(dt.date.fromisoformat(self.settings["end"]), dt.time.max, dt.timezone.utc)
        for query in queries:
            try:
                found = service.search(query, as_of=cutoff, limit=4)
                for card in found.cards:
                    if not card.version_date or card.version_date > cutoff:
                        continue
                    if not paper_relevant("pod_" + pod, card.title, card.abstract):
                        continue
                    identity = card.paper_id + ":" + card.version_key
                    if identity in self.evidence_ids:
                        continue
                    self.evidence["papers"].append({"evidence_id": identity,
                        "namespace": "pod_" + pod, "title": card.title,
                        "version_date": card.version_date.isoformat() if card.version_date else None,
                        "source": card.source, "text": (card.abstract or "")[:16000],
                        "text_scope": "abstract", "claims": []})
                    self.evidence_ids.add(identity)
            except Exception as exc:
                self.evidence["errors"].append({"stage": "pod_requested_paper_search",
                    "pod": pod, "query": query, "error": type(exc).__name__, "detail": str(exc)[:300]})
        self._save_evidence()

    def _alternative_feedback(self, pod):
        output = []
        for row in self.evidence.get("alternative_data", []):
            if row.get("namespace") != pod_spec(pod).namespace:
                continue
            compact = {k: row[k] for k in ("status", "discovery_id", "request_id", "formation_eligible", "error_type") if k in row}
            for key, fields in (
                ("entities", ("entity_id", "entity_name", "entity_representation")),
                ("metrics", ("insight_id", "insight_label", "topic_label")),
                ("datasets", ("tearsheet_name",)),
            ):
                values = row.get(key, [])
                if isinstance(values, list):
                    compact[key] = [{f: item[f] for f in fields if f in item}
                                    for item in values[:3] if isinstance(item, dict)]
            output.append(compact)
        return output[-3:]

    def _search_alternative(self, pod, searches, requests):
        """Let pods discover and acquire provider data without inventing vintages."""
        namespace = pod_spec(pod).namespace
        for query in searches:
            try:
                result = self.carbon.discover_stock(namespace, query.model_dump(mode="json"), retry_failed=True, max_attempts=3)
                self.evidence.setdefault("alternative_data", []).append(result)
            except Exception as exc:
                self.evidence["errors"].append({"stage": "pod_requested_alternative_search", "pod": pod,
                    "error": type(exc).__name__, "detail": str(exc)[:300]})
        for request in requests:
            try:
                if request.start > request.end or pd.Timestamp(request.end) > pd.Timestamp(self.settings["end"]):
                    raise ValueError("Alternative-data request crosses the fold training cutoff")
                result = self.carbon.request_stock(namespace, request.model_dump(mode="json"))
                # A current receipt is research material, not historical vintage proof.
                self.evidence.setdefault("alternative_data", []).append({**result, "namespace": namespace,
                    "formation_eligible": False, "historical_numeric_use": "requires_verified_vintage_and_dated_security_binding"})
            except Exception as exc:
                self.evidence["errors"].append({"stage": "pod_requested_alternative_request", "pod": pod,
                    "error": type(exc).__name__, "detail": str(exc)[:300]})
        self._save_evidence()

    def _save_evidence(self):
        atomic_write_json(self.output / "evidence.json", self.evidence)
        experiment = json.loads((self.output / "experiment.json").read_text())
        experiment["evidence_sha256"] = file_sha256(self.output / "evidence.json")
        atomic_write_json(self.output / "experiment.json", experiment)

    def _adopt_recorded_evidence(self, response_path):
        """Keep dated research knowledge when continuing actual agent work."""
        source = response_path.parent / "evidence.json"
        if not source.exists():
            return
        receipt = response_path.parent / "experiment.json"
        if receipt.exists():
            expected = json.loads(receipt.read_text()).get("evidence_sha256")
            if expected and expected != file_sha256(source):
                raise ValueError("Recorded paper evidence hash changed")
        cutoff = pd.Timestamp(self.settings["end"], tz="UTC") + pd.Timedelta(days=1)
        for paper in json.loads(source.read_text()).get("papers", []):
            version = paper.get("version_date")
            if not version or pd.Timestamp(version) >= cutoff or paper["evidence_id"] in self.evidence_ids:
                continue
            self.evidence["papers"].append(paper)
            self.evidence_ids.add(paper["evidence_id"])
        self._save_evidence()

    def _bounded_packet(self, packet, completion_tokens, *, response_schema=None, system_prompt=None):
        """Compact optional context while preserving schema and repair evidence."""
        packet = copy.deepcopy(packet)
        def compact(value):
            if isinstance(value, dict):
                return {k: compact(v) for k, v in value.items() if v is not None}
            if isinstance(value, list):
                return [compact(v) for v in value]
            return round(value, 6) if isinstance(value, float) else value
        packet["feedback"] = compact(packet["feedback"])
        for item in [*packet["feedback"]["best_measured"], *packet["feedback"].get("recent_measured", [])]:
            chronological = item.get("review", {}).get("chronological")
            if chronological:
                windows = chronological["windows"]
                # Keep the initial, weakest and latest rolling windows in the
                # prompt. Full overlapping diagnostics remain in the receipt;
                # they are not additional independent monthly observations.
                chosen = ({w["end_offset"]: w for w in
                    (windows[0], min(windows, key=lambda w: w["mean"]), windows[-1])}
                    if windows else {})
                chronological["windows"] = [{k: window[k] for k in
                    ("start_offset", "end_offset", "months", "mean", "hac_se", "return")}
                    for _, window in sorted(chosen.items())]
        recent_ids = {row.get("id") for row in packet["feedback"].get("recent_measured", [])}
        packet["feedback"]["best_measured"] = [row for row in packet["feedback"]["best_measured"]
            if row.get("id") not in recent_ids]
        if "response_schema" in packet:
            packet["response_schema"] = compact_response_schema(packet["response_schema"])
        definitions = packet.get("feature_definitions", {})
        relevant = set(POD_FEATURES.get(packet.get("pod"), []))
        relevant.update(row["feature"] for row in packet.get("feature_diagnostics", []))
        relevant.update({"at_be", "ni_be", "be_me", "ni_me"})
        feedback = packet["feedback"]
        for item in [*feedback["best_measured"], *feedback.get("recent_measured", []), *feedback["recent_failures"]]:
            strategy = item.get("strategy") or {}
            relevant.update(strategy.get("model_features", []))
            try:
                relevant.update(required_features(strategy.get("code", "")))
            except (ValueError, SyntaxError):
                pass
        # Supply definitions for the current hypotheses and diagnostic leads;
        # the complete registered feature list remains available to every pod.
        packet["feature_definitions"] = {name: meaning for name, meaning in definitions.items() if name in relevant}
        packet["research_pool"].pop("supported_distinct_candidates", None)
        allowance = self.policy.get("max_request_estimated_tokens", 7600)
        system_size = len(system_prompt or load_prompt("campaign.programmer"))
        schema_size = 0 if "response_schema" in packet else len(json.dumps(
            response_schema or CodeBatch.model_json_schema(), separators=(",", ":")))

        def estimate():
            size = len(json.dumps(packet, default=str, separators=(",", ":")))
            return completion_tokens + (system_size + schema_size + size) // 3 + 1

        # Compact numerical evidence first; retain its sign, uncertainty and
        # chronology. The latest rejected source and exact error stay together.
        for diagnostic in packet["feature_diagnostics"]:
            diagnostic["rank_ic"] = {k: (round(v, 5) if isinstance(v, float) else v)
                                      for k, v in diagnostic["rank_ic"].items()}
            if diagnostic["shortable_rank_ic"] is not None:
                diagnostic["shortable_rank_ic"] = round(diagnostic["shortable_rank_ic"], 5)
        for item in packet["feedback"]["best_measured"] + packet["feedback"].get("recent_measured", []):
            item["strategy"] = {k: v for k, v in item["strategy"].items()
                                 if k not in {"mechanism", "falsification", "evidence_refs"}}
        for item in packet["feedback"]["recent_failures"]:
            if item.get("strategy"):
                item["strategy"] = {k: v for k, v in item["strategy"].items()
                                    if k not in {"mechanism", "falsification"}}
        while estimate() > allowance:
            feedback = packet["feedback"]
            if len(feedback["best_measured"]) > 1:
                feedback["best_measured"].pop()
            elif len(feedback.get("recent_measured", [])) > 1:
                feedback["recent_measured"].pop(0)
            elif len(feedback["recent_failures"]) > 1:
                feedback["recent_failures"].pop(0)
            elif len(packet["feature_diagnostics"]) > 4:
                packet["feature_diagnostics"].pop()
            elif packet.get("default_feature_coverage"):
                packet["default_feature_coverage"] = {}
            elif packet.get("required_strategy_fields"):
                packet["required_strategy_fields"] = []  # Already enforced by JSON schema.
            elif packet.get("trusted_forecast_model"):
                packet.pop("trusted_forecast_model")
            elif packet.get("filing_features"):
                packet.pop("filing_features")
            elif any(len(p["excerpt"]) > 300 for p in packet["papers"]):
                for paper in packet["papers"]:
                    paper["excerpt"] = paper["excerpt"][:300]
            elif len(packet.get("alternative_data_research", [])) > 1:
                packet["alternative_data_research"].pop(0)
            elif any(p["excerpt"] for p in packet["papers"]):
                for paper in packet["papers"]:
                    paper["excerpt"] = ""
            else:
                # This is context compaction, not an application spending quota.
                # Preserve the contract and repair evidence; the provider owns
                # its actual context and throughput limits.
                break
        packet["request_estimated_tokens"] = estimate()
        return packet

    def _run_research(self):
        """Plan diverse topics, explore each, then refine measured progress."""
        rounds = []
        active_pods = self.policy.get("research_pods", list(POD_FEATURES))
        if (not isinstance(active_pods, list) or len(set(active_pods)) != len(active_pods)
            or not set(active_pods) <= set(POD_FEATURES)):
            raise ValueError("research_pods must list distinct known pods")
        self.research_loop = TopicResearch(active_pods)
        continuations = self.policy.get("research_continuations", {})
        if not isinstance(continuations, dict) or not set(continuations) <= set(active_pods):
            raise ValueError("research_continuations must name active research pods")
        resume = self.policy.get("research_resume")
        if resume:
            if continuations:
                raise ValueError("Resume a checkpoint or open monthly continuations separately")
            path = config_path(resume)
            previous_config = json.loads((path.parent / "config.json").read_text())
            if previous_config["training_lab"]["end"] != self.settings["end"]:
                raise ValueError("An interrupted research checkpoint must resume at its original cutoff")
            saved = json.loads(path.read_text())["pods"]
            if set(saved) != set(active_pods):
                raise ValueError("Research checkpoint pods differ from the requested pods")
            for pod, state in saved.items():
                for topic in state["topics"].values():
                    ResearchTopic.model_validate({k: topic[k] for k in ("name", "mechanism")})
                    if topic["best_strategy"] is not None and topic["best_strategy"] not in self.trials:
                        raise ValueError("Restore the checkpoint's measured agent programs before resuming research")
                self.research_loop.pods[pod] = copy.deepcopy(state)
                self.research_loop.pods[pod]["resumed_from"] = {
                    "path": str(path), "sha256": file_sha256(path)}
        for pod, continuation in continuations.items():
            prior_report = config_path(continuation["prior_report"])
            prior = json.loads(prior_report.read_text())
            if pd.Timestamp(prior["data"]["development_end"]) > pd.Timestamp(self.settings["end"]):
                raise ValueError("Research continuation crosses the development cutoff")
            progress = json.loads((prior_report.parent / "research_progress.json").read_text())
            if not progress["pods"][pod]["planned"]:
                raise ValueError("A continuation requires a completed initial topic plan")
            receipt = json.loads(config_path(continuation["review_response"]).read_text())
            if canonical_digest(receipt["text"]) != continuation["review_text_digest"]:
                raise ValueError("Research continuation response changed")
            text = receipt["text"].strip()
            if text.startswith(("```json\n", "```\n")) and text.endswith("\n```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            decisions = [d for d in json.loads(text)["decisions"] if d["pod"] == pod]
            if len(decisions) != 1 or decisions[0]["action"] != "research":
                raise ValueError("A continuation requires this pod's recorded research decision")
            topics = [ResearchTopic.model_validate(t) for t in decisions[0].get("topics", [])]
            if not topics:
                raise ValueError("Research continuation needs an agent-proposed topic")
            self.research_loop.add_topics(pod, topics)
            for topic in topics:
                previous = progress["pods"][pod]["topics"].get(topic.name)
                if previous and previous.get("best_strategy") in self.trials:
                    resumed = copy.deepcopy(previous)
                    resumed.update(status="review", stop_reason=None)
                    self.research_loop.pods[pod]["topics"][topic.name] = resumed
            self.research_loop.pods[pod]["planned"] = True
            self.research_loop.pods[pod]["continuation"] = {
                "prior_report": str(prior_report),
                "review_response": str(config_path(continuation["review_response"])),
                "review_text_digest": continuation["review_text_digest"],
                "provider": receipt["provider"], "model": receipt["model"],
            }
        repairs = min(3, max(0, int(self.config["models"].get("repair_attempts", 2))))
        number = 0
        while any(self.research_loop.current(p) is not None for p in active_pods):
            number += 1
            for pod in active_pods:
                context = self.research_loop.current(pod)
                if context is None:
                    continue
                self.research_context = copy.deepcopy(context)
                before = self._pool_health(pod)
                trial_ids = set(self.trials)
                evidence_before = self._evidence_signature()
                for repair in range(repairs + 1):
                    print(f"Agent research round {number}, pod {pod}, repair {repair}", flush=True)
                    result = self._agent_round(number, pod, repair=repair)
                    after = self._pool_health(pod)
                    rounds.append({"round": number, "pod": pod, "repair": repair,
                                   **result, "pool_before": before, "pool_after": after})
                    atomic_write_json(self.output / "rounds.json", rounds)
                    atomic_write_json(self.output / "research_pool.json",
                                      {p: self._pool_health(p) for p in POD_FEATURES})
                    if result.get("status") in {"provider_failed", "evidence_requested", "retained", "topics_planned"}:
                        break
                    if result.get("status") == "evaluated" and not result.get("rejected"):
                        break
                    before = after
                measured = [(identity, review_candidate(self.trials[identity], self._known_outcomes(),
                    self.policy["minimum_edge_months"], self.policy["edge_uncertainty_haircut"]))
                    for identity in sorted(set(self.trials) - trial_ids)]
                if context["phase"] in {"refine", "review"} and context["topic"]["best_strategy"] is not None:
                    parent = self.trials[context["topic"]["best_strategy"]]
                    for identity, review in measured:
                        review["parent_comparison"] = compare_refinement(self.trials[identity], parent,
                            self._known_outcomes(), self.policy["minimum_edge_months"], self.policy["edge_uncertainty_haircut"])
                evidence_after = self._evidence_signature()
                signature = evidence_after if evidence_after != evidence_before else "no_new_evidence"
                self.research_loop.observe(pod, context, result, measured, signature)
                rounds[-1]["topic_result"] = copy.deepcopy(self.research_loop.pods[pod])
                atomic_write_json(self.output / "rounds.json", rounds)
                atomic_write_json(self.output / "research_progress.json", self.research_loop.snapshot())
        return rounds, {p: self._pool_health(p) for p in POD_FEATURES}

    def _evidence_signature(self):
        """Retrieval attempts and errors are not new research information."""
        papers = {p["evidence_id"]: p.get("text", "") for p in self.evidence.get("papers", [])}
        alternatives = {}
        for row in self.evidence.get("alternative_data", []):
            if row.get("error_type") or any(x in row.get("status", "").lower() for x in ("fail", "error", "reject")):
                continue
            content = {k: row[k] for k in ("entities", "metrics", "datasets", "data", "formation_eligible") if k in row}
            if content:
                alternatives[canonical_digest(content)] = content
        return canonical_digest({"papers": papers, "alternative_data": alternatives})

    def _install_text_measurements(self, measurements, label):
        if measurements:
            if not self.settings.get("filings_research_enabled"):
                return {"status": "text_source_disabled"}
            definitions = [s.model_dump(mode="json") for s in measurements]
            for spec in measurements:
                if (
                    spec.name in self.panel.features
                    and self.custom_feature_specs.get(spec.name)
                    != canonical_digest(spec.model_dump(mode="json"))
                ):
                    raise ValueError(
                        "Revised text rules require a new feature name"
                    )
            registry = self.state / (
                "custom_text_"
                + canonical_digest(
                    {
                        "file": file_sha256(self.settings["filings_path"]),
                        "specs": definitions,
                    }
                )[:24]
                + ".parquet"
            )
            store = FilingResearchStore(
                self.settings["filings_path"],
                training_end=self.settings["end"],
                start=self.settings["start"],
            )
            measured = custom_measurements(store, measurements, registry)
            if self.decision_filings is not None:
                self.decision_filings.register_custom(measured, measurements)
            for name, values in custom_feature_arrays(
                measured,
                self.panel.dates.tz_localize("America/New_York"),
                self.panel.securities,
                measurements,
            ).items():
                if name in self.panel.features:
                    definition = next(
                        spec.model_dump(mode="json")
                        for spec in measurements
                        if spec.name == name
                    )
                    if self.custom_feature_specs.get(name) == canonical_digest(
                        definition
                    ):
                        continue
                    raise ValueError(
                        "Revised text rules require a new feature name"
                    )
                self.panel.metadata.setdefault("feature_coverage", {})[name] = (
                    float(
                        np.count_nonzero(np.isfinite(values) & self.panel.mask)
                        / max(1, self.panel.mask.sum())
                    )
                )
                self.panel.features[name] = (
                    pd.DataFrame(values)
                    .where(self.panel.mask)
                    .rank(axis=1, pct=True)
                    .to_numpy()
                    * 2
                    - 1
                ).astype(np.float32)
            for spec in measurements:
                self.custom_feature_specs[spec.name] = canonical_digest(
                    spec.model_dump(mode="json")
                )
            self.panel.metadata["custom_text_measurements"] = dict(
                self.custom_feature_specs
            )
            self.panel.metadata["available_numeric_research_features"] = sorted(
                self.panel.features
            )
            atomic_write_json(
                self.output / (label + "_text_measurements.json"),
                {
                    "definitions": definitions,
                    "data_sha256": file_sha256(registry),
                    "cloud_text_transmission": False,
                },
            )

    def _first_session_reviews(self):
        if not self.evaluation_period:
            return {}
        eligible = (self.market.dates >= pd.Timestamp(self.evaluation_period["start"])) & (
            self.market.dates <= pd.Timestamp(self.evaluation_period["end"]))
        days = np.flatnonzero(eligible)
        first = ~self.market.dates[days].to_period("M").duplicated()
        return {int(d): int(self.market.formation[d]) for d in days[first]}

    def _construct(self):
        decisions, allocation_log = {}, []
        book = SleeveExecutionBook(
            len(self.panel.securities),
            maximum_positions=self.settings["universe_size"],
        )
        min_months = self.policy["minimum_edge_months"]
        review_days = {
            int(self.market.dates.searchsorted(d)): t
            for t, d in enumerate(self.panel.execution_dates)
        }
        if self.evaluation_period:
            review_days = self._first_session_reviews()
        # Union of native strategy reviews and monthly CIO reviews; the latest
        # formation date is the only context for limits and settled outcomes.
        native_days = {
            d for trial in self.trials.values() for d in trial["schedule"]
        }
        days = sorted(native_days | set(review_days))
        if self.evaluation_period:
            days = [d for d in days if pd.Timestamp(self.evaluation_period["start"]) <= self.market.dates[d] <= pd.Timestamp(self.evaluation_period["end"])]
        for day in days:
            formation = review_days.get(
                day, int(self.market.formation[day])
            )
            if formation < 0:
                continue
            known = np.flatnonzero(
                unit_return_month_ends(self.panel)
                <= self.panel.dates[formation]
            )
            if self.portfolio_evidence_scope == "retrospective_evaluation_period":
                outcomes = unit_return_month_ends(self.panel)
                known = np.flatnonzero((outcomes >= pd.Timestamp(self.evaluation_period["start"]))
                    & (outcomes <= pd.Timestamp(self.evaluation_period["end"])))
            ownership = {
                p: select_roster(self.trials, p, known, min_months,
                    uncertainty_haircut=self.policy["edge_uncertainty_haircut"])
                for p in POD_FEATURES
            }

            def decide(
                nav,
                carried,
                *,
                day=day,
                formation=formation,
                known=known.copy(),
                ownership=ownership,
                cio_review=day in review_days,
            ):
                if not cio_review:
                    changes = {
                        identity: amount
                        * self.trials[identity]["schedule"][day]
                        for identity, amount in book.capital.items()
                        if amount > 0
                        and day in self.trials[identity]["schedule"]
                    }
                    if not changes:
                        return None
                    return book.set_targets(day, changes)
                identities, targets, edges, caps, pods = [], [], [], [], []
                for pod, owned in ownership.items():
                    for identity in owned:
                        trial = self.trials[identity]
                        eligible = candidate_history(trial, known)
                        history = trial["monthly"][eligible]
                        if len(history) < min_months:
                            continue
                        # Funding shrinks the unit evidence and pays its
                        # uncertainty before any capital is allocated.
                        review = review_candidate(trial, known, min_months,
                            self.policy["edge_uncertainty_haircut"])
                        if not review["supported"]:
                            continue
                        expected = review["expected_annual_unit_edge"]
                        if day in trial["schedule"]:
                            target = trial["schedule"][day].copy()
                        elif day in trial["edge"].get("prior_holdings", {}):
                            target = trial["edge"]["prior_holdings"][day].copy()
                        else:
                            # Legacy compatible measurement caches may retain
                            # their complete daily holdings matrix.
                            target = trial["edge"]["holdings"][day - 1].copy()
                        bound = nav * self.settings["max_pod_capital"]
                        identities.append(identity)
                        targets.append(target)
                        edges.append(expected)
                        caps.append(bound)
                        pods.append(pod)
                if identities:
                    common_known = known.copy()
                    for identity in identities:
                        common_known = np.intersect1d(common_known,
                            candidate_history(self.trials[identity], known))
                    histories = np.array(
                        [
                            self.trials[identity]["monthly"][common_known]
                            for identity in identities
                        ]
                    )
                    covariance = np.atleast_2d(np.cov(histories, ddof=1)) * 12
                    if len(identities) == 1:
                        covariance = np.array(
                            [[float(np.var(histories[0], ddof=1)) * 12]]
                        )
                    dollars, joint_economics = allocate_jointly(
                        edges,
                        caps,
                        pods,
                        targets,
                        carried * nav,
                        self.panel.frames[formation],
                        self.settings,
                        nav,
                        covariance,
                        risk_aversion=self.policy.get(
                            "allocation_risk_aversion", 2.0
                        ),
                    )
                else:
                    dollars, joint_economics = (
                        np.array([]),
                        {
                            "status": "no_positive_unit_edge",
                            "expected_dollars": 0.0,
                            "competition_hurdle_dollars": 0.04 * nav,
                            "expected_surplus_over_TB_plus_four_dollars": -0.04
                            * nav,
                        },
                    )
                changes = {
                    identity: amount * weights
                    for identity, amount, weights in zip(
                        identities, dollars, targets
                    )
                }
                for old in set(book.targets) - set(changes):
                    changes[old] = np.zeros(len(self.panel.securities))
                instruction = book.set_targets(
                    day,
                    changes,
                    capital={
                        identity: float(amount)
                        for identity, amount in zip(identities, dollars)
                    },
                )
                allocation_log.append(
                    {
                        "decision_asof": str(
                            self.panel.dates[formation].date()
                        ),
                        "execution_date": str(self.market.dates[day].date()),
                        "nav_usd": nav,
                        "portfolio_evidence_scope": self.portfolio_evidence_scope,
                        "owned": ownership,
                        "funded": [
                            {
                                "strategy_id": i,
                                "capital_usd": float(a),
                                "expected_unit_edge": e,
                                "capital_limit_usd": c,
                            }
                            for i, a, e, c in zip(
                                identities, dollars, edges, caps
                            )
                        ],
                        "unallocated_usd": float(nav - dollars.sum()),
                        "combined_order_economics": joint_economics,
                    }
                )
                return instruction

            decisions[day] = decide
        fund = replay_daily(
            self.market,
            decisions,
            {
                **self.settings,
                "execution_start_nav": self.settings["fund_capital_usd"],
                "execution_observer": book,
                "enforce_fund_mandate": True,
                "holdings_storage_path": str(self.output / "fund_holdings.npy"),
            },
            keep_holdings=True,
        )
        # The recorded static instructions must reproduce the adaptive book.
        verified = replay_daily(
            self.market,
            {
                day: target
                for day, target in fund["execution_targets"].items()
            },
            {
                **self.settings,
                "execution_start_nav": self.settings["fund_capital_usd"],
                "enforce_fund_mandate": True,
            },
        )
        difference = float(np.max(np.abs(verified["net"] - fund["net"])))
        if difference > 1e-10:
            raise ValueError("Recorded target replay does not reconcile")
        fund["execution"]["sleeve_mark_reconciliation_max_usd"] = (
            book.reconciliation_max_usd
        )
        fund["execution"]["sleeve_gross_pnl_usd"] = book.pnl
        fund["execution"]["fund_compression_events"] = book.compression_events
        return fund, allocation_log, difference

    def run(self):
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        fingerprint = {
            "source_sha256": source_digest(),
            "settings": self.settings,
            "policy": self.policy,
            "panel_sha256": file_sha256(self.settings["panel_path"]),
            "daily_sha256": file_sha256(self.settings["daily_path"]),
            "benchmark_reporting_sha256": file_sha256(
                config_path(self.policy["benchmark_fred_path"])
            ),
            "model_configuration": self.config["models"],
            "recorded_code_proposal_hashes": {
                str(config_path(path)): file_sha256(config_path(path))
                for path in self.policy.get("recorded_code_proposals", [])
            },
            "evidence_configuration": {
                k: self.config["application"].get(k)
                for k in (
                    "paper_queries",
                    "paper_ids",
                    "papers_per_query",
                    "online_papers",
                    "papers_db",
                )
            },
            "feature_documentation_sha256": file_sha256(
                config_path(self.policy["feature_documentation_path"])
            )
            if self.policy.get("feature_documentation_path")
            and config_path(self.policy["feature_documentation_path"]).exists()
            else None,
            **additional_input_hashes(self.settings),
        }
        self.output = self.state / (
            "campaign_"
            + canonical_digest(fingerprint)[:16]
            + "_"
            + uuid.uuid4().hex[:8]
        )
        self.output.mkdir(exist_ok=True)
        self._save()
        atomic_write_json(self.output / "experiment.json", fingerprint)
        atomic_write_json(self.output / "config.json", self.config)
        root = Path(__file__).parents[1]
        for source in root.rglob("*.py"):
            target = (
                self.output
                / "source_snapshot"
                / "pod_platform"
                / source.relative_to(root)
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copy2(PROMPTS_PATH, self.output / "source_snapshot" / "prompts.txt")
        self.evidence, carbon = discover_evidence(
            self.config, self.state, **self.evidence_services
        )
        self.carbon = carbon
        atomic_write_json(self.output / "evidence.json", self.evidence)
        fingerprint["evidence_sha256"] = file_sha256(
            self.output / "evidence.json"
        )
        atomic_write_json(self.output / "experiment.json", fingerprint)
        self.evidence_ids = {p["evidence_id"] for p in self.evidence["papers"]}
        self.panel = prepare_panel(
            Path(self.settings["panel_path"]),
            Path(self.settings["daily_path"]),
            self.settings,
            self.state
            / (fingerprint["daily_sha256"][:16] + "_campaign_daily.parquet"),
        )
        if self.settings.get("filings_research_enabled"):
            store = FilingResearchStore(
                self.settings["filings_path"],
                training_end=self.settings["end"],
                start=self.settings["start"],
            )
            text_cache = self.state / (
                fingerprint["filings_sha256"][:16] + "_" + VERSION + ".parquet"
            )
            measured = text_measurements(store, text_cache)
            self.decision_filings = DecisionFilingFeatures(store.metadata(), self.panel.securities, self.settings["end"])
            self.decision_filings.register_text(measured)
            for name, values in text_feature_arrays(
                measured,
                self.panel.dates.tz_localize("America/New_York"),
                self.panel.securities,
            ).items():
                self.panel.features[name] = (
                    pd.DataFrame(values)
                    .where(self.panel.mask)
                    .rank(axis=1, pct=True)
                    .to_numpy()
                    * 2
                    - 1
                ).astype(np.float32)
            self.panel.metadata["text_measurements"] = {
                "version": VERSION,
                "features": list(TEXT_FEATURES),
                "documents": len(measured),
                "sha256": file_sha256(text_cache),
                "cloud_text_transmission": False,
                "native_review_refresh": "Current filing vectors refresh from publication-bounded indices; monthly characteristic history remains month-end",
            }
        self.panel.metadata["alternative_features"] = (
            attach_alternative_features(
                self.panel, self.policy.get("alternative_bindings", []), carbon
            )
        )
        self.panel.metadata["available_numeric_research_features"] = sorted(
            self.panel.features
        )
        if self.feature_definitions:
            atomic_write_json(
                self.output / "characteristic_definitions.json",
                self.feature_definitions,
            )
        write_access_audit(self.config, self.output, self.panel.metadata)
        self.market = daily_market(self.panel, self.settings)
        self.panel.daily = None
        collect_research_memory(self.panel.features.values())
        active_dates = self.market.dates[
            self.market.dates >= self.panel.execution_dates[0]
        ]
        self.reporting_benchmark = load_reporting_benchmarks(
            config_path(self.policy["benchmark_fred_path"]), active_dates
        )
        self.market.cash_returns = (
            self.reporting_benchmark.tb3ms.reindex(self.market.dates)
            .fillna(0)
            .to_numpy()
            / 25200
        )
        self.panel.metadata["feature_coverage"] = {
            name: float(
                np.count_nonzero(np.isfinite(values) & self.panel.mask)
                / max(1, self.panel.mask.sum())
            )
            for name, values in self.panel.features.items()
        }
        for name, values in list(self.panel.features.items()):
            self.panel.features[name] = spill_array(
                self.output / "feature_arrays" / (name + ".npy"), values
            )
        self.panel.daily = None
        collect_research_memory(self.panel.features.values())
        for proposal_path in self.policy.get("recorded_code_proposals", []):
            path = config_path(proposal_path)
            self._adopt_recorded_evidence(path)
            recorded = json.loads(path.read_text())
            if not all(
                recorded.get(k)
                for k in (
                    "text",
                    "provider",
                    "model",
                    "request_hash",
                    "response_hash",
                )
            ):
                raise ValueError(
                    "Recorded code requires an actual model response and provenance"
                )
            try:
                batch, invalid = CodeBatch.validate_partial(recorded["text"])
                for rejected in invalid:
                    self.attempts.append({"round": 0, "status": "rejected_recorded_strategy_schema",
                        "strategy": rejected["strategy"],
                        "failure": {"type": "ValidationError", "reason": json.dumps(rejected["errors"])[:1200]},
                        "source_response_sha256": file_sha256(path)})
                self._save()
                if batch is None:
                    continue
            except ValueError as exc:
                self.attempts.append(
                    {
                        "status": "rejected_recorded_schema",
                        "source_response_sha256": file_sha256(path),
                        "failure_type": type(exc).__name__,
                    }
                )
                self._save()
                continue
            unavailable_features = ()
            try:
                self._install_text_measurements(
                    batch.filing_measurements,
                    "recorded_" + file_sha256(path)[:12],
                )
            except (ValueError, TypeError, KeyError) as exc:
                self.attempts.append(
                    {
                        "status": "rejected_recorded_text_definition",
                        "source_response_sha256": file_sha256(path),
                        "failure": str(exc)[:1200],
                    }
                )
                self._save()
                unavailable_features = tuple(
                    s.name for s in batch.filing_measurements
                )
            for strategy in batch.strategies:
                allowed = self.policy.get("recorded_strategy_ids")
                identity = strategy.pod + ":" + strategy.name + ":" + canonical_digest(strategy.model_dump(mode="json"))[:10]
                if allowed is not None and identity not in allowed:
                    continue
                self._evaluate(
                    strategy,
                    "recorded_agent_code",
                    0,
                    unavailable_features=unavailable_features,
                    provenance={
                        k: recorded[k]
                        for k in (
                            "provider",
                            "model",
                            "request_hash",
                            "response_hash",
                            "received_at",
                        )
                        if k in recorded
                    }
                    | {"source_response_sha256": file_sha256(path)},
                )
        rounds, research_pool = self._run_research()
        known = np.flatnonzero(
            unit_return_month_ends(self.panel)
            <= pd.Timestamp(self.settings["end"])
        )
        owned = {
            p: select_roster(
                self.trials, p, known, self.policy["minimum_edge_months"],
                uncertainty_haircut=self.policy["edge_uncertainty_haircut"]
            )
            for p in POD_FEATURES
        }
        atomic_write_json(
            self.output / "pod_books.json",
            {
                p: [
                    {
                        "strategy_id": i,
                        "strategy": self.trials[i]["strategy"].model_dump(
                            mode="json"
                        ),
                        "evidence": self.trials[i]["report"],
                    }
                    for i in ids
                ]
                for p, ids in owned.items()
            },
        )
        fund, allocations, replay_difference = self._construct()
        active = self.market.dates >= self.panel.execution_dates[0]
        if self.evaluation_period:
            active = (self.market.dates >= pd.Timestamp(self.evaluation_period["start"])) & (self.market.dates <= pd.Timestamp(self.evaluation_period["end"]))
        dates = self.market.dates[active]
        benchmark = self.reporting_benchmark.reindex(dates)
        metrics, daily, monthly = competition_report(
            dates,
            fund["net"][active],
            benchmark,
            turnover=fund["turnover"][active],
            returns_include_cash=True,
        )
        for name, values in fund["attribution"].items():
            daily[name] = values[active]
        components = daily.long_return + daily.short_return + daily.cash_return
        if not np.allclose(components, daily.net, atol=1e-12):
            raise ValueError("Long/short/cash attribution failed to reconcile")
        daily.to_csv(self.output / "daily_ledger.csv", index_label="date")
        monthly.to_csv(self.output / "monthly_returns.csv", index_label="date")
        atomic_write_json(self.output / "capital_allocations.json", allocations)
        held = fund["holdings"][active]
        counts = (np.abs(held) > 1e-10).sum(axis=1)
        constraint_report = {
            "sessions": len(dates),
            "minimum_positions_observed": int(counts.min()),
            "maximum_positions_observed": int(counts.max()),
            "minimum_position_failures": int(
                (counts < self.settings["min_positions"]).sum()
            ),
            "maximum_position_failures": int(
                (counts > self.settings["universe_size"]).sum()
            ),
            "maximum_gross": float(np.abs(held).sum(axis=1).max()),
            "maximum_absolute_net": float(np.abs(held.sum(axis=1)).max()),
            "maximum_absolute_name": float(np.abs(held).max()),
        }
        constraints_pass = (
            constraint_report["minimum_position_failures"] == 0
            and constraint_report["maximum_position_failures"] == 0
            and constraint_report["maximum_gross"] <= 2 + 1e-5
            and constraint_report["maximum_absolute_net"] <= 0.5 + 1e-5
            and constraint_report["maximum_absolute_name"] <= 0.05 + 1e-5
        )
        targets = []
        for day, target in fund["execution_targets"].items():
            for column in np.flatnonzero(abs(target) > 1e-10):
                targets.append(
                    (
                        self.market.dates[day],
                        int(self.panel.securities[column]),
                        float(target[column]),
                    )
                )
        pd.DataFrame(
            targets, columns=["execution_date", "permno", "target_weight"]
        ).to_parquet(self.output / "execution_targets.parquet", index=False)
        atomic_write_json(
            self.output / "execution_dates.json",
            [
                str(self.market.dates[d].date())
                for d in fund["execution_targets"]
            ],
        )
        row, column = np.nonzero(np.abs(held) > 1e-10)
        pd.DataFrame(
            {
                "date": dates[row],
                "permno": np.asarray(self.panel.securities)[column],
                "weight": held[row, column],
            }
        ).to_parquet(self.output / "actual_holdings.parquet", index=False)
        submission = write_bundle(
            self.output, dates, held, self.market, np.flatnonzero(active),
            self.panel.securities, daily, monthly, fund['turnover'][active],
            fund['execution_targets'], self.settings, self.policy, fund['execution'],
            opening_weights=(fund['holdings'][np.flatnonzero(active)[0]-1]
                             if np.flatnonzero(active)[0] > 0 else None),
            opening_nav=float(self.settings['fund_capital_usd'] * np.prod(
                1 + fund['net'][:np.flatnonzero(active)[0]])),
        )
        constraint_report['part2'] = submission['sector_compliance']
        constraint_report['volatility'] = submission['volatility']
        constraints_pass = bool(constraints_pass and submission['constraints_pass'])
        agent_counts = {
            p: sum(
                t["origin"] in {"agent_written_code", "recorded_agent_code"}
                and t["strategy"].pod == p
                for t in self.trials.values()
            )
            for p in POD_FEATURES
        }
        agent_stage_complete = all(n > 0 for n in agent_counts.values()) and not any(
            p["blocked"] for p in self.research_loop.pods.values())
        pool_complete = all(row["target_met"] for row in research_pool.values())
        report = {
            "status": "recorded_campaign_replayed" if (
                self.policy.get("research_pods") == [] and self.policy.get("recorded_code_proposals"))
            else "integrated_campaign_completed"
            if agent_stage_complete and pool_complete
            else ("research_completed_agent_stage_degraded" if not agent_stage_complete
                  else "research_completed_pool_incomplete"),
            "agent_coding_stage_passed": agent_stage_complete,
            "research_pool_target_met": pool_complete,
            "research_pool": research_pool,
            "accepted_agent_strategies_by_pod": agent_counts,
            "output_dir": str(self.output),
            "data": self.panel.metadata,
            "new_model_call_attempts": self.calls,
            "rounds": rounds,
            "research_progress": self.research_loop.snapshot(),
            "attempts": len(self.attempts),
            "successful_strategies": len(self.trials),
            "owned_strategy_counts": {p: len(v) for p, v in owned.items()},
            "edge_selection_uses_capital": False,
            "capital_grid_used": False,
            "evaluation_period": self.evaluation_period,
            "portfolio_evidence_scope": self.portfolio_evidence_scope,
            "evaluation_is_in_sample": self.portfolio_evidence_scope == "retrospective_evaluation_period",
            "fund": metrics,
            "execution": fund["execution"],
            "constraints": constraint_report,
            "constraints_pass": constraints_pass,
            "submission_ready": submission['submission_ready'],
            "submission": submission,
            "frozen_training_provenance": self.policy.get('frozen_training_provenance'),
            "static_target_replay_max_return_difference": replay_difference,
            "production_admission": False,
            "research_scope": f"Development observations through {self.settings['end']}; later trading outcomes excluded",
            "hypothesis_origin": "Agent-proposed code only; no seeded strategies or default trading fallback",
            "source_limits": [
                "Supplied filing corpus has survivorship and historical identity limitations",
                "Orders fill in full at the observed price; no execution model is applied",
                "Current pretrained models may contain historical knowledge; prompting does not remove this risk",
                "Adaptive training feedback means chronological diagnostics are not untouched validation",
            ],
        }
        self.panel.metadata["campaign_run_health"] = {
            "agent_coding_stage_passed": agent_stage_complete,
            "research_pool_target_met": pool_complete,
            "accepted_agent_strategies_by_pod": agent_counts,
            "paper_count": len(self.evidence["papers"]),
            "alternative_receipts": [
                {
                    "status": row.get("status"),
                    "error_type": row.get("error_type"),
                }
                for row in self.evidence["alternative_data"]
            ],
            "evidence_errors": self.evidence["errors"],
            "round_statuses": [
                {
                    "pod": row["pod"],
                    "round": row["round"],
                    "status": row["status"],
                }
                for row in rounds
            ],
            "constraints_pass": constraints_pass,
        }
        write_access_audit(self.config, self.output, self.panel.metadata)
        atomic_write_json(self.output / "campaign_report.json", report)
        atomic_write_json(
            self.output / "frozen_research_book.json",
            {
                "cutoff": self.settings["end"],
                "pod_books": owned,
                "strategies": {
                    i: self.trials[i]["strategy"].model_dump(mode="json")
                    for ids in owned.values()
                    for i in ids
                },
                "allocation_policy": self.policy,
                "production_admission": False,
            },
        )
        atomic_write_json(
            self.state / "latest_campaign.json",
            {"report": str(self.output / "campaign_report.json")},
        )
        return report


def run_research_campaign(config):
    with (
        research_job_lock(config_path(config["application"]["state_dir"])),
        memory_watchdog(
            max_rss_mib=config["research_campaign"].get("max_rss_mib", 1600)
        ),
    ):
        return ResearchCampaign(config).run()
