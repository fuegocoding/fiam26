"""Four-charter research campaign on an operator-approved dataset (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 4, 13, 14).

Order of operations (resumable and deterministic):

1. load the operator manifest and gate the source card (no data row is read);
2. write immutable ``HypothesisSpec`` and ``ExperimentSpec`` records to ``state_dir/registry.sqlite``;
3. only then run every pre-declared variant through the registered built-in executors (design partition only);
4. derive status from pre-declared gates through the existing ``PromotionOutcome`` logic. This module NEVER issues
   an admission certificate or forecast publication: design-partition evidence cannot establish implementation,
   so the outcome is ``shadow`` (belief gates passed, pending Research Integrity's check window), ``rejected``,
   or ``fixture_demo`` for synthetic data.

Identifiers and hashes depend only on the manifest, dataset, research_asof and campaign config, so a rerun
reproduces the same report and spends no additional variant budget.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Literal, Optional, Sequence, Union

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.errors import PodPlatformError
from ..contracts.experiments import (
    ExperimentAdmission,
    ExperimentBudget,
    ExperimentClaim,
    ExperimentComparisons,
    ExperimentCosts,
    ExperimentData,
    ExperimentIdentity,
    ExperimentModel,
    ExperimentPortfolio,
    ExperimentSample,
    ExperimentSpec,
    ExperimentStatistics,
    ExperimentTime,
)
from ..contracts.hypotheses import HypothesisSpec
from ..contracts.hypotheses import ResearchMode as HypothesisMode
from ..pods.charters import pod_spec
from ..pods.runner import ExecutorError, ExecutorRegistry, ExperimentRequest
from ..pods.schema import (
    ExperimentPlan,
    ExperimentVariant,
    FeatureDefinition,
    ResearchProposal,
)
from ..research.builtins import EXECUTOR_IDS, RECIPES, BuiltinContext
from ..research.builtins.engine import EVALUATION_LABEL
from ..research.builtins.loader import file_sha256
from ..research.builtins.manifest import ApprovedDataset
from ..research.builtins.timeutil import end_of_day
from ..research.promotion import PromotionCriteria, PromotionOutcome
from ..research.registry import Registry
from .base import Charter

REPORT_SCHEMA = "charter_campaign_report.v1"
CONFIG_SCHEMA = "charter_campaign_config.v1"
RESEARCH_MODE = HypothesisMode.CONTEMPORARY_RESEARCH.value
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "campaigns"
    / "charter_campaign_v1.json"
)
_IDENT = r"^[a-z][a-z0-9_]{0,63}$"
Scalar = Union[str, int, float, bool]


class CampaignError(ValueError):
    pass


class CampaignConflict(CampaignError):
    """An immutable record with this id already exists with different content."""


class VariantDef(StrictModel):
    name: str = Field(pattern=_IDENT)
    params: dict[str, Scalar] = Field(default_factory=dict)


class HypothesisDef(StrictModel):
    key: str = Field(pattern=_IDENT)
    charter: str
    recipe_id: str = Field(pattern=_IDENT)
    title: str = Field(min_length=5, max_length=200)
    question: str = Field(min_length=10)
    mechanism: str = Field(min_length=20, max_length=1500)
    falsification: list[str] = Field(min_length=1)
    variants: list[VariantDef] = Field(min_length=1, max_length=8)
    primary_variant: str

    @model_validator(mode="after")
    def _check(self) -> "HypothesisDef":
        Charter(self.charter)
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("variant names must be unique")
        if self.primary_variant not in names:
            raise ValueError(
                "primary_variant must name a declared variant (pre-declared, never chosen after the fact)"
            )
        recipe = RECIPES.get(self.recipe_id)
        if recipe is None or recipe.charter != self.charter:
            raise ValueError(
                f"recipe {self.recipe_id!r} is not a reviewed recipe of charter {self.charter}"
            )
        for variant in self.variants:
            if "dataset_id" in variant.params:
                raise ValueError(
                    "variants must not name datasets; the campaign supplies the dataset id"
                )
        return self


class Thresholds(StrictModel):
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    min_belief_ic_improvement: float = 0.0
    min_bridge_ic_candidate: float = 0.0
    require_halves_same_sign: bool = True
    min_positive_year_share: float = Field(default=0.5, ge=0.0, le=1.0)


class CampaignConfig(StrictModel):
    schema_version: Literal["charter_campaign_config.v1"] = CONFIG_SCHEMA
    config_id: str = Field(min_length=1)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    hypotheses: list[HypothesisDef] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "CampaignConfig":
        keys = [h.key for h in self.hypotheses]
        if len(set(keys)) != len(keys):
            raise ValueError("hypothesis keys must be unique")
        return self

    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


def load_campaign_config(
    source: "CampaignConfig | dict | str | Path | None" = None,
) -> CampaignConfig:
    if isinstance(source, CampaignConfig):
        return source
    if source is None:
        source = DEFAULT_CONFIG_PATH
    if isinstance(source, dict):
        raw = source
    else:
        try:
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(
                f"campaign config is not readable JSON: {exc}"
            ) from None
    try:
        return CampaignConfig.model_validate(raw)
    except ValueError as exc:
        raise CampaignError(f"invalid campaign config: {exc}") from None


def default_campaign_config() -> CampaignConfig:
    return load_campaign_config(None)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _select(
    config: CampaignConfig, charters: Optional[Sequence[str]]
) -> list[HypothesisDef]:
    if not charters:
        return list(config.hypotheses)
    wanted: set[str] = set()
    for item in charters:
        wanted.add(
            pod_spec(
                item.value if isinstance(item, Charter) else item
            ).charter.value
        )
    chosen = [h for h in config.hypotheses if h.charter in wanted]
    if {h.charter for h in chosen} != wanted:
        raise CampaignError(
            "campaign config has no hypothesis for a requested charter"
        )
    return chosen


# --------------------------------------------------------------------------
# immutable records (written before any data row is read)


def _build_specs(
    hdef: HypothesisDef,
    campaign_id: str,
    dataset: ApprovedDataset,
    research_asof: dt.datetime,
) -> tuple[HypothesisSpec, ExperimentSpec]:
    short = campaign_id[5:15]
    pod = pod_spec(Charter(hdef.charter))
    recipe = RECIPES[hdef.recipe_id]
    ds = dataset.spec
    hyp_id, exp_id = f"hyp_{short}_{hdef.key}", f"exp_{short}_{hdef.key}"
    feature_ids = sorted(
        {
            f"{recipe.recipe_id}:{f.name}"
            for _, feats in recipe.feature_sets
            for f in feats
        }
    )
    baseline_id = f"baseline:{recipe.recipe_id}"
    family = recipe.mechanism_family
    hypothesis = HypothesisSpec(
        owner=pod.pod_id,
        created_at=research_asof,
        hypothesis_id=hyp_id,
        mechanism_family=family,
        research_mode=HypothesisMode.CONTEMPORARY_RESEARCH,
        question=hdef.question,
        mechanism=hdef.mechanism,
        target_id=recipe.target.target_id,
        reference_model_id=baseline_id,
        feature_spec_ids=feature_ids,
        applicability_spec_id=f"{dataset.dataset_id}:design_universe_v1",
        baseline_experiment_ids=[baseline_id],
        falsification=list(hdef.falsification),
        variant_budget=len(hdef.variants),
        experiment_spec_id=exp_id,
    )
    ev = ds.evaluation
    start, end = dataset.design_window
    experiment = ExperimentSpec(
        owner=pod.pod_id,
        created_at=research_asof,
        experiment_spec_id=exp_id,
        identity=ExperimentIdentity(
            hypothesis_id=hyp_id,
            variant_id=hdef.primary_variant,
            owner=pod.pod_id,
        ),
        claim=ExperimentClaim(
            primary_target_id=recipe.target.target_id,
            primary_metric="belief_ic_improvement",
            economic_effect=hdef.mechanism[:400],
            declared_sign=recipe.declared_direction[:200],
        ),
        data=ExperimentData(
            source_snapshot_ids=[ds.source_snapshot_id],
            feature_spec_ids=feature_ids,
            identity_graph_version=f"{ds.security_column}_identity_v1",
        ),
        sample=ExperimentSample(
            universe_spec_id=f"{dataset.dataset_id}:universe",
            eligibility_spec_id="complete_features_target_and_label_rows",
            applicability_spec_id=f"{dataset.dataset_id}:design_universe_v1",
        ),
        time=ExperimentTime(
            training_cutoff=end_of_day(end),
            label_maturity_rule=(
                f"label of month m resolves at end of month m+{recipe.target.months} "
                f"(+{ds.label.resolve_lag_days}d); usable only if resolved inside the design window"
            ),
            partitions={"design": f"{start.isoformat()}/{end.isoformat()}"},
            purge_rule="walk-forward training uses only months whose target resolved by the test month's information date",
            embargo_rule=f"embargo equals the target horizon of {recipe.target.months} month(s) via the maturity rule",
        ),
        model=ExperimentModel(
            method_id="walk_forward_ridge_or_rank_blend",
            hyperparameter_space={
                "variants": {v.name: dict(v.params) for v in hdef.variants}
            },
            seed_policy="deterministic_no_randomness",
        ),
        portfolio=ExperimentPortfolio(
            entry_rule=f"monthly at information month end; long top and short bottom {ev.tail_fraction:.0%} equal weight",
            exit_rule="next monthly rebalance",
            horizon_trading_days=21 * ds.label.resolve_months,
            construction_policy_id="equal_weight_tails_v1",
            hedge_policy_id="none_diagnostic_only",
        ),
        costs=ExperimentCosts(
            trade_cost_model_id=f"flat_{ev.cost_bps_one_way:g}bps_one_way",
            borrow_model_id="not_modelled",
            liquidity_model_id="not_modelled",
            funding_model_id="not_modelled",
        ),
        comparisons=ExperimentComparisons(
            baseline_experiment_ids=[baseline_id],
            paired_comparison_policy_id="paired_monthly_newey_west_v1",
            permitted_sensitivities=[v.name for v in hdef.variants],
        ),
        statistics=ExperimentStatistics(
            dependence_handling=f"month-level pairing; Newey-West Bartlett lag = target horizon ({recipe.target.months}) - 1",
            uncertainty_method="normal-approximation 95% interval; descriptive",
            selection_accounting_policy_id="bonferroni_over_declared_campaign_variants_v1",
        ),
        admission=ExperimentAdmission(
            min_coverage=f">= {ev.min_eval_months} evaluable months and >= {ev.min_names_per_month} names per month",
            effect_requirement="positive paired improvement in belief rank IC with the 95% interval above zero",
            stability_requirement="same sign in both halves of the evaluated months and a majority of calendar years",
            implementation_requirement="check-window and cost evidence owned by Research Integrity; not available to executors",
        ),
        budget=ExperimentBudget(
            compute_budget="single-process pure Python, bounded by dataset max_rows",
            provider_budget="none",
            llm_budget="none",
            evaluation_access_bounds="design partition only; check/replay/prospective denied",
        ),
        promotion_conditions="belief gates and bridge translation pass on the primary variant; certificate only via independent Research Integrity",
        failure_conditions="; ".join(hdef.falsification)[:900],
    )
    return hypothesis, experiment


def _register_once(registry: Registry, table: str, record: Any) -> str:
    sealed = record.sealed()
    existing = registry.get(table, sealed.stable_id)
    if existing is not None:
        if existing.get("content_hash") != sealed.content_hash:
            raise CampaignConflict(
                f"{table} record {sealed.stable_id} exists with different content; amendments need new ids"
            )
        return sealed.content_hash
    registry.register(table, record)
    return sealed.content_hash


def _count_once(
    registry: Registry, counter_id: str, kind: str, spec_hash: str
) -> None:
    try:
        registry.count_trial(counter_id, kind, spec_hash)
    except sqlite3.IntegrityError:
        pass  # a rerun re-presents the same counted trial; it is not counted twice


def _proposal(
    hdef: HypothesisDef,
    dataset: ApprovedDataset,
    executor_id: str,
    variants: list[ExperimentVariant],
) -> ResearchProposal:
    recipe = RECIPES[hdef.recipe_id]
    seen: dict[str, FeatureDefinition] = {}
    for _, feats in recipe.feature_sets:
        for f in feats:
            seen.setdefault(
                f.name,
                FeatureDefinition(
                    name=f.name,
                    definition=f"Centered cross-sectional {f.op} of panel column {f.column}"
                    + (f" and {f.column2}" if f.column2 else "")
                    + (f" over {f.months} months" if f.months else ""),
                    timing="Known at the information month end per the dataset availability attestation.",
                    source_ids=[dataset.spec.source_id],
                ),
            )
    return ResearchProposal(
        title=hdef.title[:200],
        mechanism=hdef.mechanism,
        falsifiable_target=" ".join(hdef.falsification[:1])[:800].ljust(
            20, "."
        ),
        baseline=recipe.baseline_description,
        features=list(seen.values())[:12],
        evidence_refs=[dataset.spec.source_id],
        experiment_plan=ExperimentPlan(
            executor_id=executor_id,
            primary_metric="belief_ic_improvement",
            data_partition="design",
            variants=variants,
            description="Run the registered built-in executor on the design partition of the approved dataset.",
        ),
        competing_explanation="The effect may be compensation for common style exposure rather than the stated mechanism.",
        limitations=[lim[:600] for lim in recipe.limitations]
        or ["Design-partition retrospective evidence only."],
        missing_information=[],
        evidence_quality="weak",
    )


# --------------------------------------------------------------------------
# status from pre-declared gates through the existing promotion contracts


def _gates(
    metrics: dict[str, float],
    thresholds: Thresholds,
    dataset: ApprovedDataset,
    m_total: int,
) -> dict[str, Any]:
    ev = dataset.spec.evaluation
    z = NormalDist().inv_cdf(1.0 - thresholds.alpha / (2.0 * max(1, m_total)))
    t = metrics.get("belief_ic_improvement_t")
    first, second = (
        metrics.get("belief_ic_improvement_first_half"),
        metrics.get("belief_ic_improvement_second_half"),
    )
    halves = (
        first is not None and second is not None and first > 0 and second > 0
    )
    year_share = metrics.get("belief_positive_year_share")
    stability = (halves if thresholds.require_halves_same_sign else True) and (
        year_share is None or year_share >= thresholds.min_positive_year_share
    )
    return {
        "coverage_met": metrics.get("coverage_eval_months", 0.0)
        >= ev.min_eval_months
        and metrics.get("coverage_names_per_month_min", 0.0)
        >= ev.min_names_per_month,
        "improvement_met": metrics.get("belief_ic_improvement", -1.0)
        > thresholds.min_belief_ic_improvement,
        "direction_met": metrics.get("belief_ic_candidate", -1.0) > 0.0
        and metrics.get("bridge_ic_candidate", -1.0)
        > thresholds.min_bridge_ic_candidate,
        "stability_met": bool(stability),
        # Design-partition, proxy-cost evidence cannot establish implementation; Research Integrity owns that.
        "implementation_met": False,
        "uncertainty_met": metrics.get("belief_ic_improvement_ci95_low", -1.0)
        > 0.0,
        "multiple_testing_met": t is not None and t >= z,
        "return_translation_positive": metrics.get(
            "bridge_net_spread_candidate", -1.0
        )
        > 0.0,
        "bonferroni_z_threshold": z,
        "primary_belief_t": t,
    }


def _promotion(
    registration_id: str,
    gates: dict[str, Any],
    thresholds: Thresholds,
    dataset: ApprovedDataset,
    research_asof: dt.datetime,
    spec_hash: str,
) -> dict[str, Any]:
    criteria = PromotionCriteria(
        min_coverage=f">= {dataset.spec.evaluation.min_eval_months} months and >= {dataset.spec.evaluation.min_names_per_month} names per month",
        improvement_vs_primary_baseline=f"paired mean belief rank-IC improvement > {thresholds.min_belief_ic_improvement}",
        direction_and_economic_size="candidate belief and bridge rank IC positive after the declared bridge sign; positive net tail spread",
        stability_requirement="both halves positive"
        + f"; positive-year share >= {thresholds.min_positive_year_share}",
        return_translation_requirement="net tail spread after flat one-way cost > 0 (diagnostic only)",
        implementation_requirement="check-window and cost evidence from Research Integrity (not obtainable from the design partition)",
        permitted_statistical_uncertainty="Newey-West 95% interval of the paired improvement excludes zero",
        multiple_testing_treatment=f"Bonferroni over declared campaign variants at alpha={thresholds.alpha}",
    )
    outcome = PromotionOutcome(
        registration_id=registration_id,
        evaluated_at=research_asof,
        criteria=criteria,
        coverage_met=gates["coverage_met"],
        improvement_met=gates["improvement_met"],
        direction_met=gates["direction_met"],
        stability_met=gates["stability_met"],
        implementation_met=gates["implementation_met"],
        uncertainty_met=gates["uncertainty_met"],
        multiple_testing_met=gates["multiple_testing_met"],
    )
    if outcome.admitted:  # unreachable while implementation_met is fixed False; fail closed regardless
        raise CampaignError(
            "campaign must never admit; certificates are issued only by independent Research Integrity"
        )
    try:
        outcome.certificate_or_raise(
            model_id=f"model_{registration_id}", model_hash=spec_hash
        )
    except PodPlatformError as exc:
        denial = exc.response.summary
    else:  # pragma: no cover
        raise CampaignError("promotion logic unexpectedly issued a certificate")
    return {
        "admitted": False,
        "certificate_id": None,
        "forecast_publication_id": None,
        "denial": denial,
        "criteria": criteria.model_dump(mode="json"),
        "flags": {
            k: bool(gates[k])
            for k in (
                "coverage_met",
                "improvement_met",
                "direction_met",
                "stability_met",
                "implementation_met",
                "uncertainty_met",
                "multiple_testing_met",
            )
        },
    }


def _status(
    fixture: bool, primary_failed: bool, gates: Optional[dict[str, Any]]
) -> tuple[str, Optional[str]]:
    if fixture:
        return "fixture_demo", None
    if primary_failed or gates is None:
        return "rejected", "REJECTED"
    shadow = all(
        gates[k]
        for k in (
            "coverage_met",
            "improvement_met",
            "direction_met",
            "stability_met",
            "uncertainty_met",
            "return_translation_positive",
        )
    )
    return ("shadow", "SHADOW") if shadow else ("rejected", "REJECTED")


# --------------------------------------------------------------------------
# campaign


def run_charter_campaign(
    manifest_path: "str | Path",
    state_dir: "str | Path",
    *,
    dataset_id: str,
    research_asof: dt.datetime,
    campaign_config: "CampaignConfig | dict | str | Path | None" = None,
    charters: Optional[Sequence[str]] = None,
    clock: Optional[Callable[[], dt.datetime]] = None,
    variant_budget: Optional[int] = None,
) -> dict[str, Any]:
    """Run the four-charter (or a selected subset) campaign and return the report dict (also written to disk)."""
    research_asof = require_aware(research_asof, "research_asof")
    clock = clock or _utc_now
    if research_asof > require_aware(clock(), "clock"):
        raise CampaignError(
            "research_asof lies in the future; contemporary research cannot be dated ahead of the clock"
        )
    if not state_dir:
        raise CampaignError(
            "state_dir is required (registry, trial ledger and report are persisted there)"
        )
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    config = load_campaign_config(campaign_config)
    hdefs = _select(config, charters)
    budget = variant_budget or max(len(h.variants) for h in hdefs)
    if any(len(h.variants) > budget for h in hdefs):
        raise CampaignError(
            f"variant_budget {budget} is smaller than the number of declared variants"
        )

    context = BuiltinContext(
        manifest_path,
        state,
        clock=clock,
        research_asof=research_asof,
        variant_budget=budget,
    )
    registry = Registry(state / "registry.sqlite")
    try:
        dataset = context.manifest.dataset(
            dataset_id
        )  # fails closed if undeclared / unapproved / ineligible source
        campaign_id = (
            "camp_"
            + canonical_digest(
                {
                    "manifest": context.manifest.manifest_hash,
                    "dataset": dataset.descriptor_hash,
                    "research_asof": research_asof.isoformat(),
                    "config": config.digest(),
                    "hypotheses": [h.key for h in hdefs],
                    "budget": budget,
                }
            )[:20]
        )

        # 1) immutable hypotheses and specifications, BEFORE any data row is read
        records: dict[str, tuple[HypothesisSpec, ExperimentSpec]] = {}
        hashes: dict[str, dict[str, str]] = {
            "hypotheses": {},
            "experiment_specs": {},
        }
        for hdef in hdefs:
            hypothesis, experiment = _build_specs(
                hdef, campaign_id, dataset, research_asof
            )
            hashes["hypotheses"][hypothesis.hypothesis_id] = _register_once(
                registry, "hypotheses", hypothesis
            )
            hashes["experiment_specs"][experiment.experiment_spec_id] = (
                _register_once(registry, "experiment_specs", experiment)
            )
            _count_once(
                registry,
                f"{campaign_id}:{hdef.key}:hypothesis",
                "hypothesis",
                hashes["hypotheses"][hypothesis.hypothesis_id],
            )
            _count_once(
                registry,
                f"{campaign_id}:{hdef.key}:spec",
                "registered_specification",
                hashes["experiment_specs"][experiment.experiment_spec_id],
            )
            records[hdef.key] = (hypothesis, experiment)

        # 2) evaluation through the same registered executors a pod would use
        executors = ExecutorRegistry()
        context.register(executors)
        m_total = sum(len(h.variants) for h in hdefs)
        candidates: list[dict[str, Any]] = []
        variant_report_hashes: dict[str, str] = {}
        for hdef in hdefs:
            hypothesis, experiment = records[hdef.key]
            candidate = _run_candidate(
                hdef,
                hypothesis,
                experiment,
                dataset,
                context,
                executors,
                registry,
                config.thresholds,
                m_total,
                research_asof,
                variant_report_hashes,
            )
            candidates.append(candidate)

        fixture = dataset.fixture
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "campaign_id": campaign_id,
            "research_mode": RESEARCH_MODE,
            "research_asof": research_asof.isoformat(),
            "evaluation_label": EVALUATION_LABEL,
            "dataset_id": dataset.dataset_id,
            "data_class": dataset.spec.data_class,
            "fixture": fixture,
            "source_ids": [dataset.spec.source_id],
            "source_snapshot_ids": [dataset.spec.source_snapshot_id],
            "candidates": candidates,
            "artifact_hashes": {
                "manifest": context.manifest.manifest_hash,
                "campaign_config": config.digest(),
                "dataset_descriptor": dataset.descriptor_hash,
                "source_card": dataset.source_card_hash,
                "data_file_sha256": file_sha256(dataset.path),
                "hypotheses": hashes["hypotheses"],
                "experiment_specs": hashes["experiment_specs"],
                "variant_reports": variant_report_hashes,
            },
            "trial_accounting": {
                "declared_variants": m_total,
                "variant_budget_per_hypothesis": budget,
                "executor_ledger": context.ledger.trial_counts(),
                "registry": {
                    kind: registry.trial_count(kind)
                    for kind in (
                        "hypothesis",
                        "registered_specification",
                        "scored_variant",
                        "hyperparameter_config",
                        "fold_evaluation",
                    )
                },
            },
            "limitations": [
                "Contemporary research: publication-date information is treated as available now; historical performance is retrospective and is not a fresh holdout.",
                "Only the design partition (before 2019) was readable; check, replay and prospective partitions were denied to every executor.",
                "Status derives from one pre-declared primary variant per hypothesis; other variants are reported and counted, not selected.",
                "No certificate or forecast publication is issued here; shadow means 'eligible for Research Integrity's check window', not admitted.",
                "Belief quality (belief_*) and the return bridge (bridge_*) are separate evidence and are never merged into a single score.",
            ]
            + (
                [
                    "FIXTURE DATA: synthetic; nothing here authorizes anything about a real competition book."
                ]
                if fixture
                else []
            ),
        }
        report["report_hash"] = canonical_digest(report)
        out_dir = state / "campaigns" / campaign_id
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "report.json.tmp"
        tmp.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(tmp, out_dir / "report.json")
        return report
    finally:
        registry.close()
        context.close()


def _run_candidate(
    hdef: HypothesisDef,
    hypothesis: HypothesisSpec,
    experiment: ExperimentSpec,
    dataset: ApprovedDataset,
    context: BuiltinContext,
    executors: ExecutorRegistry,
    registry: Registry,
    thresholds: Thresholds,
    m_total: int,
    research_asof: dt.datetime,
    variant_report_hashes: dict[str, str],
) -> dict[str, Any]:
    pod = pod_spec(Charter(hdef.charter))
    recipe = RECIPES[hdef.recipe_id]
    executor_id = EXECUTOR_IDS[hdef.charter]
    registration_id = f"{pod.namespace}:{experiment.experiment_spec_id}"
    variants = [
        ExperimentVariant(
            name=v.name, params={**v.params, "dataset_id": dataset.dataset_id}
        )
        for v in hdef.variants
    ]
    proposal = _proposal(hdef, dataset, executor_id, variants)
    outputs: list[dict[str, Any]] = []
    failure_notes: list[dict[str, Any]] = []
    primary_gates: Optional[dict[str, Any]] = None
    primary_report: Optional[dict[str, Any]] = None
    primary_result = None
    for variant in variants:
        request = ExperimentRequest(
            pod.namespace,
            experiment.experiment_spec_id,
            executor_id,
            variant,
            proposal,
        )
        entry: dict[str, Any] = {
            "variant_name": variant.name,
            "params": dict(variant.params),
            "primary": variant.name == hdef.primary_variant,
        }
        try:
            result = executors.run(request)
        except ExecutorError as exc:
            entry.update({"status": "failed", "failure": str(exc)[:600]})
            failure_notes.append(
                {"variant_name": variant.name, "message": str(exc)[:600]}
            )
            outputs.append(entry)
            continue
        report = context.report(registration_id, variant.name) or {}
        gates = _gates(result.metrics, thresholds, dataset, m_total)
        entry.update(
            {
                "status": "evaluated",
                "spec_hash": result.spec_hash,
                "metrics": result.metrics,
                "notes": result.notes,
                "report_hash": report.get("report_hash"),
                "candidate_artifact": report.get("candidate_artifact"),
                "gates": {k: v for k, v in gates.items()},
            }
        )
        if report.get("report_hash"):
            variant_report_hashes[f"{registration_id}/{variant.name}"] = report[
                "report_hash"
            ]
        accounting = report.get("trial_accounting", {})
        _count_once(
            registry,
            f"{registration_id}:{variant.name}:scored_variant",
            "scored_variant",
            result.spec_hash,
        )
        if accounting.get("hyperparameter_configs"):
            _count_once(
                registry,
                f"{registration_id}:{variant.name}:hyperparameter_config",
                "hyperparameter_config",
                result.spec_hash,
            )
        for i in range(int(accounting.get("fold_evaluations", 0))):
            _count_once(
                registry,
                f"{registration_id}:{variant.name}:fold:{i}",
                "fold_evaluation",
                result.spec_hash,
            )
        try:
            registry.record_evaluation_access(
                f"{registration_id}:{variant.name}:design",
                "design",
                pod.pod_id,
                "built-in executor design-partition evaluation",
                int(report.get("data", {}).get("rows_in_window", 0)),
            )
        except sqlite3.IntegrityError:
            pass
        if variant.name == hdef.primary_variant:
            primary_gates, primary_report, primary_result = (
                gates,
                report,
                result,
            )
        outputs.append(entry)

    fixture = dataset.fixture
    status, recommended = _status(
        fixture, primary_result is None, primary_gates
    )
    promotion = None
    if primary_result is not None and primary_gates is not None:
        promotion = _promotion(
            registration_id,
            primary_gates,
            thresholds,
            dataset,
            research_asof,
            primary_result.spec_hash,
        )
    limitations = (
        list(primary_report.get("limitations", []))
        if primary_report
        else list(recipe.limitations)
    )
    limitations.append(
        "Status derives from the pre-declared primary variant; other variants are counted for multiple testing."
    )
    labels = (
        primary_report.get("labels")
        if primary_report
        else {"proxy_labels": list(recipe.proxy_labels)}
    )
    return {
        "charter": hdef.charter,
        "pod_id": pod.pod_id,
        "hypothesis_key": hdef.key,
        "hypothesis_id": hypothesis.hypothesis_id,
        "experiment_spec_id": experiment.experiment_spec_id,
        "registration_id": registration_id,
        "executor_id": executor_id,
        "recipe_id": recipe.recipe_id,
        "belief_target": recipe.target.target_id,
        "baseline": recipe.baseline_description,
        "bridge": {
            "label_column": dataset.spec.label.column,
            "sign": recipe.bridge_sign,
            "direction": recipe.declared_direction,
        },
        "status": status,
        "recommended_state": recommended,
        "primary_variant": hdef.primary_variant,
        "metrics": dict(primary_result.metrics)
        if primary_result is not None
        else {},
        "gates": {k: v for k, v in (primary_gates or {}).items()},
        "promotion": promotion,
        "variants": outputs,
        "labels": labels,
        "limitations": limitations,
        "failure_notes": failure_notes,
    }
