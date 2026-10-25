"""Experiment specification contract (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 21.5, 13.1)."""

from __future__ import annotations

import datetime as dt

from pydantic import Field, model_validator

from .base import DurableModel, StrictModel, require_aware


class ExperimentIdentity(StrictModel):
    hypothesis_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    parent_experiment_id: str | None = None
    owner: str = Field(min_length=1)


class ExperimentClaim(StrictModel):
    primary_target_id: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    economic_effect: str = Field(min_length=1)
    declared_sign: str | None = None


class ExperimentData(StrictModel):
    source_snapshot_ids: list[str] = Field(min_length=1)
    feature_spec_ids: list[str] = Field(min_length=1)
    identity_graph_version: str = Field(min_length=1)


class ExperimentSample(StrictModel):
    universe_spec_id: str = Field(min_length=1)
    eligibility_spec_id: str = Field(min_length=1)
    applicability_spec_id: str = Field(min_length=1)


class ExperimentTime(StrictModel):
    training_cutoff: dt.datetime
    label_maturity_rule: str = Field(min_length=1)
    partitions: dict[str, str] = Field(min_length=1)
    purge_rule: str = Field(min_length=1)
    embargo_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_aware(self) -> "ExperimentTime":
        require_aware(self.training_cutoff, "training_cutoff")
        return self


class ExperimentModel(StrictModel):
    method_id: str = Field(min_length=1)
    hyperparameter_space: dict[str, object] = Field(default_factory=dict)
    seed_policy: str = Field(min_length=1)


class ExperimentPortfolio(StrictModel):
    entry_rule: str = Field(min_length=1)
    exit_rule: str = Field(min_length=1)
    horizon_trading_days: int = Field(ge=1)
    construction_policy_id: str = Field(min_length=1)
    hedge_policy_id: str = Field(min_length=1)


class ExperimentCosts(StrictModel):
    trade_cost_model_id: str = Field(min_length=1)
    borrow_model_id: str = Field(min_length=1)
    liquidity_model_id: str = Field(min_length=1)
    funding_model_id: str = Field(min_length=1)


class ExperimentComparisons(StrictModel):
    baseline_experiment_ids: list[str] = Field(min_length=1)
    paired_comparison_policy_id: str = Field(min_length=1)
    permitted_sensitivities: list[str] = Field(default_factory=list)


class ExperimentStatistics(StrictModel):
    dependence_handling: str = Field(min_length=1)
    uncertainty_method: str = Field(min_length=1)
    selection_accounting_policy_id: str = Field(min_length=1)


class ExperimentAdmission(StrictModel):
    min_coverage: str = Field(min_length=1)
    effect_requirement: str = Field(min_length=1)
    stability_requirement: str = Field(min_length=1)
    implementation_requirement: str = Field(min_length=1)


class ExperimentBudget(StrictModel):
    compute_budget: str = Field(min_length=1)
    provider_budget: str = Field(min_length=1)
    llm_budget: str = Field(min_length=1)
    evaluation_access_bounds: str = Field(min_length=1)


class ExperimentSpec(DurableModel):
    ID_FIELD = "experiment_spec_id"

    experiment_spec_id: str = Field(min_length=1)
    identity: ExperimentIdentity
    claim: ExperimentClaim
    data: ExperimentData
    sample: ExperimentSample
    time: ExperimentTime
    model: ExperimentModel
    portfolio: ExperimentPortfolio
    costs: ExperimentCosts
    comparisons: ExperimentComparisons
    statistics: ExperimentStatistics
    admission: ExperimentAdmission
    budget: ExperimentBudget
    promotion_conditions: str = Field(min_length=1)
    failure_conditions: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> "ExperimentSpec":
        if self.identity.parent_experiment_id == self.experiment_spec_id:
            raise ValueError(
                "parent_experiment_id cannot reference the experiment itself"
            )
        if self.experiment_spec_id in self.comparisons.baseline_experiment_ids:
            raise ValueError("an experiment cannot be its own baseline")
        if len(set(self.comparisons.baseline_experiment_ids)) != len(
            self.comparisons.baseline_experiment_ids
        ):
            raise ValueError("baseline_experiment_ids must be unique")
        return self
