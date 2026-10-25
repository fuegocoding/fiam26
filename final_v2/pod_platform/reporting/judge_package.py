"""Judge-facing evidence package (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 28.4-28.5).

The package requires one complete investment example, one rejected idea,
one failed trade, the research funnel, incremental portfolio evidence,
implementation evidence, agent evidence, official statistics and a
position-dossier lookup. Handpicked success stories alone are rejected.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from ..contracts.base import StrictModel
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse


class WorkedExample(StrictModel):
    example_id: str = Field(min_length=1)
    kind: str = Field(
        pattern=r"^(complete_investment|rejected_idea|failed_trade)$"
    )
    security_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    measurements: str = Field(min_length=1)
    test: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ResearchFunnel(StrictModel):
    hypotheses_registered: int = Field(ge=0)
    experiments_run: int = Field(ge=0)
    variants_scored: int = Field(ge=0)
    admitted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    descriptive_only: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> "ResearchFunnel":
        total_out = self.admitted + self.rejected + self.descriptive_only
        if total_out > self.experiments_run:
            raise ValueError("outcomes cannot exceed experiments run")
        return self


class IncrementalEvidence(StrictModel):
    baseline_comparison_id: str = Field(min_length=1)
    strategy_contributions: dict[str, str] = Field(min_length=1)
    redundancy_notes: str = Field(min_length=1)
    removal_counterfactual_id: Optional[str] = None


class ImplementationEvidence(StrictModel):
    alpha_decay_curve_id: str = Field(min_length=1)
    liquidity_evidence_id: str = Field(min_length=1)
    turnover_summary: str = Field(min_length=1)
    borrow_cost_summary: str = Field(min_length=1)
    cost_sensitivity_id: str = Field(min_length=1)


class AgentEvidence(StrictModel):
    counterfactual_id: str = Field(min_length=1)
    worked_case: str = Field(min_length=1)
    interventions: int = Field(ge=0)
    net_effect_summary: str = Field(min_length=1)


class JudgePackage(StrictModel):
    package_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    complete_investment_example: WorkedExample
    rejected_idea: WorkedExample
    failed_trade: WorkedExample
    research_funnel: ResearchFunnel
    incremental_evidence: IncrementalEvidence
    implementation_evidence: ImplementationEvidence
    agent_evidence: AgentEvidence
    official_statistics_artifact_id: str = Field(min_length=1)
    dossier_demo: str = Field(min_length=1)
    presentation_plan: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_kinds(self) -> "JudgePackage":
        expected = {
            "complete_investment_example": "complete_investment",
            "rejected_idea": "rejected_idea",
            "failed_trade": "failed_trade",
        }
        for attr, kind in expected.items():
            example: WorkedExample = getattr(self, attr)
            if example.kind != kind:
                raise ValueError(
                    f"{attr} must have kind {kind!r}, got {example.kind!r}"
                )
        return self

    def narrative_coverage(self) -> dict[str, bool]:
        return {
            "one_complete_investment_example": True,
            "one_rejected_idea": True,
            "one_failed_trade": True,
            "research_funnel": self.research_funnel.experiments_run > 0,
            "incremental_portfolio_evidence": bool(
                self.incremental_evidence.baseline_comparison_id
            ),
            "implementation_evidence": bool(
                self.implementation_evidence.alpha_decay_curve_id
            ),
            "agent_evidence": bool(self.agent_evidence.counterfactual_id),
            "official_statistics": bool(self.official_statistics_artifact_id),
            "dossier_lookup": bool(self.dossier_demo),
        }


def verify_package(package: JudgePackage) -> dict:
    coverage = package.narrative_coverage()
    if not all(coverage.values()):
        missing = [name for name, ok in coverage.items() if not ok]
        raise PodPlatformError(
            ToolResponse.failure(
                summary=f"judge package incomplete: {', '.join(missing)}",
                code=ErrorCode.SCHEMA_MISMATCH,
                stop_condition="Selecting only successful examples is not acceptable; the package needs success, rejection and failure.",
            )
        )
    return {
        "package_id": package.package_id,
        "run_id": package.run_id,
        "coverage": coverage,
        "presentation_minutes": len(package.presentation_plan),
    }
