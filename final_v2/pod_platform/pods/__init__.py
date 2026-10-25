"""Four research pods (one per strategy charter) and their bounded LLM workflow."""

from .charters import POD_SPECS, PodSpec, pod_spec
from .runner import (
    ExecutorError,
    ExecutorRegistry,
    ExperimentRequest,
    ExperimentResult,
    PodResearchRunner,
    PodWorker,
    ProposalResult,
    WorkerOutcome,
    build_system_prompt,
)
from .schema import (
    CitationError,
    EvidenceError,
    EvidenceItem,
    ExperimentPlan,
    ExperimentVariant,
    FeatureDefinition,
    ResearchProposal,
    filter_eligible,
    normalise_evidence,
    validate_citations,
)

__all__ = [
    "POD_SPECS",
    "CitationError",
    "EvidenceError",
    "EvidenceItem",
    "ExecutorError",
    "ExecutorRegistry",
    "ExperimentPlan",
    "ExperimentRequest",
    "ExperimentResult",
    "ExperimentVariant",
    "FeatureDefinition",
    "PodResearchRunner",
    "PodSpec",
    "PodWorker",
    "ProposalResult",
    "ResearchProposal",
    "WorkerOutcome",
    "build_system_prompt",
    "filter_eligible",
    "normalise_evidence",
    "pod_spec",
    "validate_citations",
]
