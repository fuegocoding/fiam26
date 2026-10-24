"""Research-layer exports."""

from .partitions import (
    CHECK,
    DESIGN,
    PROSPECTIVE,
    PROTECTED_PARTITIONS,
    REPLAY,
    LabelWindow,
    LateStartPolicy,
    Partition,
    PartitionPlan,
    embargo_after,
    matured_labels,
    purge_between,
)
from .promotion import (
    ALLOWED_TRANSITIONS,
    AdmissionCertificate,
    ModelLifecycle,
    ModelState,
    PromotionCriteria,
    PromotionOutcome,
    StateTransition,
)
from .registration import (
    TRIAL_KINDS,
    RegisteredExperiment,
    ResearchLedger,
    TrialRecord,
)
from .registry import Registry

__all__ = [
    "ALLOWED_TRANSITIONS",
    "CHECK",
    "DESIGN",
    "AdmissionCertificate",
    "LabelWindow",
    "LateStartPolicy",
    "ModelLifecycle",
    "ModelState",
    "PROSPECTIVE",
    "PROTECTED_PARTITIONS",
    "Partition",
    "PartitionPlan",
    "REPLAY",
    "RegisteredExperiment",
    "Registry",
    "ResearchLedger",
    "StateTransition",
    "TRIAL_KINDS",
    "TrialRecord",
    "PromotionCriteria",
    "PromotionOutcome",
    "embargo_after",
    "matured_labels",
    "purge_between",
]
