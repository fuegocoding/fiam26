"""Built-in, reviewed experiment executors for the four strategy charters.

Use ``register_executors(registry)`` (optionally via ``run_pod_research.py --executors-module
pod_platform.research.builtins``). Datasets come only from an operator manifest (``POD_DATASET_MANIFEST``).
"""

from .catalog import (
    RECIPES,
    RECIPES_BY_CHARTER,
    Recipe,
    VariantConfig,
    parse_variant_params,
)
from .errors import (
    BuiltinConfigError,
    DataIntegrityError,
    DependencyUnavailable,
    EvidenceInsufficient,
    PartitionDenied,
    SourceNotEligible,
    VariantBudgetExhausted,
)
from .executors import (
    CHARTER_BY_EXECUTOR,
    EXECUTOR_IDS,
    BuiltinContext,
    catalog_summary,
    register_executors,
)
from .fixtures import write_fixture_dataset
from .ledger import BuiltinLedger
from .manifest import LoadedManifest, load_manifest

__all__ = [
    "CHARTER_BY_EXECUTOR",
    "EXECUTOR_IDS",
    "RECIPES",
    "RECIPES_BY_CHARTER",
    "BuiltinConfigError",
    "BuiltinContext",
    "BuiltinLedger",
    "DataIntegrityError",
    "DependencyUnavailable",
    "EvidenceInsufficient",
    "LoadedManifest",
    "PartitionDenied",
    "Recipe",
    "SourceNotEligible",
    "VariantBudgetExhausted",
    "VariantConfig",
    "catalog_summary",
    "load_manifest",
    "parse_variant_params",
    "register_executors",
    "write_fixture_dataset",
]
