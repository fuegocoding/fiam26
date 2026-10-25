"""Typed failures for the built-in executors. All are fail-closed ``ValueError``/``PermissionError`` subclasses
so existing CLIs that catch ``ValueError`` report them cleanly."""

from __future__ import annotations


class BuiltinConfigError(ValueError):
    """Manifest, path, schema or parameter is not operator-approved or is malformed."""


class SourceNotEligible(BuiltinConfigError):
    """SOURCE_NOT_FORMATION_ELIGIBLE: the source card does not authorize formation use."""


class DataIntegrityError(ValueError):
    """The data file violates the declared schema, identity, finiteness or availability rules."""


class EvidenceInsufficient(ValueError):
    """The data is valid but too thin to support the claim; reported as a failure note, never as a result."""


class PartitionDenied(PermissionError):
    """PARTITION_ACCESS_DENIED: a protected or undeclared partition was requested."""


class VariantBudgetExhausted(ValueError):
    """The registered variant budget is spent, or a variant name was reused with a different specification."""


class DependencyUnavailable(RuntimeError):
    """An optional dependency (pyarrow) needed for this dataset format is not installed."""
