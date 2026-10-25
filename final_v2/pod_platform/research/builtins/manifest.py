"""Operator dataset manifest: the only way a built-in executor learns a file path, schema or label definition.

LLM-supplied parameters can only *name* a manifest dataset id and a reviewed recipe id. They can never supply a
path, a column, a label or code. Paths are confined to the manifest directory (or explicit ``allowed_roots``),
``..`` is rejected, symlinks are resolved before the containment check, and every dataset must be marked
``approved`` and carry an approved, formation-eligible source card whose snapshot matches the declaration.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, ValidationInfo, field_validator, model_validator

from ...contracts.base import StrictModel, canonical_digest, require_aware
from ...contracts.sources import CardStatus, SourceCard
from .errors import BuiltinConfigError, SourceNotEligible

MANIFEST_SCHEMA = "dataset_manifest.v1"
ENV_MANIFEST = "POD_DATASET_MANIFEST"
IDENT = r"^[a-z][a-z0-9_]{0,63}$"
COLUMN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
DESIGN = "design"
# The check window (2019-2020), replay and prospective partitions start here; a design window may never reach it.
PROTECTED_FROM = dt.date(2019, 1, 1)
DEFAULT_DESIGN_WINDOW = (dt.date(2015, 1, 1), dt.date(2018, 12, 31))
_COL_RE = re.compile(COLUMN)


class ColumnSpec(StrictModel):
    role: Literal["feature", "context", "label"]
    dtype: Literal["float", "str", "date"] = "float"
    allow_missing: bool = True
    min_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    max_value: Optional[float] = Field(default=None, allow_inf_nan=False)
    unit: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "ColumnSpec":
        if self.role in ("feature", "label") and self.dtype != "float":
            raise ValueError("feature and label columns must be float")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value exceeds max_value")
        return self


class AvailabilitySpec(StrictModel):
    """How the operator establishes that features are known by the row's time-column date."""

    mode: Literal["known_at_time_column", "row_column"]
    attestation: str = Field(min_length=10)
    column: Optional[str] = None
    on_violation: Literal["error", "exclude"] = "error"

    @model_validator(mode="after")
    def _check(self) -> "AvailabilitySpec":
        if (self.mode == "row_column") != (self.column is not None):
            raise ValueError(
                "availability.column is required for (and only for) mode=row_column"
            )
        return self


class LabelSpec(StrictModel):
    column: str = Field(pattern=COLUMN)
    resolve_months: int = Field(
        ge=1, le=12
    )  # the label of row month m is known at the end of month m+resolve_months
    resolve_lag_days: int = Field(default=0, ge=0, le=366)
    kind: Literal["excess_return"] = "excess_return"
    unit: Literal["decimal_return"] = "decimal_return"


class PartitionSpec(StrictModel):
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _check(self) -> "PartitionSpec":
        if self.end <= self.start:
            raise ValueError("partition end must lie after start")
        return self


class EvaluationSpec(StrictModel):
    min_train_months: int = Field(default=12, ge=3, le=60)
    min_eval_months: int = Field(default=12, ge=6, le=120)
    min_names_per_month: int = Field(default=100, ge=10)
    tail_fraction: float = Field(default=0.2, ge=0.05, le=0.34)
    cost_bps_one_way: float = Field(ge=0.0, le=200.0, allow_inf_nan=False)
    cost_assumption_note: str = Field(min_length=10)
    short_min_market_cap_usd: float = Field(
        default=5e8, ge=0.0, allow_inf_nan=False
    )
    sector_digits: int = Field(default=2, ge=1, le=6)
    max_rows: int = Field(default=2_000_000, ge=1000, le=20_000_000)


class ComponentEvaluationSpec(StrictModel):
    """Only metadata used by native component transforms; no financial assumptions."""

    sector_digits: int = Field(default=2, ge=1, le=6)


class DatasetSpec(StrictModel):
    universe_id: Optional[str] = Field(default=None, pattern=IDENT)
    approved: bool
    data_class: Literal["operator_authorized", "fixture"]
    path: str = Field(min_length=1)
    format: Literal["csv", "parquet"]
    sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_id: str = Field(min_length=1)
    source_snapshot_id: str = Field(min_length=1)
    source_card_path: Optional[str] = None
    source_card: Optional[dict[str, Any]] = None
    security_column: str = Field(pattern=COLUMN)
    time_column: str = Field(pattern=COLUMN)
    frequency: Literal["monthly"] = "monthly"
    availability: AvailabilitySpec
    label: Optional[LabelSpec] = None
    columns: dict[str, ColumnSpec] = Field(min_length=1)
    sector_column: Optional[str] = None
    market_cap_column: Optional[str] = None
    market_cap_to_usd: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    exposure_columns: list[str] = Field(default_factory=list, max_length=8)
    column_aliases: dict[str, str] = Field(default_factory=dict)
    approved_recipes: list[str] = Field(default_factory=list)
    partitions: dict[str, PartitionSpec] = Field(
        default_factory=lambda: {
            DESIGN: PartitionSpec(
                start=DEFAULT_DESIGN_WINDOW[0], end=DEFAULT_DESIGN_WINDOW[1]
            )
        }
    )
    evaluation: EvaluationSpec | ComponentEvaluationSpec = Field(
        default_factory=ComponentEvaluationSpec
    )
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _check(self, info: ValidationInfo) -> "DatasetSpec":
        research_only = bool(info.context and info.context.get("research_only"))
        if not research_only and (
            self.label is None
            or not isinstance(self.evaluation, EvaluationSpec)
        ):
            raise ValueError(
                "financial execution requires a declared label and complete evaluation/cost assumptions"
            )
        for name in self.columns:
            if not _COL_RE.match(name):
                raise ValueError(f"invalid column name {name!r}")
        if (
            self.security_column in self.columns
            or self.time_column in self.columns
        ):
            raise ValueError(
                "security_column and time_column are structural and must not be listed in columns"
            )
        if (self.source_card_path is None) == (self.source_card is None):
            raise ValueError(
                "exactly one of source_card_path or source_card is required"
            )
        if self.label is not None:
            label = self.columns.get(self.label.column)
            if label is None or label.role != "label":
                raise ValueError(
                    "label.column must be declared in columns with role=label"
                )
        if sum(1 for c in self.columns.values() if c.role == "label") != (
            1 if self.label else 0
        ):
            raise ValueError(
                "exactly one label column may be declared when a label is specified"
            )
        if self.sector_column is not None:
            spec = self.columns.get(self.sector_column)
            if spec is None or spec.dtype != "str" or spec.role != "context":
                raise ValueError(
                    "sector_column must be a declared context column with dtype=str"
                )
        if self.market_cap_column is not None:
            spec = self.columns.get(self.market_cap_column)
            if spec is None or spec.dtype != "float" or spec.role == "label":
                raise ValueError(
                    "market_cap_column must be a declared non-label float column"
                )
        for name in self.exposure_columns:
            spec = self.columns.get(name)
            if spec is None or spec.dtype != "float" or spec.role == "label":
                raise ValueError(
                    f"exposure column {name!r} must be a declared non-label float column"
                )
        if self.availability.column is not None:
            spec = self.columns.get(self.availability.column)
            if spec is None or spec.dtype != "date":
                raise ValueError(
                    "availability.column must be declared with dtype=date"
                )
        for logical, physical in self.column_aliases.items():
            if not _COL_RE.match(logical) or physical not in self.columns:
                raise ValueError(
                    f"column alias {logical!r}->{physical!r} must name a declared column"
                )
            if self.columns[physical].role == "label":
                raise ValueError(
                    "a label column cannot be aliased into a feature"
                )
        for recipe_id in self.approved_recipes:
            if not re.match(IDENT, recipe_id):
                raise ValueError(f"invalid recipe id {recipe_id!r}")
        if set(self.partitions) - {DESIGN}:
            raise ValueError(
                "only the design partition may be declared; check/replay/prospective are never exposed here"
            )
        if DESIGN not in self.partitions:
            raise ValueError("a design partition is required")
        if self.partitions[DESIGN].end >= PROTECTED_FROM:
            raise ValueError(
                f"design partition must end before {PROTECTED_FROM.isoformat()}"
            )
        return self


class DatasetManifest(StrictModel):
    schema_version: Literal["dataset_manifest.v1"]
    manifest_id: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approved_at: dt.datetime
    allowed_roots: list[str] = Field(default_factory=list)
    datasets: dict[str, DatasetSpec] = Field(min_length=1)

    @field_validator("approved_at")
    @classmethod
    def _aware(cls, value: dt.datetime) -> dt.datetime:
        return require_aware(value, "approved_at")

    @field_validator("datasets")
    @classmethod
    def _ids(cls, value: dict[str, DatasetSpec]) -> dict[str, DatasetSpec]:
        for key in value:
            if not re.match(IDENT, key):
                raise ValueError(f"invalid dataset id {key!r}")
        return value


@dataclass(frozen=True)
class ApprovedDataset:
    dataset_id: str
    spec: DatasetSpec
    path: Path
    source_card: SourceCard
    source_card_hash: str
    descriptor_hash: str

    @property
    def design_window(self) -> tuple[dt.date, dt.date]:
        part = self.spec.partitions[DESIGN]
        return part.start, part.end

    @property
    def fixture(self) -> bool:
        return self.spec.data_class == "fixture"


@dataclass(frozen=True)
class LoadedManifest:
    path: Path
    manifest: DatasetManifest
    manifest_hash: str
    datasets: dict[str, ApprovedDataset]

    def dataset(self, dataset_id: Optional[str]) -> ApprovedDataset:
        if dataset_id is None:
            if len(self.datasets) == 1:
                return next(iter(self.datasets.values()))
            raise BuiltinConfigError(
                "dataset_id is required: the manifest approves several datasets or none"
            )
        if not isinstance(dataset_id, str) or not re.match(IDENT, dataset_id):
            raise BuiltinConfigError(
                "dataset_id must be an approved manifest dataset id"
            )
        found = self.datasets.get(dataset_id)
        if found is None:
            declared = dataset_id in self.manifest.datasets
            raise BuiltinConfigError(
                f"dataset {dataset_id!r} is "
                + (
                    "declared but not approved by the operator"
                    if declared
                    else "not in the operator manifest"
                )
            )
        return found


def resolve_manifest_path(manifest_path: "str | Path | None" = None) -> Path:
    raw = manifest_path if manifest_path else os.environ.get(ENV_MANIFEST)
    if not raw:
        raise BuiltinConfigError(
            f"no dataset manifest: pass manifest_path or set {ENV_MANIFEST} (private files are never auto-discovered)"
        )
    path = Path(str(raw))
    if not path.is_file():
        raise BuiltinConfigError(
            f"dataset manifest {str(path)!r} is not a file"
        )
    return path.resolve()


def _confined_path(raw: str, base: Path, roots: list[Path], what: str) -> Path:
    if (
        not raw
        or "\x00" in raw
        or raw.startswith("~")
        or (not Path(raw).is_absolute() and "\\" in raw)
    ):
        raise BuiltinConfigError(f"{what}: unsafe path {raw!r}")
    candidate = Path(raw)
    if ".." in candidate.parts:
        raise BuiltinConfigError(
            f"{what}: path traversal is not allowed ({raw!r})"
        )
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise BuiltinConfigError(
            f"{what}: file does not exist ({raw!r})"
        ) from None
    if not resolved.is_file():
        raise BuiltinConfigError(f"{what}: not a regular file ({raw!r})")
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise BuiltinConfigError(
            f"{what}: {raw!r} resolves outside the manifest directory and allowed_roots"
        )
    return resolved


def check_source_gate(
    spec: DatasetSpec, card: SourceCard, *, allow_research_only: bool = False
) -> None:
    """Fail closed unless the card is APPROVED, access-confirmed and formation-eligible for exactly this snapshot."""
    problems = []
    if allow_research_only and card.status in (
        CardStatus.REJECTED,
        CardStatus.QUARANTINED,
    ):
        problems.append(
            f"status is {card.status.value}; research access is denied"
        )
    elif not allow_research_only and card.status != CardStatus.APPROVED:
        problems.append(f"status is {card.status.value}, not APPROVED")
    if not card.access_confirmed:
        problems.append("access is not confirmed")
    if not allow_research_only and card.broker_rejects_formation():
        problems.append("source is not formation-eligible")
    if card.source_id != spec.source_id:
        problems.append(
            f"card source_id {card.source_id!r} != declared {spec.source_id!r}"
        )
    if card.snapshot_id != spec.source_snapshot_id:
        problems.append(
            f"card snapshot {card.snapshot_id!r} != declared {spec.source_snapshot_id!r}"
        )
    if problems:
        raise SourceNotEligible(
            "SOURCE_NOT_FORMATION_ELIGIBLE: " + "; ".join(problems)
        )


def _load_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuiltinConfigError(
            f"{what} is not readable JSON: {exc}"
        ) from None


def load_manifest(
    manifest_path: "str | Path | None" = None,
    *,
    allow_research_only: bool = False,
) -> LoadedManifest:
    """Parse and gate the manifest. No dataset rows are read here."""
    path = resolve_manifest_path(manifest_path)
    raw = _load_json(path, "dataset manifest")
    if not isinstance(raw, dict):
        raise BuiltinConfigError("dataset manifest must be a JSON object")
    try:
        manifest = DatasetManifest.model_validate(
            raw, context={"research_only": allow_research_only}
        )
    except ValueError as exc:
        raise BuiltinConfigError(f"invalid dataset manifest: {exc}") from None
    base = path.parent
    roots = [base]
    for root in manifest.allowed_roots:
        rp = Path(root)
        if not rp.is_absolute() or ".." in rp.parts or not rp.is_dir():
            raise BuiltinConfigError(
                f"allowed_roots entry {root!r} must be an existing absolute directory without '..'"
            )
        roots.append(rp.resolve())
    approved: dict[str, ApprovedDataset] = {}
    for dataset_id, spec in sorted(manifest.datasets.items()):
        if not spec.approved:
            continue
        data_path = _confined_path(
            spec.path, base, roots, f"dataset {dataset_id}"
        )
        if spec.source_card_path is not None:
            card_path = _confined_path(
                spec.source_card_path,
                base,
                roots,
                f"source card of {dataset_id}",
            )
            card_raw = _load_json(card_path, f"source card of {dataset_id}")
        else:
            card_raw = spec.source_card
        try:
            card = SourceCard.model_validate(card_raw)
        except ValueError as exc:
            raise BuiltinConfigError(
                f"source card of {dataset_id} is invalid: {exc}"
            ) from None
        check_source_gate(spec, card, allow_research_only=allow_research_only)
        card_hash = card.digest()
        approved[dataset_id] = ApprovedDataset(
            dataset_id=dataset_id,
            spec=spec,
            path=data_path,
            source_card=card,
            source_card_hash=card_hash,
            descriptor_hash=canonical_digest(
                {
                    "dataset_id": dataset_id,
                    "spec": spec.model_dump(mode="json"),
                    "source_card_hash": card_hash,
                }
            ),
        )
    return LoadedManifest(
        path=path,
        manifest=manifest,
        manifest_hash=canonical_digest(raw),
        datasets=approved,
    )
