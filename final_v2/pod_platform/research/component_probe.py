"""Bounded, registered DESIGN evidence about reviewed recipe components.

These retrospective measurements never produce admissions, fitted models or forecasts.
The immutable registration and one counted trial precede any dataset row access.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field, field_validator, model_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from .builtins import stats
from .builtins.catalog import LABEL, RECIPES
from .builtins.engine import (
    feature_values,
    physical,
    target_resolve_at,
    target_values,
)
from .builtins.errors import (
    BuiltinConfigError,
    DataIntegrityError,
    PartitionDenied,
    VariantBudgetExhausted,
)
from .builtins.loader import (
    MISSING_TOKENS,
    Panel,
    _iter_csv,
    _iter_parquet,
    _normalise_security,
    _parse_float,
    _parse_text,
    parse_date,
)
from .builtins.manifest import IDENT, PROTECTED_FROM, load_manifest
from .builtins.timeutil import end_of_day, is_month_end, month_index


class ProbeBudgets(StrictModel):
    max_trials: int = Field(default=3, ge=1, le=12)
    max_months: int = Field(default=12, ge=1, le=36)
    max_names: int = Field(default=100, ge=10, le=500)
    max_rows: int = Field(default=5000, ge=100, le=20000)


class ComponentProbeSpec(StrictModel):
    namespace: str = Field(pattern=IDENT)
    probe_name: str = Field(pattern=IDENT)
    dataset_id: str = Field(pattern=IDENT)
    recipe_id: str = Field(pattern=IDENT)
    feature_names: list[str] = Field(min_length=1, max_length=3)
    component_kind: Literal["feature_target", "baseline_increment"] = (
        "baseline_increment"
    )
    start: dt.date
    end: dt.date
    research_asof: dt.datetime
    evidence_refs: list[str] = Field(min_length=1, max_length=16)
    budgets: ProbeBudgets = Field(default_factory=ProbeBudgets)
    partition: Literal["design"] = "design"

    @field_validator("research_asof")
    @classmethod
    def _asof(cls, value):
        return require_aware(value, "research_asof")

    @model_validator(mode="after")
    def _bounds(self):
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if any(not ref.strip() for ref in self.evidence_refs):
            raise ValueError("evidence_refs must be nonempty")
        if self.end < self.start or self.end >= PROTECTED_FROM:
            raise ValueError(
                "probe window must lie within DESIGN, before protected check/replay dates"
            )
        if (
            month_index(self.end) - month_index(self.start) + 1
            > self.budgets.max_months
        ):
            raise ValueError(
                "probe window exceeds max_months, including feature history and target observations"
            )
        if end_of_day(self.end) > self.research_asof:
            raise ValueError("probe window is not mature at research_asof")
        return self


_SCHEMA = """
CREATE TABLE IF NOT EXISTS namespaces(namespace TEXT PRIMARY KEY, max_trials INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS registrations(probe_id TEXT PRIMARY KEY, namespace TEXT NOT NULL,
    probe_name TEXT NOT NULL, registration_json TEXT NOT NULL, registration_hash TEXT NOT NULL,
    created_at TEXT NOT NULL, UNIQUE(namespace,probe_name));
CREATE TABLE IF NOT EXISTS trials(probe_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, counted_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS results(probe_id TEXT PRIMARY KEY, result_json TEXT NOT NULL,
    result_hash TEXT NOT NULL, completed_at TEXT NOT NULL);
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ComponentProbeService:
    def __init__(
        self,
        manifest_path,
        state_dir,
        clock: Callable[[], dt.datetime] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))
        self.path = self.state_dir / "component_probes.sqlite"
        with self._db() as conn:
            conn.executescript(_SCHEMA)
            for table in ("namespaces", "registrations", "trials", "results"):
                for op in ("UPDATE", "DELETE"):
                    conn.execute(
                        f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()} BEFORE {op} ON {table} "
                        "BEGIN SELECT RAISE(ABORT, 'component evidence is append-only'); END"
                    )

    def _db(self):
        return sqlite3.connect(self.path, timeout=30)

    def _now(self):
        return require_aware(self.clock(), "clock")

    def plan(self, spec: ComponentProbeSpec | dict) -> dict:
        """Validate metadata and compute the smallest reviewed measurement; never read dataset rows."""
        spec = ComponentProbeSpec.model_validate(spec)
        if spec.research_asof > self._now():
            raise BuiltinConfigError(
                "research_asof cannot be later than the service clock"
            )
        manifest = load_manifest(self.manifest_path, allow_research_only=True)
        dataset = manifest.dataset(spec.dataset_id)
        ds = dataset.spec
        if not ds.sha256:
            raise BuiltinConfigError(
                "component probes require an operator-declared dataset sha256"
            )
        if (
            spec.start < dataset.design_window[0]
            or spec.end > dataset.design_window[1]
        ):
            raise PartitionDenied(
                "probe bounds must be contained in the operator DESIGN partition"
            )
        recipe = RECIPES.get(spec.recipe_id)
        if recipe is None or (
            ds.approved_recipes and spec.recipe_id not in ds.approved_recipes
        ):
            raise BuiltinConfigError(
                "recipe is not reviewed or not approved for this dataset"
            )
        allowlist = {
            f.name: f for _, features in recipe.feature_sets for f in features
        }
        if set(spec.feature_names) - allowlist.keys():
            raise BuiltinConfigError(
                "feature_names must select 1..3 existing reviewed recipe features"
            )
        features = [allowlist[n] for n in spec.feature_names] + [
            recipe.baseline
        ]
        logical = {c for f in features for c in (f.column, f.column2) if c}
        cols = {physical(c, ds.column_aliases) for c in logical}
        if recipe.target.column == LABEL and ds.label is None:
            raise BuiltinConfigError(
                "selected component intrinsically needs a declared return label; mechanism UNTESTED"
            )
        target_col = (
            ds.label.column
            if recipe.target.column == LABEL
            else physical(recipe.target.column, ds.column_aliases)
        )
        cols.add(target_col)
        if recipe.target.demean == "sector" or any(
            f.op == "sector_rank" for f in features
        ):
            if not ds.sector_column:
                raise BuiltinConfigError(
                    "selected component requires the declared sector column"
                )
            cols.add(ds.sector_column)
        if ds.availability.column:
            cols.add(ds.availability.column)
        missing = cols - ds.columns.keys()
        if missing:
            raise BuiltinConfigError(
                f"native component measurement requires undeclared columns: {sorted(missing)}; mechanism UNTESTED"
            )
        for c in cols:
            if c != target_col and ds.columns[c].role == "label":
                raise BuiltinConfigError(
                    "a label column cannot be used as a component feature"
                )
        probe_id = (
            "probe_"
            + canonical_digest(
                {"namespace": spec.namespace, "probe_name": spec.probe_name}
            )[:24]
        )
        return {
            "probe_id": probe_id,
            "scope": "research_only",
            "promotable": False,
            "partition": "design",
            "dataset_id": spec.dataset_id,
            "recipe_id": spec.recipe_id,
            "feature_names": spec.feature_names,
            "required_columns": sorted(cols),
            "native_target_column": target_col,
            "baseline": recipe.baseline.name,
            "original_mechanism": "UNTESTED beyond the selected native-target component",
            "omitted_financial_checks": [
                "stock_return_bridge",
                "transaction_costs",
                "borrow",
                "exposures",
                "portfolio_profitability",
            ],
            "optional_additional_controls": sorted(
                set(
                    ds.exposure_columns
                    + ([ds.label.column] if ds.label else [])
                )
                - cols
            ),
            "bounds": {
                "start": str(spec.start),
                "end": str(spec.end),
                **spec.budgets.model_dump(),
            },
            "manifest_hash": manifest.manifest_hash,
            "dataset_hash": dataset.descriptor_hash,
            "file_sha256": ds.sha256,
            "source_card_hash": dataset.source_card_hash,
            "recipe_hash": recipe.digest(),
            "fixture": dataset.fixture,
            "implementation_hash": canonical_digest(
                {
                    str(p.relative_to(Path(__file__).parent)): _sha256(p)
                    for p in [
                        Path(__file__),
                        *sorted(
                            (Path(__file__).parent / "builtins").glob("*.py")
                        ),
                    ]
                }
            ),
        }

    def register(self, spec: ComponentProbeSpec | dict) -> dict:
        spec = ComponentProbeSpec.model_validate(spec)
        plan = self.plan(spec)
        frozen = {"spec": spec.model_dump(mode="json"), "plan": plan}
        digest = canonical_digest(frozen)
        with self._db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            previous = conn.execute(
                "SELECT registration_hash FROM registrations WHERE probe_id=?",
                (plan["probe_id"],),
            ).fetchone()
            if previous:
                if previous[0] != digest:
                    raise BuiltinConfigError(
                        "registered component probe specifications and budgets are immutable"
                    )
                return {**plan, "registration_hash": digest}
            ns = conn.execute(
                "SELECT max_trials FROM namespaces WHERE namespace=?",
                (spec.namespace,),
            ).fetchone()
            if ns and ns[0] != spec.budgets.max_trials:
                raise VariantBudgetExhausted(
                    "namespace trial budget was frozen by its first probe registration"
                )
            if not ns:
                conn.execute(
                    "INSERT INTO namespaces VALUES (?,?)",
                    (spec.namespace, spec.budgets.max_trials),
                )
            conn.execute(
                "INSERT INTO registrations VALUES (?,?,?,?,?,?)",
                (
                    plan["probe_id"],
                    spec.namespace,
                    spec.probe_name,
                    json.dumps(frozen, sort_keys=True),
                    digest,
                    self._now().isoformat(),
                ),
            )
        return {**plan, "registration_hash": digest}

    def run(self, probe_id: str) -> dict:
        """Reserve exactly one counted trial, then measure the frozen component or retain its failure."""
        # Process-wide file locks also prevent two service instances treating a live trial as interrupted.
        lock_name = hashlib.sha256(probe_id.encode()).hexdigest() + ".lock"
        with (self.state_dir / lock_name).open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._run_locked(probe_id)

    def _run_locked(self, probe_id: str) -> dict:
        with self._db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT registration_json,registration_hash,namespace FROM registrations WHERE probe_id=?",
                (probe_id,),
            ).fetchone()
            if not row:
                raise BuiltinConfigError(
                    "probe must be durably registered before execution"
                )
            frozen, digest, namespace = json.loads(row[0]), row[1], row[2]
            if canonical_digest(frozen) != digest:
                raise DataIntegrityError(
                    "registered component specification hash mismatch"
                )
            cached = conn.execute(
                "SELECT result_json,result_hash FROM results WHERE probe_id=?",
                (probe_id,),
            ).fetchone()
            if cached:
                report = json.loads(cached[0])
                if (
                    report.pop("result_hash", None) != cached[1]
                    or canonical_digest(report) != cached[1]
                ):
                    raise DataIntegrityError(
                        "stored component result hash mismatch"
                    )
                return {**report, "result_hash": cached[1]}
            interrupted = (
                conn.execute(
                    "SELECT 1 FROM trials WHERE probe_id=?", (probe_id,)
                ).fetchone()
                is not None
            )
            if not interrupted:
                budget = conn.execute(
                    "SELECT max_trials FROM namespaces WHERE namespace=?",
                    (namespace,),
                ).fetchone()[0]
                used = conn.execute(
                    "SELECT COUNT(*) FROM trials WHERE namespace=?",
                    (namespace,),
                ).fetchone()[0]
                if used >= budget:
                    raise VariantBudgetExhausted(
                        "registered namespace component trial budget exhausted"
                    )
                conn.execute(
                    "INSERT INTO trials VALUES (?,?,?)",
                    (probe_id, namespace, self._now().isoformat()),
                )
        # The transaction above has committed both registration and reservation before any data read.
        plan = frozen["plan"]
        report = {
            "probe_id": probe_id,
            "scope": "research_only",
            "promotable": False,
            "partition": "design",
            "registration_hash": digest,
            "omitted_financial_checks": plan["omitted_financial_checks"],
            "original_mechanism": plan["original_mechanism"],
            "fixture": plan["fixture"],
        }
        try:
            if interrupted:
                raise RuntimeError(
                    "interrupted counted probe retained as failed; use a new budgeted probe to retry"
                )
            spec = ComponentProbeSpec.model_validate(frozen["spec"])
            if self.plan(spec) != plan:
                raise DataIntegrityError(
                    "manifest, source approval, reviewed recipe or bounds changed after registration"
                )
            dataset = load_manifest(
                self.manifest_path, allow_research_only=True
            ).dataset(spec.dataset_id)
            if _sha256(dataset.path) != plan["file_sha256"]:
                raise DataIntegrityError(
                    "dataset hash changed after registration"
                )
            panel = self._load_bounded(
                dataset, spec, set(plan["required_columns"])
            )
            report.update(self._measure(panel, dataset, spec))
            if _sha256(dataset.path) != plan["file_sha256"]:
                raise DataIntegrityError(
                    "dataset changed during component measurement"
                )
        except Exception as exc:
            report.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc)[:500],
            )
        report["limitations"] = [
            "Retrospective DESIGN component evidence; no independent holdout or full-paper replication.",
            "No fitted model, forecasts, admission or claim of trading profitability.",
            "First encountered names are sampled without observing outcomes; missing values are excluded.",
        ]
        result_hash = canonical_digest(report)
        report["result_hash"] = result_hash
        with self._db() as conn:
            conn.execute(
                "INSERT INTO results VALUES (?,?,?,?)",
                (
                    probe_id,
                    json.dumps(report, sort_keys=True, allow_nan=False),
                    result_hash,
                    self._now().isoformat(),
                ),
            )
        return report

    @staticmethod
    def _load_bounded(dataset, spec, columns):
        """Use the strict Panel parser with pre-value window/name/row limits and projected columns."""
        ds = dataset.spec
        names = [ds.security_column, ds.time_column] + sorted(columns)
        iterator = (
            _iter_csv(dataset.path, names)
            if ds.format == "csv"
            else _iter_parquet(dataset.path, names, ds.time_column, spec.end)
        )
        panel = Panel(dataset_id=dataset.dataset_id)
        for name in columns:
            (panel.num if ds.columns[name].dtype == "float" else panel.text)[
                name
            ] = []
        selected_names: set[str] = set()
        seen: set[tuple[str, int]] = set()
        for line, rec in iterator:
            where = f"row {line}"
            day = parse_date(rec[ds.time_column], where)
            if day < spec.start or day > spec.end:
                continue
            if not is_month_end(day):
                raise DataIntegrityError(
                    f"{where}: expected monthly month-end identity"
                )
            security = _normalise_security(rec[ds.security_column], where)
            if security not in selected_names:
                if len(selected_names) >= spec.budgets.max_names:
                    continue
                selected_names.add(security)
            if len(panel) >= spec.budgets.max_rows:
                raise DataIntegrityError(
                    "component panel exceeds frozen max_rows"
                )
            month = month_index(day)
            if (security, month) in seen:
                raise DataIntegrityError(
                    f"duplicate identity {security} at {day}"
                )
            seen.add((security, month))
            nums, texts = {}, {}
            for name in columns:
                cs = ds.columns[name]
                if cs.dtype == "float":
                    nums[name] = _parse_float(rec[name], name, cs, where)
                elif cs.dtype == "str":
                    texts[name] = _parse_text(rec[name], name, cs, where)
                else:
                    raw = rec[name]
                    if raw is None or (
                        isinstance(raw, str)
                        and raw.strip().lower() in MISSING_TOKENS
                    ):
                        if not cs.allow_missing:
                            raise DataIntegrityError(
                                f"{where}: missing required date {name}"
                            )
                        texts[name] = None
                    else:
                        texts[name] = parse_date(raw, where).isoformat()
            if ds.availability.column:
                avail = texts.get(ds.availability.column)
                if avail is None or dt.date.fromisoformat(avail) > day:
                    if ds.availability.on_violation == "error":
                        raise DataIntegrityError(
                            "component features unavailable at their formation date"
                        )
                    continue
            r = len(panel)
            panel.security.append(security)
            panel.month.append(month)
            panel.date.append(day)
            panel.rows_by_month.setdefault(month, []).append(r)
            panel.row_index[(security, month)] = r
            for name, value in nums.items():
                panel.num[name].append(value)
            for name, value in texts.items():
                panel.text[name].append(value)
        return panel

    @staticmethod
    def _measure(panel, dataset, spec):
        recipe = RECIPES[spec.recipe_id]
        allowed = {f.name: f for _, fs in recipe.feature_sets for f in fs}
        monthly = []
        for month in sorted(panel.rows_by_month):
            if target_resolve_at(recipe, month, dataset) > spec.research_asof:
                continue
            feats = [
                feature_values(panel, allowed[n], month, dataset)
                for n in spec.feature_names
            ]
            target = target_values(panel, recipe, month, dataset)
            base = feature_values(panel, recipe.baseline, month, dataset)
            rows = sorted(
                set(target).intersection(base, *(set(f) for f in feats))
            )
            if len(rows) < 10:
                continue
            prediction = [
                math.fsum(f[r] for f in feats) / len(feats) for r in rows
            ]
            actual = [target[r] for r in rows]
            ic = stats.spearman(prediction, actual)
            baseline = stats.spearman([base[r] for r in rows], actual)
            if ic is not None and baseline is not None:
                monthly.append(
                    {
                        "month": month,
                        "pairs": len(rows),
                        "feature_target_ic": ic,
                        "baseline_target_ic": baseline,
                        "incremental_ic": ic - baseline,
                    }
                )
        primary = (
            "incremental_ic"
            if spec.component_kind == "baseline_increment"
            else "feature_target_ic"
        )
        delta = [m[primary] for m in monthly]
        avg, se = (
            stats.mean(delta),
            stats.hac_se(delta, recipe.target.months - 1),
        )
        status = "inconclusive"
        if len(monthly) >= 3 and avg is not None:
            status = "noisy"
            if se is not None:
                if avg + 1.96 * se < 0:
                    status = "negative"
                elif avg - 1.96 * se > 0:
                    status = "positive_signal"
        return {
            "status": status,
            "primary_metric": primary,
            "metrics": {
                "mean_feature_target_ic": stats.mean(
                    [m["feature_target_ic"] for m in monthly]
                ),
                "mean_baseline_target_ic": stats.mean(
                    [m["baseline_target_ic"] for m in monthly]
                ),
                "mean_incremental_ic": stats.mean(
                    [m["incremental_ic"] for m in monthly]
                ),
                "incremental_ic_standard_error": stats.hac_se(
                    [m["incremental_ic"] for m in monthly],
                    recipe.target.months - 1,
                ),
                "primary_mean": avg,
                "primary_standard_error": se,
            },
            "evaluated_months": len(monthly),
            "evaluated_pairs": sum(m["pairs"] for m in monthly),
            "loaded_rows": len(panel),
            "loaded_names": len(set(panel.security)),
            "monthly": monthly,
        }
