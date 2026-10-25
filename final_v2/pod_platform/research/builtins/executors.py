"""Registered built-in executors: one per charter, running only reviewed recipes on operator-approved datasets.

``register_executors`` is the integration point for ``ExecutorRegistry``. The executor callable receives an
``ExperimentRequest`` whose scalar variant params may only name an approved dataset and a reviewed recipe /
grid point. It never receives, evaluates or imports caller-supplied code, paths or labels.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ...contracts.base import canonical_digest, require_aware
from ...pods.runner import ExperimentResult
from ...strategies.base import Charter
from ..partitions import DESIGN
from .catalog import (
    ALLOWED_PARAMS,
    DEFAULT_RECIPE,
    METHODS,
    RECIPES,
    RECIPES_BY_CHARTER,
    RIDGE_LAMBDAS,
    Recipe,
    VariantConfig,
    parse_variant_params,
)
from .engine import ENGINE_VERSION, EVALUATION_LABEL, evaluate, required_columns
from .errors import (
    BuiltinConfigError,
    DataIntegrityError,
    EvidenceInsufficient,
    PartitionDenied,
)
from .ledger import BuiltinLedger
from .loader import file_sha256, load_panel
from .manifest import (
    ENV_MANIFEST,
    ApprovedDataset,
    LoadedManifest,
    load_manifest,
)
from .timeutil import end_of_day

ENV_STATE_DIR = "POD_BUILTIN_STATE_DIR"
MAX_VARIANT_BUDGET = 64
REPORT_SCHEMA = "builtin_eval.v1"

EXECUTOR_IDS: dict[str, str] = {
    Charter.FUNDAMENTALS_EXPECTATIONS.value: "builtin_fundamentals_expectations_v1",
    Charter.CORPORATE_EVENTS.value: "builtin_corporate_events_v1",
    Charter.VALUATION_RELATIVE_VALUE.value: "builtin_valuation_relative_value_v1",
    Charter.MARKET_DYNAMICS.value: "builtin_market_dynamics_v1",
}
CHARTER_BY_EXECUTOR: dict[str, str] = {v: k for k, v in EXECUTOR_IDS.items()}

_EVIDENCE_ERRORS = {
    "EvidenceInsufficient": EvidenceInsufficient,
    "DataIntegrityError": DataIntegrityError,
}


def _implementation_hash() -> str:
    """Bind cached trials to the actual reviewed implementation, not a version alias."""
    directory = Path(__file__).resolve().parent
    return canonical_digest(
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.py"))
        }
    )


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class BuiltinContext:
    """Manifest + ledger + clock shared by the four executors (and by the charter campaign)."""

    def __init__(
        self,
        manifest_path: "str | Path | None" = None,
        state_dir: "str | Path | None" = None,
        *,
        clock: Optional[Callable[[], dt.datetime]] = None,
        research_asof: Optional[dt.datetime] = None,
        variant_budget: int = 8,
    ):
        if not 1 <= int(variant_budget) <= MAX_VARIANT_BUDGET:
            raise BuiltinConfigError(
                f"variant_budget must be in [1, {MAX_VARIANT_BUDGET}]"
            )
        self.manifest: LoadedManifest = load_manifest(manifest_path)
        self.clock = clock or _utc_now
        self._research_asof = (
            require_aware(research_asof, "research_asof")
            if research_asof is not None
            else None
        )
        self.variant_budget = int(variant_budget)
        state = state_dir if state_dir else os.environ.get(ENV_STATE_DIR)
        self.ledger = BuiltinLedger(state, clock=self.clock)

    # -- clocks ------------------------------------------------------------

    def research_asof(self) -> dt.datetime:
        asof = (
            self._research_asof
            if self._research_asof is not None
            else self.clock()
        )
        asof = require_aware(asof, "research_asof")
        if asof > require_aware(self.clock(), "clock"):
            raise EvidenceInsufficient(
                "research_asof lies in the future of the executor clock"
            )
        return asof

    # -- resolution --------------------------------------------------------

    def prepare(
        self, executor_id: str, params: Mapping[str, Any]
    ) -> tuple[VariantConfig, ApprovedDataset, Recipe, set[str], dict, str]:
        charter = CHARTER_BY_EXECUTOR.get(executor_id)
        if charter is None:
            raise BuiltinConfigError(
                f"{executor_id!r} is not a built-in executor"
            )
        cfg = parse_variant_params(params, charter)
        if cfg.partition != DESIGN:
            raise PartitionDenied(
                f"PARTITION_ACCESS_DENIED: partition {cfg.partition!r} is not available to pod executors"
            )
        dataset = self.manifest.dataset(cfg.dataset_id)
        cfg = VariantConfig(
            dataset.dataset_id,
            cfg.recipe_id,
            cfg.method_id,
            cfg.feature_set,
            cfg.ridge_lambda,
            cfg.partition,
        )
        recipe = RECIPES[cfg.recipe_id]
        approved = dataset.spec.approved_recipes
        if approved and cfg.recipe_id not in approved:
            raise BuiltinConfigError(
                f"recipe {cfg.recipe_id!r} is not approved for dataset {dataset.dataset_id!r}"
            )
        columns = required_columns(recipe, dataset)
        sha = file_sha256(dataset.path)
        if dataset.spec.sha256 is not None and dataset.spec.sha256 != sha:
            raise BuiltinConfigError(
                "dataset file hash does not match the sha256 declared in the operator manifest"
            )
        start, end = dataset.design_window
        spec = {
            "engine_version": ENGINE_VERSION,
            "implementation_hash": _implementation_hash(),
            "executor_id": executor_id,
            "variant": cfg.as_dict(),
            "recipe_digest": recipe.digest(),
            "dataset_descriptor_hash": dataset.descriptor_hash,
            "data_file_sha256": sha,
            "source_card_hash": dataset.source_card_hash,
            "source_snapshot_id": dataset.spec.source_snapshot_id,
            "partition": {
                "name": DESIGN,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
            "columns": sorted(columns),
        }
        return cfg, dataset, recipe, columns, spec, canonical_digest(spec)

    # -- execution ---------------------------------------------------------

    def execute(self, executor_id: str, request: Any):

        cfg, dataset, recipe, columns, spec, spec_hash = self.prepare(
            executor_id, dict(request.variant.params)
        )
        registration_id = f"{request.namespace}:{request.draft_entry_id}"
        variant_name = request.variant.name
        asof = self.research_asof()
        if (
            end_of_day(dataset.design_window[1]) > asof
        ):  # clock-dependent, so it must not consume a budget slot
            raise EvidenceInsufficient(
                f"LABEL_NOT_MATURE: design window ends {dataset.design_window[1].isoformat()} after research_asof {asof.isoformat()}"
            )
        admission = self.ledger.admit(
            registration_id, variant_name, spec_hash, budget=self.variant_budget
        )
        if admission.state == "cached":
            return ExperimentResult.model_validate(admission.result)
        if admission.state == "failed":
            error_cls = _EVIDENCE_ERRORS.get(
                str(admission.failure.get("type")), EvidenceInsufficient
            )
            raise error_cls(str(admission.failure.get("message")))
        try:
            panel = load_panel(
                dataset, columns, partition=DESIGN, research_asof=asof
            )
            self.ledger.record_access(
                registration_id,
                dataset.dataset_id,
                DESIGN,
                f"executor:{executor_id}",
                len(panel),
            )
            core = evaluate(panel, dataset, recipe, cfg, research_asof=asof)
        except (EvidenceInsufficient, DataIntegrityError) as exc:
            self.ledger.record_evidence_failure(
                registration_id,
                variant_name,
                spec_hash,
                {"type": type(exc).__name__, "message": str(exc)[:600]},
            )
            raise
        except Exception as exc:
            self.ledger.record_infrastructure_failure(
                registration_id, spec_hash, f"{type(exc).__name__}: {exc}"
            )
            raise
        report = self._report(
            executor_id,
            registration_id,
            variant_name,
            spec,
            spec_hash,
            cfg,
            dataset,
            recipe,
            asof,
            panel.meta,
            core,
        )
        report["approvals"] = {
            "manifest_id": self.manifest.manifest.manifest_id,
            "approved_by": self.manifest.manifest.approved_by,
            "manifest_hash": self.manifest.manifest_hash,
            "dataset_approved": dataset.spec.approved,
        }
        report["candidate_artifact"] = self._persist_candidate(
            registration_id, variant_name, spec_hash, spec, cfg, core
        )
        report["report_hash"] = canonical_digest(report)
        result = ExperimentResult(
            registration_id=registration_id,
            spec_hash=spec_hash,
            executor_id=executor_id,
            variant_name=variant_name,
            data_partition=DESIGN,
            metrics=core["metrics"],
            notes=self._notes(report, cfg, dataset, recipe),
        )
        self.ledger.record_success(
            registration_id,
            variant_name,
            spec_hash,
            result.model_dump(mode="json"),
            report,
            fits=int(core["metrics"].get("trial_fold_evaluations", 0)),
            ridge=cfg.method_id == "ridge",
        )
        return result

    def candidate_path(self, spec_hash: str) -> Optional[Path]:
        return (
            None
            if self.ledger.path is None
            else self.ledger.path.parent / "candidates" / f"{spec_hash}.json"
        )

    def _persist_candidate(
        self, registration_id, variant_name, spec_hash, spec, cfg, core
    ) -> Optional[dict]:
        """Write the immutable candidate provenance (spec, frozen design-trained model, exact design predictions).

        Needs a persistent ``state_dir``; the report carries the artifact digest so the registered ledger report binds it.
        """
        path = self.candidate_path(spec_hash)
        if path is None:
            return None
        model, rows = core["frozen_model"], core["predictions"]
        artifact = {
            "schema_version": "builtin_candidate.v1",
            "registration_id": registration_id,
            "variant_name": variant_name,
            "spec_hash": spec_hash,
            "spec": spec,
            "params": cfg.as_dict(),
            "frozen_model": model,
            "model_hash": canonical_digest(model),
            "predictions": rows,
            "predictions_hash": canonical_digest(rows),
        }
        text = json.dumps(artifact, sort_keys=True, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise DataIntegrityError(
                "candidate artifact exists with different content; refusing to overwrite immutable provenance"
            )
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        return {
            "file": path.name,
            "artifact_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "model_hash": artifact["model_hash"],
            "predictions_hash": artifact["predictions_hash"],
            "prediction_rows": len(rows),
        }

    @staticmethod
    def _report(
        executor_id,
        registration_id,
        variant_name,
        spec,
        spec_hash,
        cfg,
        dataset,
        recipe,
        asof,
        meta,
        core,
    ) -> dict:
        ds = dataset.spec
        limitations = list(recipe.limitations) + [
            "Design partition evaluated retrospectively; the 2015-2018 window was inspected by the team before, so this is not a fresh holdout.",
            "Execution lag (formation at the close after the information date) is not modelled: the panel label is used as supplied (PROXY).",
            f"Costs are a flat operator-declared {ds.evaluation.cost_bps_one_way} bps one way on turnover; borrow, impact, capacity and funding are not modelled.",
            "Securities without a later panel row (delisting/gaps) drop out of lead targets (survivorship in the target); missing returns are not imputed.",
            "Standard errors are descriptive (month clustering, Newey-West lag = horizon-1); no selection or multiple-testing adjustment inside a single result.",
            "No admission certificate can result from a design-partition evaluation; Research Integrity owns the check window.",
        ]
        if dataset.fixture:
            limitations.insert(
                0,
                "FIXTURE DATA: synthetic; results demonstrate mechanics only and authorize nothing about any real book.",
            )
        if ds.availability.mode == "known_at_time_column":
            limitations.append(
                "Feature availability rests on the operator attestation: "
                + ds.availability.attestation
            )
        return {
            "schema_version": REPORT_SCHEMA,
            "engine_version": ENGINE_VERSION,
            "executor_id": executor_id,
            "registration_id": registration_id,
            "variant_name": variant_name,
            "params": cfg.as_dict(),
            "spec": spec,
            "spec_hash": spec_hash,
            "recipe_id": recipe.recipe_id,
            "charter": recipe.charter,
            "dataset_id": dataset.dataset_id,
            "data_class": ds.data_class,
            "fixture": dataset.fixture,
            "source_id": ds.source_id,
            "source_snapshot_id": ds.source_snapshot_id,
            "source_card_hash": dataset.source_card_hash,
            "partition": DESIGN,
            "partition_window": [d.isoformat() for d in dataset.design_window],
            "evaluation_label": EVALUATION_LABEL,
            "research_asof": asof.isoformat(),
            "data": meta,
            "metrics": core["metrics"],
            "monthly": core["monthly"],
            "diagnostics": core["diagnostics"],
            "labels": {
                "belief_target": recipe.target.target_id,
                "belief_target_description": recipe.target.description,
                "baseline": recipe.baseline_description,
                "bridge": "next-month excess return of the dataset label column; separate from belief quality",
                "declared_direction": recipe.declared_direction,
                "proxy_labels": list(recipe.proxy_labels),
                "mechanism_transfer": [
                    p
                    for p in recipe.proxy_labels
                    if p.startswith("MECHANISM_TRANSFER")
                ],
            },
            "limitations": limitations,
            "trial_accounting": {
                "scored_variants": 1,
                "hyperparameter_configs": 1 if cfg.method_id == "ridge" else 0,
                "fold_evaluations": int(
                    core["metrics"].get("trial_fold_evaluations", 0)
                ),
                "design_partition_accesses": 1,
            },
        }

    @staticmethod
    def _notes(
        report: dict,
        cfg: VariantConfig,
        dataset: ApprovedDataset,
        recipe: Recipe,
    ) -> str:
        parts = [
            f"evaluation_label={EVALUATION_LABEL}",
            f"data_class={dataset.spec.data_class}",
            f"recipe={recipe.recipe_id}",
            f"method={cfg.method_id}/{cfg.feature_set}"
            + (
                f"/lambda={cfg.ridge_lambda}"
                if cfg.ridge_lambda is not None
                else ""
            ),
            f"report_hash={report['report_hash'][:16]}",
            "belief and return bridge are separate (belief_* vs bridge_*); no admission",
            "proxies: " + " | ".join(p[:110] for p in recipe.proxy_labels),
        ]
        return "; ".join(parts)[:1500]

    # -- convenience -------------------------------------------------------

    def register(self, registry: Any) -> None:
        """Register the four reviewed executors (stable ids) on an ``ExecutorRegistry``."""

        def make(executor_id: str):
            def run(request: Any):
                return self.execute(executor_id, request)

            run.__name__ = executor_id
            return run

        for executor_id in EXECUTOR_IDS.values():
            registry.register(executor_id, make(executor_id))

    def report(self, registration_id: str, variant_name: str) -> Optional[dict]:
        return self.ledger.report(registration_id, variant_name)

    def catalog(self) -> dict:
        return catalog_summary_for(self.manifest)

    def close(self) -> None:
        self.ledger.close()


def catalog_summary_for(manifest: Optional[LoadedManifest]) -> dict:
    out: dict[str, Any] = {
        "allowed_variant_params": sorted(ALLOWED_PARAMS),
        "methods": list(METHODS),
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "executors": {},
        "datasets": {},
    }
    for charter, executor_id in EXECUTOR_IDS.items():
        out["executors"][executor_id] = {
            "charter": charter,
            "default_recipe": DEFAULT_RECIPE[charter],
            "recipes": {
                r.recipe_id: {
                    "summary": r.summary,
                    "target": r.target.target_id,
                    "baseline": r.baseline_description,
                    "feature_sets": list(r.feature_set_names()),
                    "proxy_labels": list(r.proxy_labels),
                }
                for r in RECIPES_BY_CHARTER[charter]
            },
        }
    if manifest is not None:
        for dataset_id, ds in sorted(manifest.datasets.items()):
            out["datasets"][dataset_id] = {
                "data_class": ds.spec.data_class,
                "source_id": ds.spec.source_id,
                "source_snapshot_id": ds.spec.source_snapshot_id,
                "design_window": [d.isoformat() for d in ds.design_window],
                "approved_recipes": list(ds.spec.approved_recipes),
            }
    return out


def catalog_summary(manifest_path: "str | Path | None" = None) -> dict:
    """Safe description (ids only, no paths) that a pod prompt may show to an LLM."""
    return catalog_summary_for(
        load_manifest(manifest_path)
        if (manifest_path or os.environ.get(ENV_MANIFEST))
        else None
    )


def register_executors(
    registry: Any,
    *,
    manifest_path: "str | Path | None" = None,
    state_dir: "str | Path | None" = None,
    clock: Optional[Callable[[], dt.datetime]] = None,
    research_asof: Optional[dt.datetime] = None,
    variant_budget: int = 8,
) -> BuiltinContext:
    """Register the four reviewed built-in executors on an ``ExecutorRegistry``.

    ``manifest_path`` defaults to the ``POD_DATASET_MANIFEST`` environment variable; private files are never
    discovered automatically. ``state_dir`` defaults to ``POD_BUILTIN_STATE_DIR``, else an in-memory ledger.
    """
    context = BuiltinContext(
        manifest_path,
        state_dir,
        clock=clock,
        research_asof=research_asof,
        variant_budget=variant_budget,
    )
    context.register(registry)
    return context
