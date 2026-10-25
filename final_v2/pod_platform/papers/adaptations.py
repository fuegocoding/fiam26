"""Paper -> reviewed adaptation -> registered hypothesis/experiment -> bounded executor -> measured result (spec 10.2, 12, 23.4).

No model is called and nothing is admitted. A human-reviewed ``AdaptationReview`` selects ids only: a cached paper
version, grounded claim ids, an allowlisted builtin recipe (bound by digest), an operator-approved dataset and a
reviewed method grid point. ``adapt`` re-derives every fact from the paper store (full text, claim quotes, version
date) and the dataset manifest *descriptor* (no data file is read), then registers a normal ``HypothesisSpec`` and
``ExperimentSpec`` plus the ``RegisteredExperiment`` BEFORE any execution. ``execute_design`` runs only an
already-registered executor on the design partition and refuses unregistered adaptations. Lineage
source -> method -> hypothesis -> experiment -> result is persisted append-only with hash bindings; a retry reuses
the same ids, memory entry, registration and builtin variant budget.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Callable, Literal, Optional

from pydantic import Field, ValidationError, field_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.experiments import (ExperimentAdmission, ExperimentBudget, ExperimentClaim, ExperimentComparisons, ExperimentCosts,
                                     ExperimentData, ExperimentIdentity, ExperimentModel, ExperimentPortfolio, ExperimentSample,
                                     ExperimentSpec, ExperimentStatistics, ExperimentTime)
from ..contracts.hypotheses import HypothesisSpec
from ..contracts.hypotheses import ResearchMode as HypothesisMode
from ..pods.charters import pod_spec
from ..pods.runner import ExecutorRegistry, ExperimentRequest
from ..pods.schema import ExperimentPlan, ExperimentVariant, FeatureDefinition, ResearchProposal
from ..research.builtins.catalog import RECIPES, parse_variant_params
from ..research.builtins.executors import EXECUTOR_IDS
from ..research.builtins.timeutil import end_of_day
from ..research.jobs import open_sqlite
from ..research.registration import RegisteredExperiment
from .models import PapersError
from .service import REQUIRED_FOR
from .models import ReplicationLabel

STAGES = ("source", "method", "hypothesis", "experiment", "result")
VARIANT = "adapt_v1"
_IDENT = r"^[a-z][a-z0-9_]{0,63}$"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS adaptations (adaptation_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, record_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lineage (adaptation_id TEXT NOT NULL, stage TEXT NOT NULL, ref_id TEXT NOT NULL, ref_hash TEXT NOT NULL,
    at TEXT NOT NULL, detail_json TEXT NOT NULL, PRIMARY KEY (adaptation_id, stage));
CREATE TABLE IF NOT EXISTS rejections (seq INTEGER PRIMARY KEY AUTOINCREMENT, namespace TEXT, input_hash TEXT NOT NULL, code TEXT NOT NULL,
    message TEXT NOT NULL, at TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS lineage_no_update BEFORE UPDATE ON lineage BEGIN SELECT RAISE(ABORT, 'lineage is append-only'); END;
CREATE TRIGGER IF NOT EXISTS lineage_no_delete BEFORE DELETE ON lineage BEGIN SELECT RAISE(ABORT, 'lineage is append-only'); END;
CREATE TRIGGER IF NOT EXISTS adaptations_no_update BEFORE UPDATE ON adaptations BEGIN SELECT RAISE(ABORT, 'adaptations are immutable'); END;
"""


class AdaptationRejected(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class AdaptationReview(StrictModel):
    """Everything a reviewer must state; ids select reviewed objects, free text never selects code or data."""

    namespace: str
    paper_id: str
    version_key: str
    label: Literal["CLOSE_REPLICATION", "MECHANISM_TRANSFER", "METHOD_TRANSFER"]
    claim_ids: list[str] = Field(min_length=1, max_length=20)
    deviations: list[str] = Field(min_length=1, max_length=10)
    comparison_plan: str = Field(min_length=20, max_length=800)
    baseline: str = Field(min_length=5, max_length=600)
    recipe_id: str
    recipe_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(pattern=_IDENT)
    method_id: str = Field(pattern=_IDENT)
    feature_set: str = Field(pattern=_IDENT)
    ridge_lambda: Optional[float] = None
    economic_effect: str = Field(min_length=10, max_length=600)
    declared_sign: Optional[Literal["positive", "negative"]] = None
    falsification: list[str] = Field(min_length=1, max_length=5)
    selected_by: Literal["implementation"]  # the recipe is chosen by the implementer, never by a model
    reviewed_by: str = Field(min_length=1, max_length=100)
    reviewed_at: dt.datetime
    research_mode: Literal["contemporary", "historical"] = "contemporary"
    as_of: Optional[dt.datetime] = None

    @field_validator("deviations", "falsification")
    @classmethod
    def _text(cls, value: list[str]) -> list[str]:
        if any(not 10 <= len(v.strip()) <= 600 for v in value):
            raise ValueError("each entry must be 10-600 characters")
        return value

    @field_validator("reviewed_at", "as_of")
    @classmethod
    def _aware(cls, value):
        return None if value is None else require_aware(value, "timestamp")


def _evidence_id(card) -> str:
    return re.sub(r"[^A-Za-z0-9_.:\-]", "_", f"{card.paper_id}.{card.version_key}")[:128]


class PaperAdapter:
    def __init__(self, path, *, papers, registry, ledger, memory, context,
                 clock: Callable[[], dt.datetime]):
        """``context`` is a ``BuiltinContext`` (manifest descriptors + report ledger); ``ledger`` a ``ResearchLedger``."""
        self.path, self.papers, self.registry, self.ledger = path, papers, registry, ledger
        self.memory, self.context, self._clock = memory, context, clock
        with open_sqlite(path) as conn:
            conn.executescript(_SCHEMA)

    def _now(self) -> dt.datetime:
        return require_aware(self._clock(), "clock")

    def _reject(self, namespace, raw, code: str, message: str):
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute("INSERT INTO rejections (namespace, input_hash, code, message, at) VALUES (?,?,?,?,?)",
                         (namespace, canonical_digest({"raw": raw}), code, message[:500], self._now().isoformat()))
        raise AdaptationRejected(code, message)

    # -- adapt (registration only; no data read) --------------------------------

    def adapt(self, review: "AdaptationReview | dict") -> dict:
        raw = review if isinstance(review, dict) else review.model_dump(mode="json")
        try:
            rv = review if isinstance(review, AdaptationReview) else AdaptationReview.model_validate(review)
        except ValidationError as exc:
            self._reject(raw.get("namespace") if isinstance(raw, dict) else None, raw, "incomplete_review", str(exc)[:400])
        ns, now = rv.namespace, self._now()
        try:
            pod = pod_spec(ns)
        except KeyError:
            self._reject(ns, raw, "unknown_pod", ns)
        if pod.namespace != ns or self.memory.namespace != ns:
            self._reject(ns, raw, "namespace_mismatch", "review, pod and memory namespaces must be identical")
        adaptation_id = "ad_" + canonical_digest(rv.model_dump(mode="json"))[:20]
        existing = self.get(adaptation_id)
        if existing is not None:
            return existing
        if rv.reviewed_at > now or (rv.as_of is not None and rv.as_of > now):
            self._reject(ns, raw, "future_time", "reviewed_at/as_of lie in the future of the adapter clock")
        if (rv.research_mode == "historical") != (rv.as_of is not None):
            self._reject(ns, raw, "incomplete_review", "historical mode requires as_of and contemporary mode forbids it")
        recipe = RECIPES.get(rv.recipe_id)
        if recipe is None or recipe.digest() != rv.recipe_digest:
            self._reject(ns, raw, "unreviewed_recipe", "recipe is not an allowlisted builtin or its reviewed digest changed")
        if recipe.charter != pod.charter.value:
            self._reject(ns, raw, "unreviewed_recipe", "recipe belongs to another pod's charter")
        params = {"dataset_id": rv.dataset_id, "recipe_id": rv.recipe_id, "method_id": rv.method_id, "feature_set": rv.feature_set}
        if rv.ridge_lambda is not None:
            params["ridge_lambda"] = rv.ridge_lambda
        try:
            cfg = parse_variant_params(params, pod.charter.value)
            dataset = self.context.manifest.dataset(rv.dataset_id)  # descriptor only: the data file is not opened here
        except Exception as exc:  # builtin config / source-eligibility errors are all rejections
            self._reject(ns, raw, "unresolved_source", f"{type(exc).__name__}: {str(exc)[:300]}")
        if dataset.spec.approved_recipes and rv.recipe_id not in dataset.spec.approved_recipes:
            self._reject(ns, raw, "unresolved_source", "recipe is not approved for this dataset")
        params = {k: v for k, v in {"dataset_id": cfg.dataset_id, "recipe_id": cfg.recipe_id, "method_id": cfg.method_id,
                                    "feature_set": cfg.feature_set, "ridge_lambda": cfg.ridge_lambda}.items() if v is not None}
        card, claims = self._grounded_paper(rv, raw, now)
        executor_id = EXECUTOR_IDS[pod.charter.value]
        evidence_id = _evidence_id(card)
        mechanism = ("Grounded paper claims: " + " | ".join(c.text for c in claims))[:1500]
        proposal = ResearchProposal(
            title=f"Adaptation of {card.title}"[:200], mechanism=mechanism,
            falsifiable_target=("Falsified if: " + "; ".join(rv.falsification))[:800], baseline=recipe.baseline_description[:600],
            features=[FeatureDefinition(name=f.name, definition=f"{f.op} transform of column {f.column}"
                                        + (f" with {f.column2}" if f.column2 else "") + " (reviewed builtin recipe)",
                                        timing="Known at the panel month end per the dataset manifest attestation", source_ids=[evidence_id])
                      for f in recipe.features(cfg.feature_set)],
            evidence_refs=[evidence_id],
            experiment_plan=ExperimentPlan(executor_id=executor_id, primary_metric="belief_ic_improvement", data_partition="design",
                                           variants=[ExperimentVariant(name=VARIANT, params=params)],
                                           description=f"Design-only run of reviewed recipe {rv.recipe_id} adapted from {card.paper_id}."),
            competing_explanation=("Common style exposure may explain any effect. Declared deviations: " + "; ".join(rv.deviations))[:800],
            limitations=[d[:600] for d in rv.deviations[:5]] + [x[:600] for x in recipe.limitations[:3]],
            evidence_quality="weak")
        hyp_id = "hyp_" + canonical_digest({"a": adaptation_id, "t": "h"})[:20]
        exp_id = "exp_" + canonical_digest({"a": adaptation_id, "t": "e"})[:20]
        baseline_id = f"baseline:{rv.recipe_id}:{recipe.baseline.name}"
        hypothesis = HypothesisSpec(
            hypothesis_id=hyp_id, owner=ns, mechanism_family=recipe.mechanism_family,
            research_mode=HypothesisMode.HISTORICAL_REPLAY if rv.as_of else HypothesisMode.CONTEMPORARY_RESEARCH, created_at=now,
            question=f"Does the mechanism of {card.paper_id}@{card.version_key} hold when adapted via {rv.recipe_id} on {rv.dataset_id}?",
            mechanism=mechanism, target_id=recipe.target.target_id, reference_model_id=f"baseline:{recipe.baseline.name}",
            feature_spec_ids=[f"{rv.recipe_id}/{cfg.feature_set}/{f.name}" for f in recipe.features(cfg.feature_set)],
            applicability_spec_id=f"applicability:{rv.dataset_id}:{dataset.descriptor_hash[:16]}", baseline_experiment_ids=[baseline_id],
            falsification=rv.falsification, variant_budget=1, experiment_spec_id=exp_id)
        start, end = dataset.design_window
        experiment = ExperimentSpec(
            experiment_spec_id=exp_id, owner=ns, created_at=now,
            identity=ExperimentIdentity(hypothesis_id=hyp_id, variant_id=VARIANT, owner=ns),
            claim=ExperimentClaim(primary_target_id=recipe.target.target_id, primary_metric="belief_ic_improvement",
                                  economic_effect=rv.economic_effect, declared_sign=rv.declared_sign),
            data=ExperimentData(source_snapshot_ids=[dataset.spec.source_snapshot_id], feature_spec_ids=hypothesis.feature_spec_ids,
                                identity_graph_version="panel_security_id_as_supplied"),
            sample=ExperimentSample(universe_spec_id=f"universe:{rv.dataset_id}", eligibility_spec_id=f"eligibility:{rv.dataset_id}",
                                    applicability_spec_id=hypothesis.applicability_spec_id),
            time=ExperimentTime(training_cutoff=end_of_day(end), label_maturity_rule="operator manifest label resolve_months",
                                partitions={"design": f"{start.isoformat()}..{end.isoformat()}"},
                                purge_rule="builtin engine walk-forward purge", embargo_rule="builtin engine walk-forward embargo"),
            model=ExperimentModel(method_id=f"builtin:{cfg.method_id}", hyperparameter_space=dict(params),
                                  seed_policy="deterministic, no random restarts"),
            portfolio=ExperimentPortfolio(entry_rule="belief evaluation only; return bridge is diagnostic", exit_rule="next-month label",
                                          horizon_trading_days=max(1, recipe.target.months * 21), construction_policy_id="not_applicable_design_only",
                                          hedge_policy_id="not_applicable_design_only"),
            costs=ExperimentCosts(trade_cost_model_id=f"flat_{dataset.spec.evaluation.cost_bps_one_way}bps_one_way",
                                  borrow_model_id="not_modelled", liquidity_model_id="not_modelled", funding_model_id="not_modelled"),
            comparisons=ExperimentComparisons(baseline_experiment_ids=[baseline_id], paired_comparison_policy_id=rv.recipe_id,
                                              permitted_sensitivities=[]),
            statistics=ExperimentStatistics(dependence_handling="builtin month clustering and Newey-West",
                                            uncertainty_method="descriptive standard errors",
                                            selection_accounting_policy_id="builtin variant ledger, budget 1"),
            admission=ExperimentAdmission(min_coverage="builtin min_names_per_month", effect_requirement=rv.comparison_plan[:300],
                                          stability_requirement="not assessed in design partition",
                                          implementation_requirement="not assessed in design partition"),
            budget=ExperimentBudget(compute_budget="1 builtin variant", provider_budget="none", llm_budget="none",
                                    evaluation_access_bounds="design partition only"),
            promotion_conditions="No admission from a design-partition result; Research Integrity owns any check window.",
            failure_conditions="Candidate does not beat the declared baseline or the executor reports insufficient evidence.").sealed()
        draft = self.memory.append("draft_proposal", {
            "status": "draft", "evaluated": False, "admission": "not_admitted", "pod_id": pod.pod_id, "title": proposal.title,
            "proposal": proposal.model_dump(mode="json"), "allowed_executors": [executor_id],
            "adaptation": {"adaptation_id": adaptation_id, "hypothesis_id": hyp_id, "experiment_spec_id": exp_id,
                           "paper_card_hash": card.card_hash, "review_hash": canonical_digest(rv.model_dump(mode="json"))}},
            idempotency_key=f"adapt-draft:{adaptation_id}")
        registration_id = f"{ns}:{draft.entry_id}"  # the id the builtin executor ledger uses
        if self.registry.get("hypotheses", hyp_id) is None:
            self.registry.register("hypotheses", hypothesis)
        if self.registry.get("experiment_specs", exp_id) is None:
            self.registry.register("experiment_specs", experiment)
        record = {"registered_at": now.isoformat(), **{"adaptation_id": adaptation_id, "namespace": ns, "draft_entry_id": draft.entry_id, "registration_id": registration_id,
                  "hypothesis_id": hyp_id, "experiment_spec_id": exp_id, "experiment_content_hash": experiment.content_hash,
                  "variant": VARIANT, "executor_id": executor_id, "params": params, "label": rv.label,
                  "status": "registered_not_executed", "admitted": False, "fixture_data": dataset.fixture}}
        self._ensure_ledger(record, self.registry.get("experiment_specs", exp_id))
        stages = {
            "source": (card.paper_id + "@" + card.version_key, card.card_hash, {"content_hash": card.content_hash, "claim_ids": [c.claim_id for c in claims]}),
            "method": (rv.recipe_id, rv.recipe_digest, {"method_id": cfg.method_id, "feature_set": cfg.feature_set, "dataset_id": rv.dataset_id,
                                                        "dataset_descriptor_hash": dataset.descriptor_hash, "deviations": rv.deviations}),
            "hypothesis": (hyp_id, hypothesis.sealed().content_hash, {}),
            "experiment": (exp_id, experiment.content_hash, {"registration_id": registration_id, "draft_entry_id": draft.entry_id}),
        }
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute("INSERT INTO adaptations VALUES (?,?,?,?)", (adaptation_id, ns, json.dumps(record, sort_keys=True), now.isoformat()))
            for stage, (ref_id, ref_hash, detail) in stages.items():
                conn.execute("INSERT INTO lineage VALUES (?,?,?,?,?,?)", (adaptation_id, stage, ref_id, ref_hash, now.isoformat(),
                                                                           json.dumps(detail, sort_keys=True)))
        return record

    def _grounded_paper(self, rv: AdaptationReview, raw, now):
        as_of = rv.as_of
        try:
            card = self.papers.get(rv.paper_id, as_of=as_of)
        except PapersError as exc:
            self._reject(rv.namespace, raw, "future_version" if "after as_of" in str(exc) or "as_of" in str(exc) else "unresolved_paper", str(exc)[:300])
        if card.paper_id != rv.paper_id and rv.paper_id not in (card.arxiv_id, card.doi):
            self._reject(rv.namespace, raw, "unresolved_paper", "paper id does not resolve to the stored card")
        if card.version_key != rv.version_key:
            self._reject(rv.namespace, raw, "version_mismatch", f"reviewed {rv.version_key!r} but the cache holds {card.version_key!r} at this time")
        if card.version_date is None or card.version_date > (as_of or now):
            self._reject(rv.namespace, raw, "future_version", "paper version date is unknown or after the research time")
        try:
            body = self.papers.text_of(card)
        except PapersError:  # stored text failed its content hash
            body = None
        if card.text_status != "full_text" or card.content_truncated or not body:
            self._reject(rv.namespace, raw, "fulltext_absent", f"cached full text required (text_status={card.text_status})")
        by_id = {c.claim_id: c for c in card.claims}
        missing = [c for c in rv.claim_ids if c not in by_id]
        problems = self.papers.verify_claims(card)
        if missing or problems:
            self._reject(rv.namespace, raw, "ungrounded_claims", f"unknown claim ids {missing}; verification {problems[:3]}")
        claims = [by_id[c] for c in rv.claim_ids]
        need = REQUIRED_FOR[ReplicationLabel(rv.label)]
        if [k for k in need if k not in {c.kind for c in claims}]:
            self._reject(rv.namespace, raw, "ungrounded_claims", "selected claims lack required kinds " + str([k for k in need if k not in {c.kind for c in claims}]))
        spec = self.papers.create_replication_spec(card.paper_id, requested_label=rv.label, as_of=as_of, deviations=rv.deviations,
                                                   comparison_plan=rv.comparison_plan)
        if spec["label"] != rv.label:
            self._reject(rv.namespace, raw, "ungrounded_claims", "; ".join(spec["reasons"])[:300])
        return card, claims

    # -- reads ------------------------------------------------------------------

    def get(self, adaptation_id: str) -> Optional[dict]:
        with open_sqlite(self.path) as conn:
            row = conn.execute("SELECT record_json FROM adaptations WHERE adaptation_id=?", (adaptation_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def lineage(self, adaptation_id: str) -> list[dict]:
        with open_sqlite(self.path) as conn:
            rows = {r[0]: r for r in conn.execute("SELECT stage, ref_id, ref_hash, at, detail_json FROM lineage WHERE adaptation_id=?", (adaptation_id,))}
        return [{"stage": s, "ref_id": rows[s][1], "ref_hash": rows[s][2], "at": rows[s][3], **json.loads(rows[s][4])} for s in STAGES if s in rows]

    def _ensure_ledger(self, rec: dict, stored: Optional[dict]) -> None:
        """Make sure the durable ``ResearchLedger`` holds this registration, recovering a known one from its stored record.

        ``ensure_registered`` returns the existing record (counts untouched) or registers once; a conflicting hash raises."""
        if stored is None or stored["content_hash"] != rec["experiment_content_hash"]:
            return
        ts = dt.datetime.fromisoformat(rec["registered_at"])
        self.ledger.ensure_registered(RegisteredExperiment(
            registration_id=rec["registration_id"], experiment_spec_id=rec["experiment_spec_id"], spec_content_hash=stored["content_hash"],
            primary_metric="belief_ic_improvement", baseline_experiment_ids=stored["comparisons"]["baseline_experiment_ids"],
            owner=rec["namespace"], created_at=ts, promotion_conditions=stored["promotion_conditions"],
            failure_conditions=stored["failure_conditions"], variant_budget=1, registered_at=ts).sealed())

    def _registered(self, adaptation_id: str) -> dict:
        rec = self.get(adaptation_id)
        if rec is None:
            raise AdaptationRejected("unknown_adaptation", adaptation_id)
        stored = self.registry.get("experiment_specs", rec["experiment_spec_id"])
        self._ensure_ledger(rec, stored)
        try:
            reg = self.ledger.registration(rec["registration_id"])
        except KeyError:
            reg = None
        if stored is None or reg is None or stored["content_hash"] != rec["experiment_content_hash"] or reg.spec_content_hash != stored["content_hash"]:
            raise AdaptationRejected("unregistered", "experiment is not registered with a matching hash; nothing is executed")
        return rec

    # -- routing and execution (design partition only) --------------------------

    def submit_job(self, adaptation_id: str, queue):
        """Queue the worker's own ``experiment`` job (idempotent); refused unless registration is verified."""
        rec = self._registered(adaptation_id)
        return queue.submit(rec["namespace"], "experiment", {"draft_entry_id": rec["draft_entry_id"], "variants": [rec["variant"]]},
                            idempotency_key=f"adapt:{adaptation_id}")

    def execute_design(self, adaptation_id: str, executors: ExecutorRegistry) -> dict:
        """Run the registered executor in-process exactly as ``PodResearchRunner`` would; reuses a stored result on retry."""
        rec = self._registered(adaptation_id)
        draft = self.memory.get(rec["draft_entry_id"])
        proposal = ResearchProposal.model_validate(draft.content["proposal"])
        key = f"eval:{rec['draft_entry_id']}:{rec['variant']}"
        entry = self.memory.get_by_key(key)
        if entry is None:
            variant = proposal.experiment_plan.variants[0]
            result = executors.run(ExperimentRequest(rec["namespace"], rec["draft_entry_id"], rec["executor_id"], variant, proposal))
            entry = self.memory.append("evaluated_result", {"draft_entry_id": rec["draft_entry_id"], "admission": "not_admitted",
                                                            "variant": variant.model_dump(mode="json"), **result.model_dump(mode="json")},
                                       idempotency_key=key)
        return self.record_result(adaptation_id)

    def record_result(self, adaptation_id: str) -> dict:
        """Bind the measured executor result (memory entry + builtin report hash) into the lineage."""
        rec = self._registered(adaptation_id)
        entry = self.memory.get_by_key(f"eval:{rec['draft_entry_id']}:{rec['variant']}")
        if entry is None or entry.kind != "evaluated_result" or entry.content.get("registration_id") != rec["registration_id"]:
            raise AdaptationRejected("no_result", "no executor result for this registration")
        report = self.context.report(rec["registration_id"], rec["variant"]) or {}
        detail = {"entry_id": entry.entry_id, "metrics": entry.content["metrics"], "executor_spec_hash": entry.content["spec_hash"],
                  "report_hash": report.get("report_hash"), "fixture": report.get("fixture"), "partition": entry.content["data_partition"],
                  "evaluation_label": report.get("evaluation_label")}
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute("INSERT OR IGNORE INTO lineage VALUES (?,?,?,?,?,?)", (adaptation_id, "result", entry.entry_id, entry.content_hash,
                                                                                  self._now().isoformat(), json.dumps(detail, sort_keys=True)))
        return {"adaptation_id": adaptation_id, "lineage": self.lineage(adaptation_id), "admitted": False}
