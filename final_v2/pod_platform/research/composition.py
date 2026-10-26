"""Multi-paper and original hypotheses, measured with the existing reviewed DSL.

The trusted host grants inputs and binds observation snapshots. Models select
only structured features and request missing data; they never supply loaders,
paths, execution code or admission authority.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.sources import SourceCard
from ..papers.components import ComponentEvidenceService, ComponentRequest
from ..pods.charters import pod_spec
from ..pods.schema import scan_injection
from .builtins import stats
from .builtins.timeutil import month_end_known_at, month_index
from .evaluation import ComparisonPlan, ForecastOutcome, RegisteredEvaluator
from .features import FeatureObservation, FeatureRecipe, materialize
from .jobs import open_sqlite

IDENT = r"^[a-z][a-z0-9_]{0,63}$"


class NativeTarget(StrictModel):
    source_id: str = Field(pattern=IDENT)
    metric: str = Field(pattern=IDENT)
    horizon_months: int = Field(ge=1, le=36)
    resolve_lag_days: int = Field(default=0, ge=0, le=365)


class MeasurementGrant(StrictModel):
    grant_id: str = Field(pattern=IDENT)
    namespace: str
    code_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_metrics: dict[str, list[str]] = Field(min_length=1, max_length=12)
    source_cards: list[SourceCard] = Field(min_length=1, max_length=12)
    targets: dict[str, NativeTarget] = Field(min_length=1, max_length=8)
    formation_dates: list[dt.datetime] = Field(min_length=1, max_length=72)
    security_ids: list[str] = Field(min_length=3, max_length=500)
    partition: Literal["design", "prospective"] = "design"
    protected_from: dt.datetime = dt.datetime(
        2019, 1, 1, tzinfo=dt.timezone.utc
    )
    authorized_training_end: dt.datetime | None = None
    max_observations: int = Field(default=50000, ge=1, le=300_000)

    @model_validator(mode="after")
    def bounds(self):
        require_aware(self.protected_from, "protected_from")
        if len(set(self.security_ids)) != len(self.security_ids) or len(
            set(self.formation_dates)
        ) != len(self.formation_dates):
            raise ValueError("duplicate formation or security identity")
        for day in self.formation_dates:
            require_aware(day, "formation date")
            if self.partition == "design" and day >= self.protected_from:
                raise ValueError(
                    "protected check/replay dates cannot become a composition design"
                )
            if self.partition == "design" and any(
                month_end_known_at(
                    month_index(day.date()) + t.horizon_months,
                    t.resolve_lag_days,
                )
                >= self.protected_from
                for t in self.targets.values()
            ):
                raise ValueError(
                    "native target maturity crosses the host-protected boundary"
                )
        cards = {c.source_id: c for c in self.source_cards}
        if self.authorized_training_end is not None:
            require_aware(
                self.authorized_training_end, "authorized_training_end"
            )
        if self.protected_from.date() > dt.date(2019, 1, 1) and any(
            source.lower().startswith("comp_") for source in cards
        ):
            # A new host grant can adopt an explicitly authorized training
            # split. Existing grants and their exposure ledgers stay immutable.
            if (
                self.authorized_training_end != self.protected_from
                or self.protected_from.date() > dt.date(2021, 1, 1)
            ):
                raise ValueError(
                    "Expanded competition training needs host authorization "
                    "and must stop before the 2021 evaluation."
                )
        if set(cards) != set(self.source_metrics):
            raise ValueError("each granted source needs its actual source card")
        if any(
            not c.access_confirmed
            or c.status.value in ("REJECTED", "QUARANTINED")
            for c in cards.values()
        ):
            raise ValueError("source access unavailable")
        for target in self.targets.values():
            if target.metric not in self.source_metrics.get(
                target.source_id, []
            ):
                raise ValueError(
                    "native target is not in granted source metrics"
                )
        return self


class FeatureNode(StrictModel):
    node_id: str = Field(pattern=IDENT)
    operation: Literal[
        "input", "product", "difference", "ratio", "mean", "rank"
    ]
    recipe: FeatureRecipe | None = None
    inputs: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def shape(self):
        if self.operation == "input":
            if self.recipe is None or self.inputs:
                raise ValueError(
                    "input node requires one existing FeatureRecipe and no inputs"
                )
        elif (
            self.recipe is not None
            or len(self.inputs) != (1 if self.operation == "rank" else 2)
            and self.operation != "mean"
        ):
            raise ValueError("operator arity or input recipe invalid")
        elif self.operation == "mean" and len(self.inputs) < 2:
            raise ValueError("mean requires at least two prior nodes")
        return self


class DataDependency(StrictModel):
    dependency_id: str = Field(pattern=IDENT)
    role: Literal["core", "optional_control"]
    source_id: str = Field(pattern=IDENT)
    metric: str = Field(pattern=IDENT)
    request_id: str | None = Field(default=None, min_length=1, max_length=200)
    missing_reason: str | None = Field(
        default=None, min_length=1, max_length=600
    )


class CompositionSpec(StrictModel):
    composition_name: str = Field(pattern=IDENT)
    grant_id: str = Field(pattern=IDENT)
    reasoning_mode: Literal[
        "literature_replication",
        "mechanism_transfer",
        "error_diagnosis",
        "contradiction",
        "cross_source_composition",
        "peer_transmission",
        "measurement_improvement",
        "method_improvement",
        "own_reasoning",
    ]
    question: str = Field(min_length=10, max_length=1000)
    mechanism: str = Field(min_length=20, max_length=2000)
    own_reasoning: str | None = Field(
        default=None, min_length=20, max_length=2000
    )
    paper_components: list[ComponentRequest] = Field(
        default_factory=list, max_length=16
    )
    dependencies: list[DataDependency] = Field(min_length=1, max_length=24)
    target_id: str = Field(pattern=IDENT)
    nodes: list[FeatureNode] = Field(min_length=1, max_length=24)
    outputs: list[str] = Field(min_length=1, max_length=8)
    baselines: list[str] = Field(min_length=1, max_length=8)
    causal_edges: list[tuple[str, str, str]] = Field(
        min_length=1, max_length=24
    )
    applicability: str = Field(min_length=10, max_length=1000)
    competing_explanations: list[str] = Field(min_length=1, max_length=8)
    falsification: list[str] = Field(min_length=1, max_length=8)
    variant_budget: int = Field(default=8, ge=1, le=32)

    @model_validator(mode="after")
    def graph(self):
        if not self.paper_components and not self.own_reasoning:
            raise ValueError(
                "select paper components or declare original reasoning"
            )
        seen = set()
        for node in self.nodes:
            if node.node_id in seen or set(node.inputs) - seen:
                raise ValueError(
                    "feature graph needs unique, prior-only node references"
                )
            seen.add(node.node_id)
        if (
            set(self.outputs + self.baselines) - seen
            or len(set(self.outputs)) != len(self.outputs)
            or len(set(self.baselines)) != len(self.baselines)
        ):
            raise ValueError("unknown or duplicate output/baseline")
        if len({d.dependency_id for d in self.dependencies}) != len(
            self.dependencies
        ):
            raise ValueError("duplicate dependency ids")
        if any(
            not s.strip()
            for s in self.competing_explanations + self.falsification
        ) or any(not all(edge) for edge in self.causal_edges):
            raise ValueError("causal claims and falsification must be explicit")
        if scan_injection(json.dumps(self.model_dump(mode="json"))):
            raise ValueError("unsafe instruction content")
        return self


def observation_digest(observations):
    rows = [
        FeatureObservation.model_validate(o).model_dump(mode="json")
        for o in observations
    ]
    return canonical_digest(sorted(rows, key=lambda o: o["observation_id"]))


def evaluate_graph(nodes, observations, formation_at, security_ids):
    """Evaluate a reviewed graph; the caller owns calendar/partition authorization."""
    cutoff = require_aware(formation_at, "formation_at")
    rows = [FeatureObservation.model_validate(o) for o in observations]
    rows = [
        o
        for o in rows
        if max(o.period_end, o.available_at, o.version_known_at) <= cutoff
    ]
    values = {}
    for raw in nodes:
        node = FeatureNode.model_validate(raw)
        if node.recipe:
            values[node.node_id] = {
                r["security_id"]: r["value"]
                for r in materialize(
                    node.recipe,
                    rows,
                    formation_at=cutoff,
                    security_ids=security_ids,
                )
            }
            continue
        out = {}
        for security in security_ids:
            parts = [values[n][security] for n in node.inputs]
            if any(p is None for p in parts):
                out[security] = None
                continue
            if node.operation == "product":
                value = math.prod(parts)
            elif node.operation == "difference":
                value = parts[0] - parts[1]
            elif node.operation == "ratio":
                value = parts[0] / parts[1] if parts[1] else None
            else:
                value = sum(parts) / len(parts)
            out[security] = (
                value if value is None or math.isfinite(value) else None
            )
        if node.operation == "rank":
            names = [s for s, v in out.items() if v is not None]
            out.update(zip(names, stats.centered_rank([out[s] for s in names])))
        values[node.node_id] = out
    return values


class CompositionService:
    def __init__(
        self,
        path,
        *,
        papers,
        namespace,
        clock,
        code_digest,
        dependency_lookup=None,
        max_trials=12,
    ):
        if (
            pod_spec(namespace).namespace != namespace
            or not 1 <= max_trials <= 12
        ):
            raise ValueError("canonical pod and bounded trial budget required")
        self.path, self.papers, self.namespace = Path(path), papers, namespace
        self.clock, self.code_digest, self.dependency_lookup = (
            clock,
            code_digest,
            dependency_lookup,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open_sqlite(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS compositions (id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, hash TEXT NOT NULL)"
            )
            for op in ("UPDATE", "DELETE"):
                conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS composition_no_{op.lower()} BEFORE {op} ON compositions BEGIN SELECT RAISE(ABORT,'immutable composition'); END"
                )
        self._put(
            "policy",
            "policy",
            {
                "namespace": namespace,
                "code_digest": code_digest,
                "max_trials": max_trials,
            },
        )
        self.max_trials = max_trials

    def _put(self, key, kind, value):
        sha = canonical_digest([kind, value])
        with open_sqlite(self.path, immediate=True) as conn:
            row = conn.execute(
                "SELECT hash FROM compositions WHERE id=?", (key,)
            ).fetchone()
            if row and row[0] != sha:
                raise ValueError("immutable composition identity differs")
            conn.execute(
                "INSERT OR IGNORE INTO compositions VALUES(?,?,?,?)",
                (
                    key,
                    kind,
                    json.dumps(value, sort_keys=True, allow_nan=False),
                    sha,
                ),
            )
        return value

    def _get(self, key):
        with open_sqlite(self.path) as conn:
            row = conn.execute(
                "SELECT kind,payload,hash FROM compositions WHERE id=?", (key,)
            ).fetchone()
        if not row:
            return None
        value = json.loads(row[1])
        if canonical_digest([row[0], value]) != row[2]:
            raise ValueError("composition integrity failure")
        return value

    def register_grant(self, grant):
        """Register immutable host-reviewed measurements for one pod."""
        value = MeasurementGrant.model_validate(grant)
        if (
            value.namespace != self.namespace
            or value.code_digest != self.code_digest
        ):
            raise ValueError("foreign or stale measurement grant")
        return self._put(
            "grant:" + value.grant_id, "grant", value.model_dump(mode="json")
        )

    def submit(self, spec):
        """Freeze a multi-source hypothesis and its comparisons before data access."""
        value = CompositionSpec.model_validate(spec)
        grant = MeasurementGrant.model_validate(
            self._get("grant:" + value.grant_id) or {}
        )
        if value.target_id not in grant.targets:
            raise ValueError("target is not granted")
        for node in value.nodes:
            if (
                node.recipe
                and node.recipe.metric
                not in grant.source_metrics.get(node.recipe.source_id, [])
            ):
                raise ValueError("feature reads an ungranted source/metric")
        deps = {
            (d.source_id, d.metric)
            for d in value.dependencies
            if d.role == "core"
        }
        intrinsic = {
            (n.recipe.source_id, n.recipe.metric)
            for n in value.nodes
            if n.recipe
        }
        target = grant.targets[value.target_id]
        if not intrinsic | {(target.source_id, target.metric)} <= deps:
            raise ValueError(
                "every feature and native target measurement must be a core dependency"
            )
        components = ComponentEvidenceService(
            papers=self.papers, namespace=self.namespace, clock=self.clock
        )
        evidence = [
            components.resolve(c).model_dump(mode="json")
            for c in value.paper_components
        ]
        at = require_aware(self.clock(), "clock")
        if grant.partition == "prospective" and any(
            day < at for day in grant.formation_dates
        ):
            raise ValueError("prospective formations must follow registration")
        candidates = list(
            dict.fromkeys(
                value.outputs + [n.node_id for n in value.nodes if n.recipe]
            )
        )
        comparisons = [
            (n, b) for n in candidates for b in value.baselines if n != b
        ]
        if not comparisons or len(comparisons) > value.variant_budget:
            raise ValueError(
                "candidate/component-alone comparisons exceed declared variant budget"
            )
        key = (
            "composition_"
            + canonical_digest([self.namespace, value.composition_name])[:24]
        )
        old = self._get(key)
        frozen = {
            "composition_id": key,
            "namespace": self.namespace,
            "spec": value.model_dump(mode="json"),
            "grant": grant.model_dump(mode="json"),
            "evidence": evidence,
            "comparisons": comparisons,
            "registered_at": old["registered_at"] if old else at.isoformat(),
            "scope": "research_only",
            "admitted": False,
        }
        return self._put(key, "plan", frozen)

    def dependency_status(self, composition_id):
        """Read missing core measurements and optional controls separately."""
        plan = self._plan(composition_id)
        bindings = self._get("inputs:" + composition_id) or {}
        rows = []
        for dep in plan["spec"]["dependencies"]:
            receipt = None
            if dep["request_id"] and self.dependency_lookup:
                try:
                    receipt = self.dependency_lookup(
                        self.namespace, dep["request_id"]
                    )
                except ValueError:
                    receipt = {"status": "not_available_in_namespace"}
            bound = dep["source_id"] in bindings
            rows.append({**dep, "bound": bound, "receipt": receipt})
        missing = [r for r in rows if r["role"] == "core" and not r["bound"]]
        return {
            "status": "waiting_data" if missing else "ready",
            "dependencies": rows,
            "missing_core": [r["dependency_id"] for r in missing],
            "omitted_controls": [
                r["dependency_id"]
                for r in rows
                if r["role"] == "optional_control" and not r["bound"]
            ],
        }

    def _plan(self, composition_id):
        plan = self._get(composition_id)
        if (
            not plan
            or plan.get("namespace") != self.namespace
            or plan.get("scope") != "research_only"
        ):
            raise ValueError("unknown composition for this pod")
        return plan

    def get_plan(self, composition_id):
        """Read an integrity-checked frozen plan owned by this pod."""
        return self._plan(composition_id)

    def bind_inputs(self, composition_id, source_snapshots):
        """Trusted host only: pin ingested normalized observations, before loader access."""
        plan = self._plan(composition_id)
        if set(source_snapshots) - set(plan["grant"]["source_metrics"]) or any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in source_snapshots.values()
        ):
            raise ValueError(
                "unknown source or invalid observation snapshot digest"
            )
        return self._put(
            "inputs:" + composition_id, "inputs", dict(source_snapshots)
        )

    def predict(self, composition_id, observations, formation_at):
        """Pure frozen native graph; as-of filters precede all transformations."""
        plan = self._plan(composition_id)
        grant = MeasurementGrant.model_validate(plan["grant"])
        cutoff = require_aware(formation_at, "formation_at")
        if cutoff not in grant.formation_dates:
            raise ValueError("formation is outside the registered calendar")
        return evaluate_graph(
            plan["spec"]["nodes"], observations, cutoff, grant.security_ids
        )

    def materialize_graph(self, composition_id, observations, formation_at):
        """Return frozen feature values and conservative evidence clocks, without labels."""
        values = self.predict(composition_id, observations, formation_at)
        plan = self._plan(composition_id)
        cutoff = require_aware(formation_at, "formation_at")
        inputs = {
            (n["recipe"]["source_id"], n["recipe"]["metric"])
            for n in plan["spec"]["nodes"]
            if n.get("recipe")
        }
        eligible = [FeatureObservation.model_validate(o) for o in observations]
        timestamps = [
            max(o.available_at, o.version_known_at)
            for o in eligible
            if (o.source_id, o.metric) in inputs
            and max(o.available_at, o.version_known_at, o.period_end) <= cutoff
        ]
        latest = max(timestamps, default=None)
        return [
            {
                "security_id": security,
                "values": {
                    node: column[security] for node, column in values.items()
                },
                "latest_evidence_at": latest.isoformat() if latest else None,
            }
            for security in plan["grant"]["security_ids"]
        ]

    def run(self, composition_id, loader):
        """Reserve a trial, verify bound observations and measure native forecasts."""
        plan = self._plan(composition_id)
        old = self._get("result:" + composition_id)
        if old:
            return old
        dependency = self.dependency_status(composition_id)
        if dependency["status"] != "ready":
            return {
                **dependency,
                "composition_id": composition_id,
                "admitted": False,
            }
        with open_sqlite(self.path, immediate=True) as conn:
            if conn.execute(
                "SELECT 1 FROM compositions WHERE id=?",
                ("trial:" + composition_id,),
            ).fetchone():
                raise ValueError(
                    "counted composition attempt already exists; interrupted attempts are not rerun"
                )
            if (
                conn.execute(
                    "SELECT COUNT(*) FROM compositions WHERE kind='trial'"
                ).fetchone()[0]
                >= self.max_trials
            ):
                raise ValueError("composition trial budget exhausted")
            value = {
                "composition_id": composition_id,
                "at": require_aware(self.clock(), "clock").isoformat(),
            }
            conn.execute(
                "INSERT INTO compositions VALUES(?,?,?,?)",
                (
                    "trial:" + composition_id,
                    "trial",
                    json.dumps(value),
                    canonical_digest(["trial", value]),
                ),
            )
        evaluator = RegisteredEvaluator(
            self.path.with_suffix(".evaluation.sqlite")
        )
        try:
            grant = MeasurementGrant.model_validate(plan["grant"])
            now = require_aware(self.clock(), "clock")
            evaluator.register(
                ComparisonPlan(
                    experiment_id=composition_id,
                    owner=self.namespace,
                    hypothesis_id=composition_id,
                    registered_at=dt.datetime.fromisoformat(
                        plan["registered_at"]
                    ),
                    target_id=plan["spec"]["target_id"],
                    source_ids=list(grant.source_metrics),
                    variant_budget=plan["spec"]["variant_budget"],
                    partition=grant.partition,
                )
            )
            observations = [
                FeatureObservation.model_validate(o) for o in loader()
            ]
            if len(observations) > grant.max_observations:
                raise ValueError("observation budget exceeded")
            identities = {}
            for observation in observations:
                if (
                    observation.metric
                    not in grant.source_metrics.get(observation.source_id, [])
                    or observation.security_id not in grant.security_ids
                ):
                    raise ValueError(
                        "observation is outside the granted source, metric or universe"
                    )
                if (
                    grant.partition == "design"
                    and observation.period_end >= grant.protected_from
                ):
                    raise ValueError(
                        "protected rows supplied to native research"
                    )
                raw = observation.model_dump(mode="json")
                if (
                    observation.observation_id in identities
                    and identities[observation.observation_id] != raw
                ):
                    raise ValueError("observation ID has conflicting content")
                identities[observation.observation_id] = raw
            bindings = self._get("inputs:" + composition_id)
            for source, sha in bindings.items():
                if (
                    observation_digest(
                        [o for o in observations if o.source_id == source]
                    )
                    != sha
                ):
                    raise ValueError("normalized observation snapshot changed")
            target = grant.targets[plan["spec"]["target_id"]]
            results = []
            values_by_day = {
                day: self.predict(composition_id, observations, day)
                for day in grant.formation_dates
            }
            labels_by_identity = {}
            for observation in observations:
                if (
                    observation.source_id == target.source_id
                    and observation.metric == target.metric
                    and max(
                        observation.available_at, observation.version_known_at
                    )
                    <= now
                ):
                    key = (
                        observation.security_id,
                        month_index(observation.period_end.date()),
                    )
                    previous = labels_by_identity.get(key)
                    if (
                        previous is None
                        or observation.version_known_at
                        > previous.version_known_at
                    ):
                        labels_by_identity[key] = observation
            for candidate, baseline in plan["comparisons"]:
                pairs = []
                for day in grant.formation_dates:
                    values = values_by_day[day]
                    target_month = (
                        month_index(day.date()) + target.horizon_months
                    )
                    for security in grant.security_ids:
                        label = labels_by_identity.get((security, target_month))
                        if (
                            label is None
                            or values[candidate][security] is None
                            or values[baseline][security] is None
                        ):
                            continue
                        resolves = max(
                            label.available_at,
                            label.version_known_at,
                            month_end_known_at(
                                target_month, target.resolve_lag_days
                            ),
                        )
                        if resolves > now:
                            continue
                        pairs.append(
                            ForecastOutcome(
                                security_id=security,
                                formation_at=day,
                                feature_available_at=day,
                                label_resolves_at=resolves,
                                prediction=values[candidate][security],
                                baseline_prediction=values[baseline][security],
                                target=label.value,
                            )
                        )
                if pairs:
                    result = evaluator.evaluate(
                        composition_id,
                        candidate + "__vs__" + baseline,
                        pairs,
                        owner=self.namespace,
                        evaluation_asof=now,
                        source_cards=grant.source_cards,
                        allowed_partitions=(grant.partition,),
                    )
                    results.append(
                        {"candidate": candidate, "baseline": baseline, **result}
                    )
            report = {
                "status": "measured" if results else "inconclusive",
                "comparisons": results,
                "data_dependencies": dependency,
                "scope": "research_only",
                "admitted": False,
                "protected_check_accesses": 0,
                "orders": 0,
                "limitations": [
                    "Native-target research only; paper interpretations are not full replications.",
                    "No return bridge, independent challenge, trading approval or profitability established.",
                ],
            }
        except Exception as exc:
            report = {
                "status": "failed",
                "error": str(exc)[:600],
                "scope": "research_only",
                "admitted": False,
            }
        finally:
            evaluator.close()
        return self._put("result:" + composition_id, "result", report)

    def integrated_plan(self, composition_id):
        """Expose the exact frozen measurement for a separate financial review."""
        plan = self._plan(composition_id)
        return {
            **plan,
            "input_snapshots": self._get("inputs:" + composition_id),
            "measurement": self._get("result:" + composition_id),
            "plan_digest": canonical_digest(plan),
            "code_digest": self.code_digest,
            "required_gates": [
                "exact_feature_mapping_review",
                "registered_financial_bridge",
                "frozen_independent_check",
                "existing_admission_and_risk",
            ],
            "executable_operators": "FeatureRecipe/materialize plus declared pure composition graph",
            "admitted": False,
        }

    def status(self):
        """Read this pod's grants, dependency state and measured research results."""
        with open_sqlite(self.path) as conn:
            rows = conn.execute(
                "SELECT id,kind FROM compositions WHERE kind IN ('grant','plan') ORDER BY rowid DESC LIMIT 200"
            ).fetchall()
        grants = [self._get(key) for key, kind in rows if kind == "grant"]
        plans = [
            {
                "composition_id": key,
                "composition_name": self._get(key)["spec"]["composition_name"],
                "dependencies": self.dependency_status(key),
                "measurement": self._get("result:" + key),
            }
            for key, kind in rows
            if kind == "plan"
        ]
        return {
            "namespace": self.namespace,
            "grants": grants,
            "compositions": plans,
            "scope": "research_only",
        }
