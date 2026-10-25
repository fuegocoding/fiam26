"""Independent challenge and promotion service (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 5.4, 11.7, 13, 14).

Research Integrity is code: it owns the protected partitions, counts every scored
variant and check access, runs the baseline ladder and a timing-shift placebo,
and issues an admission certificate only when *every* promotion criterion was
measured and met under a policy frozen before outcomes were opened. A candidate
that cannot be evaluated is BLOCKED (never given invented evidence). Prose
approval is not an admission decision, and a fixture certificate is labelled so
that it can never authorize a competition book.

Statistics are descriptive: standard errors are across formation dates, not
HAC/block-bootstrap, and a t-threshold with a Bonferroni-style multiplicity
factor does not guarantee FDR control (spec 13.8).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import hmac
import math
import statistics
import uuid
from typing import Any, Mapping, Optional, Sequence

from pydantic import Field, field_validator, model_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.errors import ErrorCode
from ..contracts.experiments import (
    ExperimentAdmission,
    ExperimentBudget,
    ExperimentClaim,
    ExperimentComparisons,
    ExperimentCosts,
    ExperimentData,
    ExperimentIdentity,
    ExperimentModel,
    ExperimentPortfolio,
    ExperimentSample,
    ExperimentSpec,
    ExperimentStatistics,
    ExperimentTime,
)
from ..contracts.hypotheses import HypothesisSpec, ResearchMode
from ..contracts.sources import SourceCard
from ..research.certificates import CertificateScope
from ..research.evaluation import (
    ComparisonPlan,
    ForecastOutcome,
    RegisteredEvaluator,
)
from ..research.promotion import (
    AdmissionCertificate,
    ModelLifecycle,
    ModelState,
    PromotionCriteria,
    PromotionOutcome,
    StateTransition,
)
from ..research.registration import (
    TRIAL_KINDS,
    RegisteredExperiment,
    ResearchLedger,
    TrialRecord,
)
from ..research.registry import Registry
from .common import FIXTURE_LIMITATION, SYNTHETIC_LABEL, fail, now_utc
from .namespaces import RoleContext, RoleGateway
from .publication import SCOPE_FIXTURE, SCOPE_FORMATION, SCOPE_RESEARCH, SCOPES

SYNTHETIC_DATASET_ID = "SYNTHETIC_FIXTURE"


class Outcome(str, enum.Enum):
    ADMITTED = "ADMITTED"
    SHADOW = "SHADOW"
    REJECTED = "REJECTED"
    DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"
    BLOCKED = "BLOCKED"


class IntegrityPolicy(StrictModel):
    """Promotion criteria, fixed before any outcome partition is opened (spec 13.10)."""

    policy_id: str = Field(min_length=1)
    frozen_at: dt.datetime
    min_dates_design: int = Field(default=8, ge=2)
    min_dates_check: int = Field(default=4, ge=2)
    min_obs_per_date: int = Field(default=30, ge=3)
    min_mean_ic: float = Field(default=0.03, gt=0)
    min_improvement: float = Field(default=0.02, gt=0)
    min_bridge_ic: float = 0.0
    # Preregistered net-of-cost bridge: mean monthly tail-portfolio spread after trading costs must exceed min_net_spread
    # in design and check. cost_bps_floor_one_way is the host cost assumption; data-declared costs may only raise it.
    min_net_spread: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    cost_bps_floor_one_way: float = Field(
        default=0.0, ge=0, le=1000, allow_inf_nan=False
    )
    max_placebo_fraction: float = Field(default=0.5, gt=0, le=1)
    min_t_stat: float = Field(default=2.0, gt=0)
    family_alpha: float = Field(default=0.05, gt=0, lt=1)
    variant_budget: int = Field(default=4, ge=4, le=100)
    expiry_policy: str = "expires with the 20-day forecast horizon; review at the next scheduled research review"

    @field_validator("frozen_at")
    @classmethod
    def _aware(cls, value: dt.datetime) -> dt.datetime:
        return require_aware(value, "frozen_at")

    def criteria(self) -> PromotionCriteria:
        return PromotionCriteria(
            min_coverage=f">= {self.min_dates_design} design and {self.min_dates_check} check formation dates, >= {self.min_obs_per_date} names per date",
            improvement_vs_primary_baseline=f"paired mean rank-IC improvement >= {self.min_improvement} in design and check",
            direction_and_economic_size=f"mean rank IC has the declared sign and |IC| >= {self.min_mean_ic} in design and check",
            stability_requirement=f"declared sign and |IC| >= {self.min_mean_ic / 2:.4f} in both design halves and check; timing-shift placebo |IC| <= {self.max_placebo_fraction} x |IC|",
            return_translation_requirement=f"mean rank IC of prediction vs bridge-signed forward return (gross, descriptive) in check >= {self.min_bridge_ic}",
            implementation_requirement=(
                "forecasts carry the 20-day horizon, decimal_return unit and a declared bridge; measured tail-portfolio net spread "
                f"(declared turnover x max(data cost, {self.cost_bps_floor_one_way}bp one-way floor)) > {self.min_net_spread} in design and check"
            ),
            permitted_statistical_uncertainty=f"descriptive t-stat across formation dates >= {self.min_t_stat} (not dependence-adjusted)",
            multiple_testing_treatment=f"t >= normal quantile for alpha {self.family_alpha}/(2m) across m candidates; scored variants <= {self.variant_budget}",
        )


class CandidateRecord(StrictModel):
    candidate_id: str = Field(min_length=1)
    charter: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    experiment_spec_id: str = Field(min_length=1)
    mechanism_family: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_id: str = Field(min_length=1)
    horizon_trading_days: int = Field(ge=1)
    source_ids: list[str] = Field(min_length=1)
    question: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    declared_sign: int
    approved_universe: str = Field(min_length=1)
    feature_hashes: dict[str, str] = Field(min_length=1)
    data_hashes: dict[str, str] = Field(min_length=1)
    uses_pretrained_model: bool = False
    evidence_mode: str = "synthetic_fixture"

    @model_validator(mode="after")
    def _check(self) -> "CandidateRecord":
        if self.declared_sign not in (-1, 1):
            raise ValueError("declared_sign must be +1 or -1")
        return self

    @property
    def registration_id(self) -> str:
        return f"reg_{self.experiment_spec_id}"


class IntegrityDecision(StrictModel):
    candidate_id: str
    charter: str
    outcome: Outcome
    scope: str
    reasons: list[str] = Field(default_factory=list)
    criteria: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, Optional[float]] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    evidence: dict[str, str] = Field(default_factory=dict)
    certificate_id: Optional[str] = None
    lifecycle: list[dict[str, str]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    trial_counts: dict[str, int] = Field(default_factory=dict)
    uncertainty: dict[str, Optional[float]] = Field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return self.outcome == Outcome.ADMITTED

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------- statistics


def _finite_series(result: Mapping[str, Any], key: str) -> list[float]:
    return [float(d[key]) for d in result["dates"] if d.get(key) is not None]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def timing_shift_placebo(
    observations: Sequence[ForecastOutcome],
) -> list[ForecastOutcome]:
    """Cyclically shift each security's prediction by one formation date (spec 13.9).

    Everything else (timing, labels, baseline) is unchanged, so the economic
    interpretation of the prediction is removed while cross-sectional structure
    and dependence over time are preserved. A genuine signal should vanish.
    """
    by_security: dict[str, list[ForecastOutcome]] = {}
    for obs in sorted(
        observations, key=lambda o: (o.security_id, o.formation_at)
    ):
        by_security.setdefault(obs.security_id, []).append(obs)
    out: list[ForecastOutcome] = []
    for rows in by_security.values():
        if len(rows) < 2:
            continue
        predictions = [r.prediction for r in rows]
        shifted = predictions[1:] + predictions[:1]
        for row, prediction in zip(rows, shifted):
            out.append(
                ForecastOutcome(
                    security_id=row.security_id,
                    formation_at=row.formation_at,
                    feature_available_at=row.feature_available_at,
                    label_resolves_at=row.label_resolves_at,
                    prediction=prediction,
                    baseline_prediction=row.baseline_prediction,
                    target=row.target,
                    net_forward_return=row.net_forward_return,
                )
            )
    return out


def _date_counts(observations: Sequence[ForecastOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for obs in observations:
        key = obs.formation_at.isoformat()
        counts[key] = counts.get(key, 0) + 1
    return counts


# --------------------------------------------------------- producer-sealed evidence


def _obs_digest(observations: Sequence[ForecastOutcome]) -> str:
    return canonical_digest(
        [
            o.model_dump(mode="json")
            for o in sorted(
                observations, key=lambda o: (o.formation_at, o.security_id)
            )
        ]
    )


@dataclasses.dataclass(frozen=True)
class EvidenceBundle:
    """Observations plus tail-portfolio membership, sealed by the trusted producer AFTER it verified provenance.

    Not a pod / model argument: only a holder of the host-derived ``EvidenceSeal`` key can mint one, and the seal covers the
    candidate record, model hash, scope, provenance digests, every observation and every portfolio membership.
    """

    bundle_id: str
    candidate_id: str
    record_digest: str
    model_hash: str
    scope: str
    provenance: tuple[tuple[str, str], ...]
    design_observations: tuple[ForecastOutcome, ...]
    check_observations: tuple[ForecastOutcome, ...]
    design_portfolio: tuple[Mapping[str, Any], ...]
    check_portfolio: tuple[Mapping[str, Any], ...]
    bridge_sign: int
    tail_fraction: float
    cost_bps_one_way: float
    cost_source: str
    issued_at: dt.datetime
    seal: str

    def body(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "candidate_id": self.candidate_id,
            "record_digest": self.record_digest,
            "model_hash": self.model_hash,
            "scope": self.scope,
            "provenance": [list(p) for p in self.provenance],
            "design_observations": _obs_digest(self.design_observations),
            "check_observations": _obs_digest(self.check_observations),
            "design_portfolio": canonical_digest(
                [dict(p) for p in self.design_portfolio]
            ),
            "check_portfolio": canonical_digest(
                [dict(p) for p in self.check_portfolio]
            ),
            "bridge_sign": self.bridge_sign,
            "tail_fraction": self.tail_fraction,
            "cost_bps_one_way": self.cost_bps_one_way,
            "cost_source": self.cost_source,
            "issued_at": self.issued_at.isoformat(),
        }


class EvidenceSeal:
    """Host-held HMAC capability. Derive from the host authority key; give the same seal to the producer and to integrity."""

    def __init__(self, key: bytes, gateway: Optional[RoleGateway] = None):
        if not isinstance(key, bytes) or len(key) < 16:
            raise ValueError(
                "evidence seal needs a secret key of at least 16 bytes"
            )
        self._key = key
        self._gateway = gateway or RoleGateway()

    def __repr__(self) -> str:
        return "EvidenceSeal(<key hidden>)"

    def __reduce__(self):  # keys never travel through pickles / logs
        raise TypeError("EvidenceSeal cannot be serialized")

    @property
    def fingerprint(self) -> str:
        return hmac.new(
            self._key, b"evidence-seal-fingerprint", hashlib.sha256
        ).hexdigest()[:16]

    def _mac(self, body: Mapping[str, Any]) -> str:
        return hmac.new(
            self._key, canonical_digest(dict(body)).encode(), hashlib.sha256
        ).hexdigest()

    def issue(
        self,
        ctx: RoleContext,
        *,
        record: "CandidateRecord",
        scope: str,
        provenance: Mapping[str, str],
        design_observations: Sequence[ForecastOutcome],
        check_observations: Sequence[ForecastOutcome],
        design_portfolio: Sequence[Mapping[str, Any]],
        check_portfolio: Sequence[Mapping[str, Any]],
        bridge_sign: int,
        tail_fraction: float,
        cost_bps_one_way: float,
        cost_source: str,
        issued_at: dt.datetime,
    ) -> EvidenceBundle:
        self._gateway.authorize(ctx, "integrity.evaluate_protected")
        self._gateway.authorize(ctx, "certificate.issue")
        draft = EvidenceBundle(
            "",
            record.candidate_id,
            canonical_digest(record.model_dump(mode="json")),
            record.model_hash,
            scope,
            tuple(sorted((str(k), str(v)) for k, v in provenance.items())),
            tuple(design_observations),
            tuple(check_observations),
            tuple(dict(p) for p in design_portfolio),
            tuple(dict(p) for p in check_portfolio),
            int(bridge_sign),
            float(tail_fraction),
            float(cost_bps_one_way),
            cost_source,
            require_aware(issued_at, "issued_at"),
            "",
        )
        with_id = dataclasses.replace(
            draft, bundle_id="ev_" + canonical_digest(draft.body())[:20]
        )
        return dataclasses.replace(with_id, seal=self._mac(with_id.body()))

    def verify(self, bundle: Any) -> bool:
        return isinstance(bundle, EvidenceBundle) and hmac.compare_digest(
            str(bundle.seal), self._mac(bundle.body())
        )


def net_spread_series(
    observations: Sequence[ForecastOutcome],
    portfolio: Sequence[Mapping[str, Any]],
    *,
    bridge_sign: int,
    tail_fraction: float,
    cost_bps_one_way: float,
) -> Optional[list[dict[str, float]]]:
    """Gross tail spread, one-way turnover and net spread per formation date, from sealed observations and membership.

    ``net_forward_return`` carries ``bridge_sign x`` raw return, so the raw return is ``bridge_sign x`` it. The first date pays
    full entry turnover. Returns None when membership is inconsistent with the observations (unmeasured, not assumed).
    """
    rets = {
        (o.security_id, o.formation_at.isoformat()): bridge_sign
        * o.net_forward_return
        for o in observations
        if o.net_forward_return is not None
    }
    scores: dict[str, dict[str, float]] = {}
    for o in observations:
        scores.setdefault(o.formation_at.isoformat(), {})[o.security_id] = (
            bridge_sign * o.prediction
        )
    cost, prev, out = cost_bps_one_way / 1e4, {}, []
    for entry in sorted(portfolio, key=lambda e: e["formation_at"]):
        date, longs, shorts = (
            entry["formation_at"],
            list(entry["longs"]),
            list(entry["shorts"]),
        )
        signed = scores.get(date, {})
        q = max(1, int(len(signed) * tail_fraction))
        expected = sorted(signed, key=lambda s: (-signed[s], s))[:q]
        if (
            not longs
            or not shorts
            or len(set(longs)) != len(longs)
            or len(set(shorts)) != len(shorts)
            or set(longs) & set(shorts)
            or sorted(longs) != sorted(expected)
            or len(shorts) > q
            or any((s, date) not in rets for s in longs + shorts)
        ):
            return None
        gross = math.fsum(rets[(s, date)] for s in longs) / len(
            longs
        ) - math.fsum(rets[(s, date)] for s in shorts) / len(shorts)
        weights = {s: 1.0 / len(longs) for s in longs}
        for s in shorts:
            weights[s] = weights.get(s, 0.0) - 1.0 / len(shorts)
        turnover = math.fsum(
            abs(weights.get(s, 0.0) - prev.get(s, 0.0))
            for s in set(weights) | set(prev)
        )
        prev = weights
        out.append(
            {
                "gross": gross,
                "turnover": turnover,
                "net": gross - cost * turnover,
            }
        )
    return out


class IntegrityService:
    def __init__(
        self,
        *,
        registry: Registry,
        ledger: ResearchLedger,
        evaluator: RegisteredEvaluator,
        policy: IntegrityPolicy,
        source_cards: Sequence[SourceCard],
        clock=now_utc,
        gateway: Optional[RoleGateway] = None,
        evidence_seal: Optional[EvidenceSeal] = None,
        allow_unbound_fixture: bool = True,
    ):
        """``evidence_seal`` verifies producer-sealed bundles (required for any non-fixture scope).

        ``allow_unbound_fixture`` keeps the legacy caller-supplied-observation path for FIXTURE certificates only; it can never
        issue research or formation certificates. Hosts that run real challenges should set it to False.
        """
        self.evidence_seal = evidence_seal
        self.allow_unbound_fixture = allow_unbound_fixture
        self.registry = registry
        self.ledger = ledger
        self.evaluator = evaluator
        self.policy = policy
        self.cards = {card.source_id: card for card in source_cards}
        self.clock = clock
        self.gateway = gateway or RoleGateway()
        self.lifecycle = ModelLifecycle()
        self._decisions: dict[str, IntegrityDecision] = {}
        self._certificates: dict[str, AdmissionCertificate] = {}

    # ------------------------------------------------------------ registration
    def register_candidate(
        self, record: CandidateRecord, *, registered_at: dt.datetime
    ) -> RegisteredExperiment:
        """Register the hypothesis and experiment before any outcome access (spec 13.1).

        Re-registering identical content is idempotent; changed content under the
        same id fails (registered specifications are immutable).
        """
        registered_at = require_aware(registered_at, "registered_at")
        if registered_at > self.policy.frozen_at:
            fail(
                ErrorCode.ASOF_VIOLATION,
                "the promotion policy must be frozen after registration and before outcome access",
                "Register hypotheses, then freeze the policy, then evaluate.",
            )
        baselines = [
            "naive_persistence_baseline",
            "public_information_baseline",
        ]
        hypothesis = HypothesisSpec(
            owner=record.namespace,
            created_at=registered_at,
            hypothesis_id=record.hypothesis_id,
            mechanism_family=record.mechanism_family,
            research_mode=ResearchMode.CONTEMPORARY_RESEARCH,
            question=record.question,
            mechanism=record.mechanism,
            target_id=record.target_id,
            reference_model_id=baselines[0],
            feature_spec_ids=sorted(record.feature_hashes),
            applicability_spec_id=f"{record.approved_universe}_applicability_v1",
            baseline_experiment_ids=baselines,
            falsification=[
                "No paired improvement over the naive baseline in design and check partitions",
                "Effect vanishes under a one-date timing shift placebo",
                "Sign differs between design and check or between design halves",
            ],
            variant_budget=self.policy.variant_budget,
            experiment_spec_id=record.experiment_spec_id,
        )
        spec = ExperimentSpec(
            owner=record.namespace,
            created_at=registered_at,
            experiment_spec_id=record.experiment_spec_id,
            identity=ExperimentIdentity(
                hypothesis_id=record.hypothesis_id,
                variant_id="base",
                owner=record.namespace,
            ),
            claim=ExperimentClaim(
                primary_target_id=record.target_id,
                primary_metric="paired_mean_rank_ic_improvement",
                economic_effect=record.mechanism,
                declared_sign="+" if record.declared_sign > 0 else "-",
            ),
            data=ExperimentData(
                source_snapshot_ids=[
                    f"{sid}@{self.cards[sid].snapshot_id}"
                    if sid in self.cards
                    else f"{sid}@unresolved"
                    for sid in record.source_ids
                ],
                feature_spec_ids=sorted(record.feature_hashes),
                identity_graph_version="security_master_demo_v1"
                if record.evidence_mode.startswith("synthetic")
                else "unversioned",
            ),
            sample=ExperimentSample(
                universe_spec_id=record.approved_universe,
                eligibility_spec_id="tradable_and_priced_v1",
                applicability_spec_id=f"{record.approved_universe}_applicability_v1",
            ),
            time=ExperimentTime(
                training_cutoff=registered_at,
                label_maturity_rule="label resolves at the close of the next formation date",
                partitions={
                    "design": "evaluation partition 'design'",
                    "check": "referee-controlled partition 'check'",
                },
                purge_rule="no overlapping label windows between partitions",
                embargo_rule="one formation period between design and check",
            ),
            model=ExperimentModel(
                method_id=record.candidate_id,
                hyperparameter_space={},
                seed_policy="deterministic_no_random_seed",
            ),
            portfolio=ExperimentPortfolio(
                entry_rule="information through t; book at close of t+1",
                exit_rule="rebalanced at the next formation date",
                horizon_trading_days=record.horizon_trading_days,
                construction_policy_id="deterministic_selection_retention_band_v1",
                hedge_policy_id="none_permitted_stock_hedge_baseline_only",
            ),
            costs=ExperimentCosts(
                trade_cost_model_id="declared_in_run_config",
                borrow_model_id="declared_in_run_config",
                liquidity_model_id="declared_in_run_config",
                funding_model_id="cash_at_benchmark_rate",
            ),
            comparisons=ExperimentComparisons(
                baseline_experiment_ids=baselines,
                paired_comparison_policy_id="paired_mean_rank_ic_by_formation_date_v1",
                permitted_sensitivities=["timing_shift_placebo"],
            ),
            statistics=ExperimentStatistics(
                dependence_handling="one rank IC per formation date; stock-days are not treated as independent",
                uncertainty_method="descriptive standard error of the date-level mean IC (not HAC)",
                selection_accounting_policy_id=self.policy.policy_id,
            ),
            admission=ExperimentAdmission(
                min_coverage=self.policy.criteria().min_coverage,
                effect_requirement=self.policy.criteria().direction_and_economic_size,
                stability_requirement=self.policy.criteria().stability_requirement,
                implementation_requirement=self.policy.criteria().implementation_requirement,
            ),
            budget=ExperimentBudget(
                compute_budget="bounded local computation",
                provider_budget="none: no provider calls",
                llm_budget="none: no language model in the loop",
                evaluation_access_bounds="one check-partition access per candidate",
            ),
            promotion_conditions=self.policy.criteria().multiple_testing_treatment,
            failure_conditions="; ".join(hypothesis.falsification),
        )
        self._register_once("hypotheses", hypothesis)
        self._register_once("experiment_specs", spec)
        experiment = RegisteredExperiment(
            owner=record.namespace,
            created_at=registered_at,
            registration_id=record.registration_id,
            experiment_spec_id=record.experiment_spec_id,
            spec_content_hash=spec.digest(),
            primary_metric="paired_mean_rank_ic_improvement",
            baseline_experiment_ids=baselines,
            promotion_conditions=spec.promotion_conditions,
            failure_conditions=spec.failure_conditions,
            variant_budget=self.policy.variant_budget,
            registered_at=registered_at,
        )
        try:
            existing = self.ledger.registration(record.registration_id)
        except KeyError:
            self.ledger.register(experiment)
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.IDEA,
                    to_state=ModelState.REGISTERED,
                    authority="integrity",
                    evidence_ids=[record.registration_id],
                    reason="hypothesis and experiment registered before outcome access",
                    at=registered_at,
                )
            )
            return experiment
        if existing.spec_content_hash != experiment.spec_content_hash:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"registration {record.registration_id} exists with a different specification",
                "Registered specifications are immutable; amend with a new id.",
            )
        if self.lifecycle.state(record.model_id) == ModelState.IDEA:
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.IDEA,
                    to_state=ModelState.REGISTERED,
                    authority="integrity",
                    evidence_ids=[record.registration_id],
                    reason="restored immutable registration for identical retry",
                    at=registered_at,
                )
            )
        return existing

    def _register_once(self, table: str, record: Any) -> None:
        existing = self.registry.get(table, record.stable_id)
        if existing is None:
            self.registry.register(table, record)
        elif existing.get("content_hash") != record.sealed().content_hash:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"{table} record {record.stable_id} already registered with different content",
                "Registered specifications are immutable.",
            )

    # ---------------------------------------------------------------- evaluate
    def evaluate(
        self,
        record: CandidateRecord,
        design_observations: Sequence[ForecastOutcome] = (),
        check_observations: Sequence[ForecastOutcome] = (),
        *,
        ctx: RoleContext,
        evaluation_asof: dt.datetime,
        certificate_scope: str = SCOPE_FIXTURE,
        multiplicity: int = 1,
        sign_cost_bp: Optional[float] = None,
        evidence: Optional[EvidenceBundle] = None,
    ) -> IntegrityDecision:
        """Challenge and (maybe) promote one candidate. Needs the integrity role (protected partitions).

        Authority comes from a producer-sealed ``evidence`` bundle. Without one, only a FIXTURE-scope evaluation of caller
        observations is possible (labelled ``unbound_fixture_observations``; net-of-cost bridge unmeasured).
        """
        self.gateway.authorize(ctx, "integrity.evaluate_protected")
        self.gateway.authorize(ctx, "certificate.issue")
        evaluation_asof = require_aware(evaluation_asof, "evaluation_asof")
        if certificate_scope not in SCOPES:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"unknown certificate scope {certificate_scope!r}",
                f"Use one of {SCOPES}.",
            )
        design_observations, check_observations = self._bind_evidence(
            record,
            evidence,
            design_observations,
            check_observations,
            certificate_scope,
            evaluation_asof,
        )
        if multiplicity < 1:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                "multiplicity must be at least 1",
                "Count every compared candidate.",
            )
        if evaluation_asof < self.policy.frozen_at:
            fail(
                ErrorCode.ASOF_VIOLATION,
                "the evaluation clock precedes the frozen promotion policy",
                "Criteria are fixed before outcomes are opened.",
            )
        try:
            experiment = self.ledger.registration(record.registration_id)
        except KeyError:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"candidate {record.candidate_id} is not registered",
                "Register before outcome access (spec 13.1).",
            )
        policy = self.policy
        limitations = [
            "Descriptive date-level standard errors only; not dependence-adjusted (HAC/block bootstrap).",
            "A t-threshold with a multiplicity factor does not guarantee FDR control.",
            "Evaluated against realized next-period excess return (bridge level), not the declared belief target.",
        ]
        labels = {
            "evidence_mode": record.evidence_mode,
            "certificate_scope": certificate_scope,
            "evidence_binding": f"producer_sealed:{evidence.bundle_id}"
            if evidence
            else "unbound_fixture_observations",
        }
        if evidence is None:
            limitations.append(
                "Caller-supplied observations (fixture compatibility): net-of-cost tail-portfolio bridge was NOT measured."
            )
        if certificate_scope == SCOPE_FIXTURE:
            limitations += [
                FIXTURE_LIMITATION,
                f"{SYNTHETIC_LABEL}: planted-signal data; no result here is investment evidence.",
            ]
            labels["synthetic"] = SYNTHETIC_LABEL

        def blocked(reason: str, **extra: Any) -> IntegrityDecision:
            decision = IntegrityDecision(
                candidate_id=record.candidate_id,
                charter=record.charter,
                outcome=Outcome.BLOCKED,
                scope=certificate_scope,
                reasons=[reason],
                labels=labels,
                evidence={"registration_id": record.registration_id},
                lifecycle=self._lifecycle_dicts(record.model_id),
                limitations=limitations,
                trial_counts=self._trial_counts(),
                **extra,
            )
            self._decisions[record.candidate_id] = decision
            return decision

        if certificate_scope == SCOPE_FORMATION and (
            record.evidence_mode.startswith("synthetic")
            or any(
                self.cards.get(sid)
                and self.cards[sid].dataset_id == SYNTHETIC_DATASET_ID
                for sid in record.source_ids
            )
        ):
            return blocked(
                "synthetic evidence cannot issue formation certificates"
            )

        # source gate: never relax eligibility
        problems: list[str] = []
        ineligible: list[str] = []
        for source_id in record.source_ids:
            card = self.cards.get(source_id)
            if card is None:
                problems.append(f"no source card for {source_id}")
            elif (
                certificate_scope == SCOPE_FIXTURE
                and card.dataset_id != SYNTHETIC_DATASET_ID
                and not (
                    evidence is not None
                    and record.evidence_mode.startswith("synthetic")
                    and dict(evidence.provenance).get("dataset_class")
                    == "fixture"
                )
            ):
                problems.append(
                    f"fixture scope is limited to synthetic sources, but {source_id} is dataset {card.dataset_id}"
                )
            elif card.broker_rejects_formation():
                ineligible.append(source_id)
        if certificate_scope == SCOPE_FORMATION and ineligible:
            problems.append(
                "sources not formation-eligible: "
                + ", ".join(sorted(ineligible))
            )
        if problems:
            return blocked("; ".join(problems))
        if ineligible:
            limitations.append(
                "sources not formation-eligible (research use only): "
                + ", ".join(sorted(ineligible))
            )

        # coverage: cannot evaluate without enough dated, populated cross-sections
        design_counts, check_counts = (
            _date_counts(design_observations),
            _date_counts(check_observations),
        )
        if (
            len(design_counts) < policy.min_dates_design
            or len(check_counts) < policy.min_dates_check
        ):
            return blocked(
                f"insufficient dated coverage: design {len(design_counts)}/{policy.min_dates_design} dates, "
                f"check {len(check_counts)}/{policy.min_dates_check} dates"
            )
        thin = [
            d
            for d, n in {**design_counts, **check_counts}.items()
            if n < policy.min_obs_per_date
        ]
        if thin:
            return blocked(
                f"{len(thin)} formation date(s) have fewer than {policy.min_obs_per_date} names (first: {sorted(thin)[0]})"
            )
        if max(o.formation_at for o in design_observations) >= min(
            o.formation_at for o in check_observations
        ):
            return blocked(
                "design observations overlap or follow the check partition; partitions must be chronological"
            )

        spec_hash = experiment.spec_content_hash
        design_plan, check_plan = self._plans(
            record, experiment, certificate_scope != SCOPE_FORMATION
        )
        digests = {
            "design_plan": self.evaluator.register(design_plan),
            "check_plan": self.evaluator.register(check_plan),
        }
        self.lifecycle.transition(
            StateTransition(
                model_id=record.model_id,
                from_state=ModelState.REGISTERED,
                to_state=ModelState.DATA_READY,
                authority="integrity",
                evidence_ids=[digests["design_plan"][:16]],
                reason="source gate and coverage checks passed",
                at=evaluation_asof,
            )
        )

        def scored(
            variant: str,
            partition_plan: ComparisonPlan,
            obs: Sequence[ForecastOutcome],
            partitions: tuple[str, ...],
        ) -> dict[str, Any]:
            trial = TrialRecord(
                kind="scored_variant",
                spec_hash=spec_hash,
                detail=f"{record.candidate_id}:{variant}",
                counted_at=evaluation_asof,
            )
            if hasattr(self.ledger, "reserve_trial"):
                self.ledger.reserve_trial(record.registration_id, trial)
            else:
                self.ledger.enforce_variant_budget(record.registration_id)
                self.ledger.record_trial(trial)
            return self.evaluator.evaluate(
                partition_plan.experiment_id,
                variant,
                list(obs),
                owner=record.namespace,
                evaluation_asof=evaluation_asof,
                source_cards=list(self.cards.values()),
                allowed_partitions=partitions,
            )

        design = scored(
            "design_base", design_plan, design_observations, ("design",)
        )
        design_placebo = scored(
            "design_timing_shift_placebo",
            design_plan,
            timing_shift_placebo(design_observations),
            ("design",),
        )
        self.lifecycle.transition(
            StateTransition(
                model_id=record.model_id,
                from_state=ModelState.DATA_READY,
                to_state=ModelState.DEVELOPMENT,
                authority="integrity",
                evidence_ids=[design["input_hash"][:16]],
                reason="design-partition comparison completed",
                at=evaluation_asof,
            )
        )
        # protected partition: recorded referee access, then the limited registered evaluation
        self.ledger.record_outcome_access(
            access_id=f"acc_{record.candidate_id}_check_{uuid.uuid4().hex[:12]}",
            partition="check",
            actor="integrity",
            purpose=f"registered limited evaluation of {record.candidate_id}",
            record_count=len(check_observations),
            authorized=True,
        )
        self.ledger.record_trial(
            TrialRecord(
                kind="check_window_access",
                spec_hash=spec_hash,
                detail=f"{record.candidate_id}: {len(check_observations)} observations",
                counted_at=evaluation_asof,
            )
        )
        check = scored(
            "check_base", check_plan, check_observations, ("design", "check")
        )
        check_placebo = scored(
            "check_timing_shift_placebo",
            check_plan,
            timing_shift_placebo(check_observations),
            ("design", "check"),
        )
        self.lifecycle.transition(
            StateTransition(
                model_id=record.model_id,
                from_state=ModelState.DEVELOPMENT,
                to_state=ModelState.CHALLENGED,
                authority="integrity",
                evidence_ids=[
                    check["input_hash"][:16],
                    check_placebo["input_hash"][:16],
                ],
                reason="placebo and check-partition challenges completed",
                at=evaluation_asof,
            )
        )

        sign = float(record.declared_sign)
        design_ic, check_ic = (
            _finite_series(design, "candidate_ic"),
            _finite_series(check, "candidate_ic"),
        )
        if (
            len(design_ic) < policy.min_dates_design
            or len(check_ic) < policy.min_dates_check
        ):
            return blocked(
                "per-date rank ICs are undefined for too many formation dates"
            )
        half = len(design_ic) // 2
        design_baseline = _finite_series(design, "baseline_ic")
        check_baseline = _finite_series(check, "baseline_ic")
        bridge_check = _finite_series(check, "bridge_ic")
        bridge_design = _finite_series(design, "bridge_ic")
        placebo_design = _finite_series(design_placebo, "candidate_ic")
        placebo_check = _finite_series(check_placebo, "candidate_ic")
        all_ic = design_ic + check_ic
        ic_sd = _sd(all_ic)
        t_stat = (
            _mean(all_ic) / (ic_sd / math.sqrt(len(all_ic)))
            if ic_sd > 0
            else (math.inf if _mean(all_ic) > 0 else 0.0)
        )
        # signed t so a wrong-sign candidate cannot pass on magnitude alone
        t_signed = (
            sign * t_stat
            if math.isfinite(t_stat)
            else (1e9 if sign * _mean(all_ic) > 0 else -1e9)
        )
        z = statistics.NormalDist().inv_cdf(
            1.0 - policy.family_alpha / (2.0 * multiplicity)
        )
        t_required = max(policy.min_t_stat, z)
        d_mean, c_mean = _mean(design_ic), _mean(check_ic)
        d_improve = design.get("paired_mean_ic_improvement")
        c_improve = check.get("paired_mean_ic_improvement")
        placebo_d = _mean(placebo_design) if placebo_design else None
        placebo_c = _mean(placebo_check) if placebo_check else None
        bridge_c = _mean(bridge_check) if bridge_check else None
        bridge_d = _mean(bridge_design) if bridge_design else None
        halves = (_mean(design_ic[:half]), _mean(design_ic[half:]))
        min_stab = policy.min_mean_ic / 2.0
        scored_variants = (
            policy.variant_budget
            - self.ledger.variant_budget_remaining(record.registration_id)
        )
        # improvement over the baseline is read in the direction of the declared sign
        improvement_met = (
            d_improve is not None
            and c_improve is not None
            and sign * d_improve >= policy.min_improvement
            and sign * c_improve >= policy.min_improvement
        )

        net_metrics, net_met = self._net_bridge(evidence)
        criteria = {
            "coverage_met": True,
            "improvement_met": improvement_met,
            "direction_met": sign * d_mean >= policy.min_mean_ic
            and sign * c_mean >= policy.min_mean_ic,
            "stability_met": (
                sign * halves[0] >= min_stab
                and sign * halves[1] >= min_stab
                and sign * c_mean >= min_stab
                and placebo_d is not None
                and placebo_c is not None
                and abs(placebo_d) <= policy.max_placebo_fraction * abs(d_mean)
                and abs(placebo_c) <= policy.max_placebo_fraction * abs(c_mean)
            ),
            "implementation_met": bridge_c is not None
            and bridge_c * sign >= policy.min_bridge_ic,
            "net_bridge_met": net_met,
            "uncertainty_met": t_signed >= policy.min_t_stat,
            "multiple_testing_met": t_signed >= t_required
            and scored_variants <= policy.variant_budget,
        }

        def fin(x: Optional[float]) -> Optional[float]:
            return None if x is None or not math.isfinite(x) else float(x)

        metrics: dict[str, Optional[float]] = {
            "design_mean_ic": fin(d_mean),
            "check_mean_ic": fin(c_mean),
            "design_first_half_ic": fin(halves[0]),
            "design_second_half_ic": fin(halves[1]),
            "design_baseline_ic": fin(_mean(design_baseline))
            if design_baseline
            else None,
            "check_baseline_ic": fin(_mean(check_baseline))
            if check_baseline
            else None,
            "design_paired_improvement": fin(d_improve),
            "check_paired_improvement": fin(c_improve),
            "design_placebo_ic": fin(placebo_d),
            "check_placebo_ic": fin(placebo_c),
            "design_bridge_ic": fin(bridge_d),
            "check_bridge_ic": fin(bridge_c),
            "ic_sd_across_dates": fin(ic_sd),
            "t_stat_signed": fin(t_signed),
            "t_required": fin(t_required),
            "n_dates_design": float(len(design_ic)),
            "n_dates_check": float(len(check_ic)),
            "scored_variants": float(scored_variants),
            "multiplicity": float(multiplicity),
            **net_metrics,
        }
        reasons: list[str] = []
        for name, ok in criteria.items():
            if not ok:
                reasons.append(f"criterion not met: {name}")
        if not reasons:
            reasons.append(
                "all registered promotion criteria were measured and met under the frozen policy"
            )

        if all(criteria.values()):
            outcome = Outcome.ADMITTED
        elif not criteria["direction_met"] or not criteria["improvement_met"]:
            outcome = Outcome.REJECTED
        else:
            outcome = Outcome.SHADOW
        if (
            ineligible
            and certificate_scope != SCOPE_FIXTURE
            and outcome != Outcome.REJECTED
        ):
            outcome = Outcome.DESCRIPTIVE_ONLY
            reasons.append(
                "sources are not formation-eligible; evidence is descriptive only"
            )

        evidence = {
            "registration_id": record.registration_id,
            "design_plan_digest": digests["design_plan"],
            "check_plan_digest": digests["check_plan"],
            "design_result_input_hash": design["input_hash"],
            "check_result_input_hash": check["input_hash"],
            "design_placebo_input_hash": design_placebo["input_hash"],
            "check_placebo_input_hash": check_placebo["input_hash"],
            **({"evidence_bundle_id": evidence.bundle_id} if evidence else {}),
        }
        certificate_id: Optional[str] = None
        if outcome == Outcome.ADMITTED:
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.CHALLENGED,
                    to_state=ModelState.CHECK_ELIGIBLE,
                    authority="integrity",
                    evidence_ids=[evidence["check_result_input_hash"][:16]],
                    reason="check-partition criteria met",
                    at=evaluation_asof,
                )
            )
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.CHECK_ELIGIBLE,
                    to_state=ModelState.SHADOW,
                    authority="integrity",
                    evidence_ids=[record.registration_id],
                    reason="shadow forecasts publish separately before admission",
                    at=evaluation_asof,
                )
            )
            certificate = self._issue_certificate(
                record,
                evidence,
                certificate_scope,
                limitations,
                evaluation_asof,
                criteria,
            )
            certificate_id = certificate.certificate_id
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.SHADOW,
                    to_state=ModelState.ADMITTED,
                    authority="integrity",
                    evidence_ids=[certificate_id],
                    reason="admission certificate issued",
                    at=evaluation_asof,
                )
            )
            self._certificates[record.candidate_id] = certificate
        elif outcome == Outcome.SHADOW:
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.CHALLENGED,
                    to_state=ModelState.CHECK_ELIGIBLE,
                    authority="integrity",
                    evidence_ids=[evidence["check_result_input_hash"][:16]],
                    reason="direction and improvement hold; other criteria unproven",
                    at=evaluation_asof,
                )
            )
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.CHECK_ELIGIBLE,
                    to_state=ModelState.SHADOW,
                    authority="integrity",
                    evidence_ids=[record.registration_id],
                    reason="shadow only: tracked, never traded",
                    at=evaluation_asof,
                )
            )
        elif outcome == Outcome.DESCRIPTIVE_ONLY:
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.CHALLENGED,
                    to_state=ModelState.DESCRIPTIVE_ONLY,
                    authority="integrity",
                    evidence_ids=[record.registration_id],
                    reason="non-formation sources: descriptive only",
                    at=evaluation_asof,
                )
            )
        else:
            self.lifecycle.transition(
                StateTransition(
                    model_id=record.model_id,
                    from_state=ModelState.CHALLENGED,
                    to_state=ModelState.REJECTED,
                    authority="integrity",
                    evidence_ids=[evidence["check_result_input_hash"][:16]],
                    reason="; ".join(reasons)[:300],
                    at=evaluation_asof,
                )
            )
        decision = IntegrityDecision(
            candidate_id=record.candidate_id,
            charter=record.charter,
            outcome=outcome,
            scope=certificate_scope,
            reasons=reasons,
            criteria=criteria,
            metrics=metrics,
            labels=labels,
            evidence=evidence,
            certificate_id=certificate_id,
            lifecycle=self._lifecycle_dicts(record.model_id),
            limitations=limitations,
            trial_counts=self._trial_counts(),
            uncertainty={
                "ic_sd_across_dates": fin(ic_sd),
                "n_dates": float(len(all_ic)),
                "mean_ic": fin(_mean(all_ic)),
            },
        )
        self._decisions[record.candidate_id] = decision
        return decision

    # ------------------------------------------------------------------ helpers
    def _bind_evidence(
        self,
        record: CandidateRecord,
        evidence: Optional[EvidenceBundle],
        design: Sequence[ForecastOutcome],
        check: Sequence[ForecastOutcome],
        scope: str,
        asof: dt.datetime,
    ) -> tuple[list[ForecastOutcome], list[ForecastOutcome]]:
        """Observations may only come from a verified bundle bound to this exact candidate, model and scope (or fixture compat)."""

        def refuse(summary: str) -> None:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                summary,
                "Protected evaluation needs producer-sealed evidence bound to this candidate.",
            )

        if evidence is None:
            if scope != SCOPE_FIXTURE or not self.allow_unbound_fixture:
                refuse(
                    "unbound observations can only support an explicitly allowed FIXTURE evaluation, never research or formation"
                )
            return list(design), list(check)
        if self.evidence_seal is None or not self.evidence_seal.verify(
            evidence
        ):
            refuse("evidence bundle is unknown, forged or tampered")
        if (
            evidence.candidate_id != record.candidate_id
            or evidence.model_hash != record.model_hash
            or evidence.record_digest
            != canonical_digest(record.model_dump(mode="json"))
            or evidence.scope != scope
        ):
            refuse(
                "evidence bundle is bound to a different candidate record, model or scope"
            )
        if evidence.issued_at > asof:
            refuse("evidence bundle was issued after the evaluation clock")
        for supplied, sealed in (
            (design, evidence.design_observations),
            (check, evidence.check_observations),
        ):
            if supplied and _obs_digest(supplied) != _obs_digest(sealed):
                refuse("supplied observations differ from the sealed evidence")
        return list(evidence.design_observations), list(
            evidence.check_observations
        )

    def _net_bridge(
        self, evidence: Optional[EvidenceBundle]
    ) -> tuple[dict[str, Optional[float]], bool]:
        """Measured tail-portfolio net spread after costs (design and check). Unbound fixture compat reports it as unmeasured."""
        if evidence is None:
            return (
                {},
                True,
            )  # fixture compatibility only; decision limitations state the bridge was not measured
        policy = self.policy
        cost = max(evidence.cost_bps_one_way, policy.cost_bps_floor_one_way)
        series = {
            part: net_spread_series(
                obs,
                port,
                bridge_sign=evidence.bridge_sign,
                tail_fraction=evidence.tail_fraction,
                cost_bps_one_way=cost,
            )
            for part, obs, port in (
                (
                    "design",
                    evidence.design_observations,
                    evidence.design_portfolio,
                ),
                (
                    "check",
                    evidence.check_observations,
                    evidence.check_portfolio,
                ),
            )
        }
        metrics: dict[str, Optional[float]] = {
            "cost_bps_one_way_effective": cost
        }
        met = True
        for part, minimum in (
            ("design", policy.min_dates_design),
            ("check", policy.min_dates_check),
        ):
            rows = series[part]
            if not rows or len(rows) < minimum:
                metrics[f"{part}_net_spread"] = None
                met = False
                continue
            for key in ("gross", "turnover", "net"):
                metrics[
                    f"{part}_{key}_spread"
                    if key != "turnover"
                    else f"{part}_turnover"
                ] = _mean([r[key] for r in rows])
            met = met and metrics[f"{part}_net_spread"] > policy.min_net_spread
        return metrics, met

    def certificate(self, candidate_id: str) -> AdmissionCertificate:
        try:
            return self._certificates[candidate_id]
        except KeyError:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"no certificate was issued for {candidate_id}",
                "Only an ADMITTED candidate holds a certificate.",
            )

    def restore_admission(
        self, decision: IntegrityDecision, *, model_hash: str
    ) -> AdmissionCertificate:
        """Trusted bridge restart: recover an existing sealed certificate, never issue one."""
        raw = (
            self.registry.get("admission_certificates", decision.certificate_id)
            if decision.admitted and decision.certificate_id
            else None
        )
        cert = AdmissionCertificate.model_validate(raw) if raw else None
        if (
            cert is None
            or not cert.verify_seal()
            or cert.model_hash != model_hash
        ):
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                "stored admission certificate is absent or changed",
                "Resume requires the original sealed integrity certificate.",
            )
        self._certificates[decision.candidate_id] = cert
        self._decisions[decision.candidate_id] = decision
        return cert

    def decision(self, candidate_id: str) -> Optional[IntegrityDecision]:
        return self._decisions.get(candidate_id)

    def decisions(self) -> list[IntegrityDecision]:
        return [self._decisions[k] for k in sorted(self._decisions)]

    def _plans(
        self,
        record: CandidateRecord,
        experiment: RegisteredExperiment,
        research_only: bool,
    ) -> tuple[ComparisonPlan, ComparisonPlan]:
        common: dict[str, Any] = dict(
            owner=record.namespace,
            hypothesis_id=record.hypothesis_id,
            registered_at=experiment.registered_at,
            target_id=record.target_id,
            source_ids=list(record.source_ids),
            variant_budget=2,
            research_only=research_only,
            min_dates=2,
        )
        return (
            ComparisonPlan(
                experiment_id=f"{record.experiment_spec_id}_design",
                partition="design",
                **common,
            ),
            ComparisonPlan(
                experiment_id=f"{record.experiment_spec_id}_check",
                partition="check",
                **common,
            ),
        )

    def _trial_counts(self) -> dict[str, int]:
        counts = {kind: self.ledger.trial_count(kind) for kind in TRIAL_KINDS}
        return {kind: n for kind, n in sorted(counts.items()) if n}

    def _lifecycle_dicts(self, model_id: str) -> list[dict[str, str]]:
        return [
            {
                "from": t.from_state.value,
                "to": t.to_state.value,
                "authority": t.authority,
                "reason": t.reason,
                "at": t.at.isoformat(),
            }
            for t in self.lifecycle.history(model_id)
        ]

    def _issue_certificate(
        self,
        record: CandidateRecord,
        evidence: Mapping[str, str],
        scope: str,
        limitations: Sequence[str],
        issued_at: dt.datetime,
        criteria: Mapping[str, bool],
    ) -> AdmissionCertificate:
        outcome = PromotionOutcome(
            registration_id=record.registration_id,
            evaluated_at=issued_at,
            criteria=self.policy.criteria(),
            coverage_met=criteria["coverage_met"],
            improvement_met=criteria["improvement_met"],
            direction_met=criteria["direction_met"],
            stability_met=criteria["stability_met"],
            implementation_met=criteria["implementation_met"]
            and criteria["net_bridge_met"],
            uncertainty_met=criteria["uncertainty_met"],
            multiple_testing_met=criteria["multiple_testing_met"],
        )
        mode = {
            SCOPE_FIXTURE: "fixture_synthetic_demo",
            SCOPE_RESEARCH: "research_design_only",
            SCOPE_FORMATION: "formation_replay",
        }[scope]
        certificate = outcome.certificate_or_raise(
            model_id=record.model_id,
            model_hash=record.model_hash,
            scope=CertificateScope(
                environment={
                    "fixture": "test",
                    "research": "shadow",
                    "formation": "formation",
                }[scope],
                issued_at=issued_at,
                valid_from=issued_at,
                valid_until=issued_at + dt.timedelta(days=366),
                allowed_source_ids=record.source_ids,
                allowed_source_snapshots={
                    s: [self.cards[s].snapshot_id] for s in record.source_ids
                },
                universe_ids=[record.approved_universe],
                namespaces=[record.namespace] if scope == "formation" else None,
                issuer="integrity",
            ),
            feature_hashes=dict(record.feature_hashes),
            data_hashes=dict(record.data_hashes),
            approved_universe=record.approved_universe,
            target_id=record.target_id,
            horizon_trading_days=record.horizon_trading_days,
            effective_start=issued_at,
            approved_mode=mode,
            risk_restrictions=[
                "internal absolute net exposure limit applies",
                "independent risk review required before execution",
            ],
            capacity_restrictions=[
                "participation capped by the execution policy; capacity untested beyond declared liquidity"
            ],
            expiry_policy=self.policy.expiry_policy,
            known_limitations=list(limitations),
            promotion_evidence_ids=[
                f"{k}:{v[:16]}" if len(v) > 16 else f"{k}:{v}"
                for k, v in sorted(evidence.items())
            ],
        ).sealed()
        if (
            self.registry.get(
                "admission_certificates", certificate.certificate_id
            )
            is None
        ):
            self.registry.register("admission_certificates", certificate)
        return certificate


# ------------------------------------------------------------ research mode


def _clean_metrics(raw: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                out[str(key)] = number
    return out


def normalize_campaign(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize a charter-campaign JSON into candidate summaries (fail closed on a malformed shape)."""
    if not isinstance(campaign, Mapping) or not isinstance(
        campaign.get("candidates"), list
    ):
        fail(
            ErrorCode.SCHEMA_MISMATCH,
            "campaign JSON has no 'candidates' list",
            "The research engine must return its documented schema.",
        )
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(campaign["candidates"]):
        if not isinstance(raw, Mapping):
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"campaign candidate #{index} is not an object",
                "Candidates are JSON objects.",
            )
        hypothesis = raw.get("hypothesis")
        hypothesis_id = (
            hypothesis.get("hypothesis_id")
            if isinstance(hypothesis, Mapping)
            else hypothesis
        )
        candidate_id = str(
            raw.get("candidate_id")
            or raw.get("id")
            or raw.get("hypothesis_id")
            or hypothesis_id
            or f"candidate_{index}"
        )
        if candidate_id in seen:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"duplicate campaign candidate id {candidate_id}",
                "Candidate ids are unique.",
            )
        seen.add(candidate_id)
        limitations = raw.get("limitations") or []
        summaries.append(
            {
                "candidate_id": candidate_id,
                "charter": str(raw.get("charter") or "unknown"),
                "hypothesis": hypothesis if hypothesis is not None else None,
                "spec": raw.get("spec"),
                "metrics": _clean_metrics(raw.get("metrics")),
                "status": str(raw.get("status") or "unspecified"),
                "limitations": [str(x) for x in limitations]
                if isinstance(limitations, list)
                else [str(limitations)],
                "source_ids": [
                    str(x)
                    for x in raw.get(
                        "source_ids", campaign.get("source_ids", [])
                    )
                ],
            }
        )
    return summaries


_IMPROVEMENT_KEYS = (
    "belief_ic_improvement",
    "paired_mean_ic_improvement",
    "ic_improvement",
    "improvement",
    "paired_improvement",
)


def classify_research_candidate(
    summary: Mapping[str, Any],
    policy: IntegrityPolicy,
    source_cards: Optional[Mapping[str, SourceCard]] = None,
) -> IntegrityDecision:
    """Classify a design-partition research result. Never returns ADMITTED.

    Admission needs controlled check-partition evidence from ``IntegrityService.evaluate``;
    a campaign's own metrics are design-partition research and can at best earn SHADOW.
    """
    metrics = dict(summary.get("metrics") or {})
    status = str(summary.get("status") or "").lower()
    reasons = [
        "design-partition research evidence only; no controlled check or prospective evaluation",
        "admission requires IntegrityService.evaluate with protected check access and a frozen policy",
    ]
    improvement = next(
        (metrics[k] for k in _IMPROVEMENT_KEYS if k in metrics), None
    )
    ineligible: list[str] = []
    cards = source_cards or {}
    for source_id in summary.get("source_ids", []):
        card = cards.get(source_id)
        if card is None or card.broker_rejects_formation():
            ineligible.append(source_id)
    if "block" in status:
        outcome = Outcome.BLOCKED
        reasons.append(f"campaign reported status {summary.get('status')!r}")
    elif "reject" in status or "fail" in status:
        outcome = Outcome.REJECTED
        reasons.append(f"campaign reported status {summary.get('status')!r}")
    elif improvement is None:
        outcome = Outcome.DESCRIPTIVE_ONLY
        reasons.append(
            "no comparable measured improvement over a baseline was supplied"
        )
    elif improvement < policy.min_improvement:
        outcome = Outcome.REJECTED
        reasons.append(
            f"measured improvement {improvement:.4f} is below the frozen minimum {policy.min_improvement}"
        )
    elif ineligible:
        outcome = Outcome.DESCRIPTIVE_ONLY
        reasons.append(
            "sources not formation-eligible or without a card: "
            + ", ".join(sorted(ineligible))
        )
    else:
        outcome = Outcome.SHADOW
        reasons.append(
            "design-partition improvement meets the frozen minimum; tracked as shadow, never traded"
        )
    return IntegrityDecision(
        candidate_id=str(summary["candidate_id"]),
        charter=str(summary.get("charter") or "unknown"),
        outcome=outcome,
        scope=SCOPE_RESEARCH,
        reasons=reasons,
        criteria={},
        metrics={k: float(v) for k, v in metrics.items()},
        labels={
            "evidence_mode": "design_partition_research",
            "admitted": "false",
        },
        evidence={"campaign_status": str(summary.get("status"))},
        certificate_id=None,
        lifecycle=[],
        limitations=list(summary.get("limitations", []))
        + ["Not admitted: no check-partition or prospective evidence exists."],
        trial_counts={},
        uncertainty={},
    )
