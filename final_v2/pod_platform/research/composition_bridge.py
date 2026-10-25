"""Trusted exact-graph producer for the existing sealed integrity service.

Native composition probes do not authorize this path. The host separately pins
financial measurements, partition snapshots, costs and eligible short names.
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from ..application.integrity import (
    CandidateRecord,
    IntegrityDecision,
    net_spread_series,
)
from ..application.publication import ModelVersionRecord
from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.forecasts import Forecast, ReturnForecast, TargetPrediction
from ..pods.charters import pod_spec
from .bridge_economics import calibrate_return_bridge
from .builtins import stats
from .builtins.timeutil import month_end_known_at, month_index
from .composition import FeatureObservation, evaluate_graph, observation_digest
from .evaluation import ForecastOutcome
from .jobs import open_sqlite


class FinancialReview(StrictModel):
    composition_id: str
    output_node: str
    baseline_node: str
    mechanism_family: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    return_source_id: str
    return_metric: str
    return_horizon_months: int = Field(default=1, ge=1, le=12)
    horizon_trading_days: int = Field(default=21, ge=1, le=252)
    design_end: dt.date
    check_dates: list[dt.datetime] = Field(min_length=2, max_length=36)
    design_snapshots: dict[str, str]
    check_snapshots: dict[str, str]
    short_eligible_security_ids: list[str] = Field(min_length=3, max_length=500)
    cost_bps_one_way: float = Field(ge=0, le=1000, allow_inf_nan=False)
    cost_source: str = Field(min_length=10, max_length=500)
    tail_fraction: float = Field(default=0.2, ge=0.05, le=0.34)
    bridge_sign: Literal[-1, 1] = 1
    certificate_scope: Literal["fixture", "research", "formation"] = "research"
    score_transform: Literal["centered_rank"] = "centered_rank"
    max_feature_age_days: int = Field(default=62, ge=1, le=366)

    @model_validator(mode="after")
    def clocks(self):
        for day in self.check_dates:
            require_aware(day, "check date")
            if day.date() <= self.design_end:
                raise ValueError("check must follow the frozen design window")
        if len(set(self.check_dates)) != len(self.check_dates):
            raise ValueError("duplicate check dates")
        for snapshots in (self.design_snapshots, self.check_snapshots):
            if not snapshots or any(
                len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                for v in snapshots.values()
            ):
                raise ValueError("actual source snapshot digests required")
        return self


class CompositionBridge:
    """Freeze host-reviewed graph and spend each protected access at most once."""

    def __init__(
        self, compositions, integrity, *, ctx, clock, max_family_accesses=1
    ):
        integrity.gateway.authorize(ctx, "integrity.evaluate_protected")
        integrity.gateway.authorize(ctx, "certificate.issue")
        if (
            integrity.evidence_seal is None
            or integrity.allow_unbound_fixture
            or not 1 <= max_family_accesses <= 3
        ):
            raise ValueError(
                "sealed integrity authority and finite family budget required"
            )
        self.compositions, self.integrity, self.ctx, self.clock = (
            compositions,
            integrity,
            ctx,
            clock,
        )
        self.max_family_accesses = max_family_accesses
        compositions._put(
            "financial_policy",
            "financial_policy",
            {
                "max_family_accesses": max_family_accesses,
                "policy": integrity.policy.model_dump(mode="json"),
            },
        )

    def freeze(self, review):
        """Persist exact financial mapping and registration before either loader runs."""
        review = FinancialReview.model_validate(review)
        plan = self.compositions.integrated_plan(review.composition_id)
        grant = plan["grant"]
        spec = plan["spec"]
        if (
            review.output_node not in spec["outputs"]
            or review.baseline_node not in spec["baselines"]
        ):
            raise ValueError(
                "financial mapping must select the frozen graph outputs and baselines"
            )
        if review.return_metric not in grant["source_metrics"].get(
            review.return_source_id, []
        ):
            raise ValueError(
                "return measurement is not explicitly host granted"
            )
        sources = set(review.design_snapshots)
        required = {
            n["recipe"]["source_id"] for n in spec["nodes"] if n.get("recipe")
        }
        required |= {
            grant["targets"][spec["target_id"]]["source_id"],
            review.return_source_id,
        }
        if (
            not required <= sources
            or sources != set(review.check_snapshots)
            or sources - set(grant["source_metrics"])
        ):
            raise ValueError(
                "financial snapshot mappings omit intrinsic measurements"
            )
        target = grant["targets"][spec["target_id"]]
        last = max(
            dt.datetime.fromisoformat(d) for d in grant["formation_dates"]
        )
        if last.date() > review.design_end or month_end_known_at(
            month_index(last.date())
            + max(target["horizon_months"], review.return_horizon_months)
        ) >= min(review.check_dates):
            raise ValueError(
                "design native/return outcomes overlap the independent check"
            )
        if not set(review.short_eligible_security_ids) <= set(
            grant["security_ids"]
        ):
            raise ValueError(
                "short eligibility must belong to the reviewed universe"
            )
        frozen = {
            "composition_id": review.composition_id,
            "review": review.model_dump(mode="json"),
            "graph": spec["nodes"],
            "target": target,
            "code_digest": self.compositions.code_digest,
            "policy_digest": canonical_digest(
                self.integrity.policy.model_dump(mode="json")
            ),
        }
        model_hash = canonical_digest(frozen)
        identifier = (
            "composed_"
            + canonical_digest([self.compositions.namespace, model_hash])[:24]
        )
        record = CandidateRecord(
            candidate_id=identifier,
            charter=pod_spec(self.compositions.namespace).charter.value,
            namespace=self.compositions.namespace,
            hypothesis_id=review.composition_id,
            experiment_spec_id=identifier,
            mechanism_family=review.mechanism_family,
            model_id=identifier + "_model",
            model_hash=model_hash,
            target_id=spec["target_id"],
            horizon_trading_days=review.horizon_trading_days,
            source_ids=sorted(sources),
            question=spec["question"],
            mechanism=spec["mechanism"],
            declared_sign=review.bridge_sign,
            approved_universe="universe_"
            + canonical_digest(grant["security_ids"])[:16],
            feature_hashes={
                n["node_id"]: canonical_digest(n) for n in spec["nodes"]
            },
            data_hashes={
                **{
                    "design_" + s: h for s, h in review.design_snapshots.items()
                },
                **{"check_" + s: h for s, h in review.check_snapshots.items()},
            },
            evidence_mode="synthetic_fixture"
            if review.certificate_scope == "fixture"
            else "exact_composition_graph",
        )
        at = dt.datetime.fromisoformat(plan["registered_at"])
        self.integrity.register_candidate(record, registered_at=at)
        return self.compositions._put(
            "financial:" + identifier,
            "financial",
            {
                **frozen,
                "candidate_id": identifier,
                "record": record.model_dump(mode="json"),
                "registered_at": at.isoformat(),
            },
        )

    def _outcomes(self, frozen, observations, dates):
        plan = self.compositions.integrated_plan(frozen["composition_id"])
        grant = plan["grant"]
        review = frozen["review"]
        now = require_aware(self.clock(), "clock")
        labels = {}
        for observation in observations:
            if (
                max(observation.available_at, observation.version_known_at)
                > now
            ):
                continue
            key = (
                observation.source_id,
                observation.metric,
                observation.security_id,
                month_index(observation.period_end.date()),
            )
            old = labels.get(key)
            if (
                old
                and old.version_known_at == observation.version_known_at
                and old.value != observation.value
            ):
                raise ValueError("ambiguous native or return outcome version")
            if (
                old is None
                or observation.version_known_at > old.version_known_at
            ):
                labels[key] = observation
        out = []
        target = frozen["target"]
        for day in dates:
            values = evaluate_graph(
                frozen["graph"], observations, day, grant["security_ids"]
            )
            for node in (review["output_node"], review["baseline_node"]):
                values[node] = self._scores(values[node])
            month = month_index(day.date())
            for security in grant["security_ids"]:
                prediction = values[review["output_node"]][security]
                baseline = values[review["baseline_node"]][security]
                label = labels.get(
                    (
                        target["source_id"],
                        target["metric"],
                        security,
                        month + target["horizon_months"],
                    )
                )
                ret = labels.get(
                    (
                        review["return_source_id"],
                        review["return_metric"],
                        security,
                        month + review["return_horizon_months"],
                    )
                )
                if (
                    prediction is None
                    or baseline is None
                    or label is None
                    or ret is None
                ):
                    continue
                resolved = max(
                    label.available_at,
                    label.version_known_at,
                    ret.available_at,
                    ret.version_known_at,
                    month_end_known_at(
                        month + target["horizon_months"],
                        target["resolve_lag_days"],
                    ),
                    month_end_known_at(month + review["return_horizon_months"]),
                )
                if resolved > now:
                    continue
                out.append(
                    ForecastOutcome(
                        security_id=security,
                        formation_at=day,
                        feature_available_at=day,
                        label_resolves_at=resolved,
                        prediction=prediction,
                        baseline_prediction=baseline,
                        target=label.value,
                        net_forward_return=review["bridge_sign"] * ret.value,
                    )
                )
        return out

    @staticmethod
    def _scores(values):
        names = [s for s, value in values.items() if value is not None]
        return {
            **values,
            **dict(zip(names, stats.centered_rank([values[s] for s in names]))),
        }

    @staticmethod
    def _read(loader, snapshots):
        rows = [FeatureObservation.model_validate(o) for o in loader()]
        for source, sha in snapshots.items():
            if (
                observation_digest([o for o in rows if o.source_id == source])
                != sha
            ):
                raise ValueError(
                    "financial source observations changed after freeze"
                )
        return rows

    @staticmethod
    def _portfolio(rows, review):
        """Build the same declared tail book for design and independent check."""
        entries = []
        for day in sorted({o.formation_at for o in rows}):
            scores = {
                o.security_id: review.bridge_sign * o.prediction
                for o in rows
                if o.formation_at == day
            }
            count = max(1, int(len(scores) * review.tail_fraction))
            entries.append(
                {
                    "formation_at": day.isoformat(),
                    "longs": sorted(scores, key=lambda s: (-scores[s], s))[
                        :count
                    ],
                    "shorts": sorted(
                        (
                            s
                            for s in scores
                            if s in review.short_eligible_security_ids
                        ),
                        key=lambda s: (scores[s], s),
                    )[:count],
                }
            )
        return entries

    def challenge(self, candidate_id, *, design_loader, check_loader):
        """Verify design conversion, reserve protected access, then delegate all gates."""
        stored = self.compositions._get("financial_result:" + candidate_id)
        frozen = self.compositions._get("financial:" + candidate_id)
        if frozen is None:
            raise ValueError("freeze an exact financial mapping first")
        record = CandidateRecord.model_validate(frozen["record"])
        review = FinancialReview.model_validate(frozen["review"])
        if stored:
            if stored.get("decision", {}).get("outcome") == "ADMITTED":
                self.integrity.restore_admission(
                    IntegrityDecision.model_validate(stored["decision"]),
                    model_hash=record.model_hash,
                )
            return stored
        if frozen["policy_digest"] != canonical_digest(
            self.integrity.policy.model_dump(mode="json")
        ):
            raise ValueError("integrity policy changed after freeze")
        plan = self.compositions.integrated_plan(frozen["composition_id"])
        source_cards = {
            c["source_id"]: c for c in plan["grant"]["source_cards"]
        }
        if any(
            s not in self.integrity.cards
            or self.integrity.cards[s].model_dump(mode="json")
            != source_cards[s]
            for s in record.source_ids
        ):
            raise ValueError(
                "integrity source cards differ from frozen composition grant"
            )
        if review.certificate_scope != "fixture" and any(
            self.integrity.cards[s].broker_rejects_formation()
            for s in record.source_ids
        ):
            return self.compositions._put(
                "financial_result:" + candidate_id,
                "financial_result",
                {
                    "status": "shadow",
                    "reason": "source eligibility unresolved",
                    "admitted": False,
                    "check_opened": False,
                },
            )
        design = self._read(design_loader, review.design_snapshots)
        dates = [
            dt.datetime.fromisoformat(d)
            for d in plan["grant"]["formation_dates"]
        ]
        design_obs = self._outcomes(frozen, design, dates)
        native = []
        for day in dates:
            group = [o for o in design_obs if o.formation_at == day]
            if len(group) < self.integrity.policy.min_obs_per_date:
                continue
            candidate = stats.spearman(
                [o.prediction for o in group], [o.target for o in group]
            )
            baseline = stats.spearman(
                [o.baseline_prediction for o in group],
                [o.target for o in group],
            )
            if candidate is not None and baseline is not None:
                native.append((candidate, candidate - baseline))
        net = net_spread_series(
            design_obs,
            self._portfolio(design_obs, review),
            bridge_sign=review.bridge_sign,
            tail_fraction=review.tail_fraction,
            cost_bps_one_way=max(
                review.cost_bps_one_way,
                self.integrity.policy.cost_bps_floor_one_way,
            ),
        )
        if (
            len(native) < self.integrity.policy.min_dates_design
            or not net
            or stats.mean([v[0] for v in native]) * review.bridge_sign
            < self.integrity.policy.min_mean_ic
            or stats.mean([v[1] for v in native])
            < self.integrity.policy.min_improvement
            or stats.mean([v["net"] for v in net])
            <= self.integrity.policy.min_net_spread
        ):
            return self.compositions._put(
                "financial_result:" + candidate_id,
                "financial_result",
                {
                    "status": "design_rejected",
                    "reason": "Native prediction or net return bridge failed the fixed design gates.",
                    "admitted": False,
                    "check_opened": False,
                },
            )
        # ForecastOutcome stores signed returns; undo that sign here because the
        # calibration helper applies bridge_sign once to raw realised returns.
        calibration = calibrate_return_bridge(
            [
                {
                    "p": o.prediction,
                    "r": review.bridge_sign * o.net_forward_return,
                    "m": month_index(o.formation_at.date()),
                    "x": o.label_resolves_at.isoformat(),
                }
                for o in design_obs
            ],
            bridge_sign=review.bridge_sign,
            design_end=review.design_end,
            min_months=self.integrity.policy.min_dates_design,
        )
        access_key = "financial_access:" + candidate_id
        with open_sqlite(self.compositions.path, immediate=True) as conn:
            if conn.execute(
                "SELECT 1 FROM compositions WHERE id=?", (access_key,)
            ).fetchone():
                raise ValueError(
                    "protected attempt was already consumed; interrupted checks cannot rerun"
                )
            attempts = [
                json.loads(row[0])
                for row in conn.execute(
                    "SELECT payload FROM compositions WHERE kind='financial_access'"
                )
            ]
            if (
                sum(a["family"] == review.mechanism_family for a in attempts)
                >= self.max_family_accesses
            ):
                raise ValueError("composition family check budget exhausted")
            value = {
                "family": review.mechanism_family,
                "candidate_id": candidate_id,
            }
            conn.execute(
                "INSERT INTO compositions VALUES(?,?,?,?)",
                (
                    access_key,
                    "financial_access",
                    json.dumps(value),
                    canonical_digest(["financial_access", value]),
                ),
            )
        try:
            check = self._read(check_loader, review.check_snapshots)
            check_obs = self._outcomes(
                frozen, design + check, review.check_dates
            )
            bundle = self.integrity.evidence_seal.issue(
                self.ctx,
                record=record,
                scope=review.certificate_scope,
                design_observations=design_obs,
                check_observations=check_obs,
                design_portfolio=self._portfolio(design_obs, review),
                check_portfolio=self._portfolio(check_obs, review),
                bridge_sign=review.bridge_sign,
                tail_fraction=review.tail_fraction,
                cost_bps_one_way=review.cost_bps_one_way,
                cost_source=review.cost_source,
                issued_at=self.clock(),
                provenance={
                    "composition_hash": canonical_digest(plan),
                    "frozen_digest": canonical_digest(frozen),
                    "dataset_class": "fixture"
                    if review.certificate_scope == "fixture"
                    else "operator_authorized",
                    "calibration_hash": canonical_digest(calibration),
                },
            )
            decision = self.integrity.evaluate(
                record,
                ctx=self.ctx,
                evaluation_asof=self.clock(),
                certificate_scope=review.certificate_scope,
                evidence=bundle,
            )
            result = {
                "status": "completed",
                "decision": decision.model_dump(mode="json"),
                "admitted": decision.admitted,
                "check_opened": True,
                "return_calibration": calibration,
                "model_hash": record.model_hash,
                "candidate_id": candidate_id,
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "error": str(exc)[:600],
                "admitted": False,
                "check_opened": True,
            }
        return self.compositions._put(
            "financial_result:" + candidate_id, "financial_result", result
        )

    def publish(
        self,
        candidate_id,
        *,
        information_cutoff,
        observations,
        publication,
        artifacts,
        published_at,
        ctx,
    ):
        """Publish only an admitted frozen graph using current observable features."""
        result = self.compositions._get("financial_result:" + candidate_id)
        frozen = self.compositions._get("financial:" + candidate_id)
        if not result or not result.get("admitted") or not frozen:
            raise ValueError(
                "no stored admitted graph challenge for publication"
            )
        cutoff = require_aware(information_cutoff, "information_cutoff")
        published = require_aware(published_at, "published_at")
        if cutoff > published:
            raise ValueError("information cutoff lies after publication")
        review = FinancialReview.model_validate(frozen["review"])
        plan = self.compositions.integrated_plan(frozen["composition_id"])
        grant = plan["grant"]
        inputs = {
            (n["recipe"]["source_id"], n["recipe"]["metric"])
            for n in frozen["graph"]
            if n.get("recipe")
        }
        rows = [FeatureObservation.model_validate(o) for o in observations]
        if len(rows) > grant["max_observations"] or any(
            (o.source_id, o.metric) not in inputs
            or o.security_id not in grant["security_ids"]
            or max(o.period_end, o.available_at, o.version_known_at) > cutoff
            for o in rows
        ):
            raise ValueError(
                "publication accepts only observable granted feature inputs, never future outcomes"
            )
        latest = {}
        for row in rows:
            key = (row.security_id, row.source_id, row.metric)
            latest[key] = max(latest.get(key, row.period_end), row.period_end)
        stale = {
            key
            for key, day in latest.items()
            if cutoff - day > dt.timedelta(days=review.max_feature_age_days)
        }
        rows = [
            row
            for row in rows
            if (row.security_id, row.source_id, row.metric) not in stale
        ]
        record = CandidateRecord.model_validate(frozen["record"])
        self.integrity.restore_admission(
            IntegrityDecision.model_validate(result["decision"]),
            model_hash=record.model_hash,
        )
        certificate = self.integrity.certificate(candidate_id)
        cal = result["return_calibration"]
        if cal["slope"] <= 0:
            raise ValueError(
                "return calibration has no positive declared-direction slope"
            )
        values = evaluate_graph(
            frozen["graph"], rows, cutoff, grant["security_ids"]
        )
        scores = self._scores(values[review.output_node])
        target = frozen["target"]
        resolves = month_end_known_at(
            month_index(cutoff.date()) + target["horizon_months"],
            target["resolve_lag_days"],
        )
        if resolves <= published:
            raise ValueError(
                "native forecast target already resolved by publication"
            )
        inputs_digest = observation_digest(rows)
        evidence_id = (
            "composition_evidence_"
            + canonical_digest(
                [candidate_id, cutoff.isoformat(), inputs_digest]
            )[:24]
        )
        uncertainty_id = "composition_uncertainty_" + candidate_id
        artifacts.register_payload(
            evidence_id,
            "sealed_composition_challenge",
            self.compositions.namespace,
            {
                "frozen_hash": canonical_digest(frozen),
                "decision": result["decision"],
                "current_inputs_hash": inputs_digest,
                "information_cutoff": cutoff.isoformat(),
            },
        )
        artifacts.register_payload(
            uncertainty_id,
            "design_only_return_calibration",
            self.compositions.namespace,
            cal,
        )
        model = ModelVersionRecord(
            owner=record.namespace,
            created_at=published,
            model_id=record.model_id,
            candidate_id=candidate_id,
            charter=record.charter,
            namespace=record.namespace,
            model_hash=record.model_hash,
            bridge_id="composition_design_ols_" + canonical_digest(cal)[:16],
            bridge_slope_bp=str(cal["slope"] * 1e4),
            training_cutoff=dt.datetime.combine(
                review.design_end, dt.time.max, tzinfo=dt.timezone.utc
            ),
            feature_names=[n["node_id"] for n in frozen["graph"]],
        )
        forecasts = [
            Forecast(
                owner=record.namespace,
                created_at=published,
                forecast_id="fc_"
                + canonical_digest(
                    [candidate_id, cutoff.isoformat(), security]
                )[:32],
                strategy_id=candidate_id,
                model_id=record.model_id,
                admission_certificate_id=certificate.certificate_id,
                security_id=security,
                information_cutoff=cutoff,
                earliest_formation_at=published,
                target=TargetPrediction(
                    target_id=record.target_id,
                    prediction=Decimal(str(score)),
                    unit="centered_rank",
                    reference_prediction=Decimal("0"),
                    reference_type="cross_sectional_neutral_centered_rank",
                    expected_resolution_at=resolves,
                ),
                return_forecast=ReturnForecast(
                    expected_excess_return=Decimal(
                        str(review.bridge_sign * cal["slope"] * score)
                    ),
                    unit="decimal_return",
                    horizon_trading_days=review.horizon_trading_days,
                    bridge_id=model.bridge_id,
                    uncertainty_artifact_id=uncertainty_id,
                ),
                mechanism_ids=[review.mechanism_family],
                source_family_ids=record.source_ids,
                evidence_artifact_ids=[evidence_id],
                expires_at=published
                + dt.timedelta(days=review.horizon_trading_days),
            ).sealed()
            for security, score in scores.items()
            if score is not None
        ]
        return publication.publish(
            ctx,
            forecasts=forecasts,
            certificate=certificate,
            model=model,
            published_at=published,
            artifact_exists=artifacts.exists,
            integration_horizon_days=review.horizon_trading_days,
            source_snapshots={
                s: self.integrity.cards[s].snapshot_id
                for s in record.source_ids
            },
        )
