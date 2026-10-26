"""Registered comparison of supplied forecasts; never executes generated code.

This small executor measures forecast quality and a separately supplied return
bridge. It does not fit a model, simulate execution, or issue admission
certificates. Outcomes must be mature at the evaluation clock. A newly designed
experiment on old outcomes is explicitly retrospective.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import statistics
from pathlib import Path

from pydantic import Field, model_validator

from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.sources import SourceCard


class ComparisonPlan(StrictModel):
    experiment_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    hypothesis_id: str = Field(min_length=1)
    registered_at: dt.datetime
    target_id: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    variant_budget: int = Field(ge=1, le=100)
    partition: str = "design"
    research_only: bool = True
    min_dates: int = Field(default=3, ge=2)

    @model_validator(mode="after")
    def clocks(self):
        require_aware(self.registered_at, "registered_at")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be unique")
        if self.partition not in {"design", "check", "replay", "prospective"}:
            raise ValueError("unknown evaluation partition")
        return self


class ForecastOutcome(StrictModel):
    security_id: str = Field(min_length=1)
    formation_at: dt.datetime
    feature_available_at: dt.datetime
    label_resolves_at: dt.datetime
    prediction: float = Field(allow_inf_nan=False)
    baseline_prediction: float = Field(allow_inf_nan=False)
    target: float = Field(allow_inf_nan=False)
    net_forward_return: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def clocks(self):
        for name in (
            "formation_at",
            "feature_available_at",
            "label_resolves_at",
        ):
            require_aware(getattr(self, name), name)
        if self.feature_available_at > self.formation_at:
            raise ValueError("feature not available at formation")
        if self.label_resolves_at <= self.formation_at:
            raise ValueError("label must resolve after formation")
        return self


def _rank(values: list[float]) -> list[float]:
    # Average tied ranks, so flat forecasts have no manufactured skill.
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and values[ordered[end]] == values[ordered[start]]
        ):
            end += 1
        for index in ordered[start:end]:
            ranks[index] = (start + end - 1) / 2
        start = end
    return ranks


def _ic(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(statistics.correlation(_rank(x), _rank(y)))


class RegisteredEvaluator:
    """Persistent, idempotent executor with a per-experiment variant budget."""

    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(str(path), timeout=30)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS comparison_plans
                (id TEXT PRIMARY KEY, digest TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS comparison_trials
                (experiment_id TEXT NOT NULL, variant_id TEXT NOT NULL,
                 input_hash TEXT NOT NULL, result TEXT,
                 PRIMARY KEY(experiment_id, variant_id));
        """)

    def close(self):
        self.connection.close()

    def register(self, plan: ComparisonPlan) -> str:
        payload = plan.model_dump(mode="json")
        digest = canonical_digest(payload)
        with self.connection:
            row = self.connection.execute(
                "SELECT digest FROM comparison_plans WHERE id=?",
                (plan.experiment_id,),
            ).fetchone()
            if row and row[0] != digest:
                raise ValueError(
                    "registered plan is immutable; use a new experiment ID"
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO comparison_plans VALUES(?,?,?)",
                (
                    plan.experiment_id,
                    digest,
                    json.dumps(payload, sort_keys=True),
                ),
            )
        return digest

    def evaluate(
        self,
        experiment_id: str,
        variant_id: str,
        observations: list[ForecastOutcome],
        *,
        owner: str,
        evaluation_asof: dt.datetime,
        source_cards: list[SourceCard],
        allowed_partitions: tuple[str, ...] = ("design",),
    ) -> dict:
        require_aware(evaluation_asof, "evaluation_asof")
        if not variant_id or not observations or len(observations) > 300_000:
            raise ValueError(
                "nonempty variant and 1..300000 observations required"
            )
        row = self.connection.execute(
            "SELECT payload FROM comparison_plans WHERE id=?", (experiment_id,)
        ).fetchone()
        if row is None:
            raise ValueError(
                "register the experiment before accessing outcomes"
            )
        plan = ComparisonPlan.model_validate_json(row[0])
        if plan.owner != owner:
            raise PermissionError("experiment belongs to another pod")
        if plan.partition not in allowed_partitions:
            raise PermissionError(
                "protected evaluation partition is not authorized"
            )
        if plan.registered_at > evaluation_asof:
            raise ValueError("registration lies after evaluation clock")
        cards = {card.source_id: card for card in source_cards}
        if set(plan.source_ids) - cards.keys():
            raise ValueError("missing source cards")
        blocked = sorted(
            s for s in plan.source_ids if cards[s].broker_rejects_formation()
        )
        if blocked and not plan.research_only:
            raise ValueError(
                "ineligible formation sources: " + ", ".join(blocked)
            )
        seen = set()
        groups: dict[str, list[ForecastOutcome]] = {}
        retrospective = False
        for observation in observations:
            if observation.label_resolves_at > evaluation_asof:
                raise ValueError("outcome has not matured at evaluation clock")
            key = (observation.security_id, observation.formation_at)
            if key in seen:
                raise ValueError("duplicate security and formation")
            seen.add(key)
            retrospective |= observation.formation_at < plan.registered_at
            groups.setdefault(observation.formation_at.isoformat(), []).append(
                observation
            )
        if plan.partition == "prospective" and retrospective:
            raise ValueError(
                "prospective observations must follow registration"
            )
        payload = {
            "observations": [
                o.model_dump(mode="json")
                for o in sorted(
                    observations, key=lambda o: (o.formation_at, o.security_id)
                )
            ],
            "evaluation_asof": evaluation_asof.isoformat(),
            "source_cards": [
                cards[s].model_dump(mode="json")
                for s in sorted(plan.source_ids)
            ],
        }
        input_hash = canonical_digest(payload)
        # BEGIN IMMEDIATE makes budget admission atomic across worker processes.
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            prior = self.connection.execute(
                "SELECT input_hash,result FROM comparison_trials WHERE experiment_id=? AND variant_id=?",
                (experiment_id, variant_id),
            ).fetchone()
            if prior:
                if prior[0] != input_hash:
                    raise ValueError(
                        "variant ID already used with different inputs"
                    )
                if prior[1] is None:
                    raise ValueError(
                        "variant was already consumed by an interrupted attempt"
                    )
                return json.loads(prior[1])
            count = self.connection.execute(
                "SELECT COUNT(*) FROM comparison_trials WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()[0]
            if count >= plan.variant_budget:
                raise ValueError("registered variant budget exhausted")
            self.connection.execute(
                "INSERT INTO comparison_trials VALUES(?,?,?,NULL)",
                (experiment_id, variant_id, input_hash),
            )
            self.connection.commit()
        finally:
            if self.connection.in_transaction:
                self.connection.rollback()
        dates = []
        for formation, rows in sorted(groups.items()):
            pred, target = (
                [r.prediction for r in rows],
                [r.target for r in rows],
            )
            baseline = [r.baseline_prediction for r in rows]
            bridge_rows = [r for r in rows if r.net_forward_return is not None]
            dates.append(
                {
                    "formation_at": formation,
                    "observations": len(rows),
                    "candidate_ic": _ic(pred, target),
                    "baseline_ic": _ic(baseline, target),
                    "bridge_ic": _ic(
                        [r.prediction for r in bridge_rows],
                        [r.net_forward_return for r in bridge_rows],
                    ),
                }
            )
        paired = [
            d["candidate_ic"] - d["baseline_ic"]
            for d in dates
            if d["candidate_ic"] is not None and d["baseline_ic"] is not None
        ]
        improvement = statistics.mean(paired) if paired else None
        # Descriptive SE only; not a dependence-adjusted significance claim.
        se = (
            statistics.stdev(paired) / math.sqrt(len(paired))
            if len(paired) > 1
            else None
        )
        result = {
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "input_hash": input_hash,
            "dates": dates,
            "paired_dates": len(paired),
            "paired_mean_ic_improvement": improvement,
            "descriptive_standard_error": se,
            "enough_dates": len(paired) >= plan.min_dates,
            "evaluation_label": "retrospective_research"
            if retrospective
            else "registered_research",
            "research_only": plan.research_only,
            "blocked_formation_sources": blocked,
            "admitted": False,
            "limitations": [
                "Supplied forecasts only; no model fitting or execution simulation.",
                "Descriptive comparisons do not establish significance after dependence or selection.",
            ],
        }
        with self.connection:
            self.connection.execute(
                "UPDATE comparison_trials SET result=? WHERE experiment_id=? AND variant_id=?",
                (
                    json.dumps(result, sort_keys=True, allow_nan=False),
                    experiment_id,
                    variant_id,
                ),
            )
        return result
