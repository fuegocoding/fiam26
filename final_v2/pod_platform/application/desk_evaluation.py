"""Reserved research books on measured graphs and actual forward returns.

Every graph output, input component and baseline is counted before loading
returns. These hypothetical long/short research books confer no execution,
borrow, admission or publication authority.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter
from types import SimpleNamespace

from ..contracts.base import canonical_digest, require_aware
from ..research.builtins.timeutil import month_end_known_at, month_index
from ..research.composition_bridge import CompositionBridge, financial_outcomes
from ..research.desk_metrics import compute_desk_metrics
from ..research.jobs import open_sqlite
from .integrity import net_spread_series
from .research_inputs import prepare_validation_inputs

_SOURCE = "comp_chars_panel"
_RETURN = "ret_exc_lead1m"
_DEFAULTS = {
    "enabled": True,
    "periods_per_year": 12,
    "cost_bps_one_way": 10.0,
    "tail_fraction": 0.2,
    "min_training_periods": 24,
    "min_training_psr": 0.95,
    "min_training_mean_net": 0.0,
    "min_names_per_date": 20,
    "max_validation_candidates": 4,
    "min_validation_periods": 6,
    "min_validation_psr": 0.95,
    "min_validation_mean_net": 0.0,
    # psr gates on PSR, dsr on DSR deflated by every counted attempt.
    "training_gate": "psr",
    "min_training_dsr": 0.95,
    # standalone tests validation alone; pooled tests training + validation
    # and still requires a positive validation mean.
    "validation_rule": "standalone",
    # complete_history needs every attempt measured; count_unmeasured counts
    # failed proposals as trials without inventing their Sharpe ratios.
    "deflation": "complete_history",
}


def _policy(workflow):
    policy = {
        **_DEFAULTS,
        **workflow.settings["research"].get("desk_evaluation", {}),
    }
    if policy["periods_per_year"] != 12:
        raise ValueError("The declared forward-return observations are monthly")
    if not 0.05 <= policy["tail_fraction"] <= 0.34:
        raise ValueError("tail_fraction must be between 0.05 and 0.34")
    if not 0 <= policy["cost_bps_one_way"] <= 1000:
        raise ValueError("cost_bps_one_way must be between zero and 1000")
    if not 0.5 <= policy["min_training_psr"] <= 1:
        raise ValueError("min_training_psr must be between 0.5 and one")
    if not 1 <= policy["max_validation_candidates"] <= 4:
        raise ValueError("Validation selects at most four frozen candidates")
    if not 0.5 <= policy["min_validation_psr"] <= 1:
        raise ValueError("min_validation_psr must be between 0.5 and one")
    if not 0.5 <= policy["min_training_dsr"] <= 1:
        raise ValueError("min_training_dsr must be between 0.5 and one")
    if (
        policy["training_gate"] not in ("psr", "dsr")
        or policy["validation_rule"] not in ("standalone", "pooled")
        or policy["deflation"] not in ("complete_history", "count_unmeasured")
    ):
        raise ValueError("Unknown training gate, validation rule or deflation")
    if (
        policy["min_training_periods"] < 3
        or policy["min_names_per_date"] < 3
        or not math.isfinite(policy["min_training_mean_net"])
        or policy["min_validation_periods"] < 3
        or not math.isfinite(policy["min_validation_mean_net"])
    ):
        raise ValueError("Invalid minimum training evidence policy")
    return policy


def _variants(plan, policy):
    spec = plan["spec"]
    baseline = spec["baselines"][0]
    nodes = list(
        dict.fromkeys(
            spec["outputs"]
            + [n["node_id"] for n in spec["nodes"] if n.get("recipe")]
            + spec["baselines"]
        )
    )
    return [
        {
            "variant_id": plan["composition_id"] + ":" + node,
            "namespace": plan["namespace"],
            "composition_id": plan["composition_id"],
            "output_node": node,
            "roles": [
                role
                for role, selected in (
                    ("candidate", node in spec["outputs"]),
                    (
                        "component_alone",
                        any(
                            n["node_id"] == node and n.get("recipe")
                            for n in spec["nodes"]
                        ),
                    ),
                    ("baseline", node in spec["baselines"]),
                )
                if selected
            ],
            "graph": spec["nodes"],
            "target": plan["grant"]["targets"][spec["target_id"]],
            "review": {
                "output_node": node,
                "baseline_node": baseline,
                "return_source_id": _SOURCE,
                "return_metric": _RETURN,
                "return_horizon_months": 1,
                "bridge_sign": 1,
                "tail_fraction": policy["tail_fraction"],
                "cost_bps_one_way": policy["cost_bps_one_way"],
                "short_eligible_security_ids": plan["grant"]["security_ids"],
            },
        }
        for node in nodes
    ]


def _calendar(raw_dates):
    dates = sorted(
        require_aware(dt.datetime.fromisoformat(day), "formation date")
        for day in raw_dates
    )
    months = [month_index(day.date()) for day in dates]
    if not dates or any(b - a != 1 for a, b in zip(months, months[1:])):
        raise ValueError("Missing or duplicate monthly formation dates")
    return dates


def _required_history(plan):
    required = 1
    for node in plan["spec"]["nodes"]:
        recipe = node.get("recipe")
        if not recipe:
            continue
        operation = recipe["operation"]
        lookback = recipe.get("lookback", 1)
        periods = (
            1
            if operation == "latest"
            else lookback
            if operation == "rolling_mean"
            else lookback + 1
        )
        required = max(required, periods)
    return required


def _measure(plan, variant, observations, dates, now, policy):
    if _RETURN not in plan["grant"]["source_metrics"].get(_SOURCE, []):
        raise ValueError("Actual monthly excess-return measurement not granted")
    rows = financial_outcomes(plan, variant, observations, dates, now)
    expected_dates = [day.isoformat() for day in dates]
    counts = Counter(row.formation_at.isoformat() for row in rows)
    if set(counts) != set(expected_dates) or any(
        counts[day] < policy["min_names_per_date"] for day in expected_dates
    ):
        raise ValueError(
            "Missing formation dates or insufficient matched names"
        )
    # Source availability is granted; short availability is only an assumption.
    portfolio = CompositionBridge._portfolio(
        rows, SimpleNamespace(**variant["review"])
    )
    net = net_spread_series(
        rows,
        portfolio,
        bridge_sign=1,
        tail_fraction=policy["tail_fraction"],
        cost_bps_one_way=policy["cost_bps_one_way"],
    )
    if net is None or len(net) != len(dates):
        raise ValueError("Actual tail memberships do not reconcile")
    series = [
        {
            "formation_at": day,
            "matched_names": counts[day],
            **entry,
            "cost": entry["gross"] - entry["net"],
            "stress_net": entry["gross"] - 2 * (entry["gross"] - entry["net"]),
        }
        for day, entry in zip(expected_dates, net)
    ]
    return {
        "status": "measured",
        "series": series,
        "portfolio": portfolio,
        "metrics": compute_desk_metrics(
            [entry["net"] for entry in series],
            periods_per_year=policy["periods_per_year"],
        ),
        "cost_stress_2x_metrics": compute_desk_metrics(
            [entry["stress_net"] for entry in series],
            periods_per_year=policy["periods_per_year"],
        ),
        "total_turnover": math.fsum(entry["turnover"] for entry in series),
        "mean_turnover": math.fsum(entry["turnover"] for entry in series)
        / len(series),
    }


def _selection_metrics(candidates, policy):
    calendars = {
        tuple(row["formation_at"] for row in candidate["series"])
        for candidate in candidates
        if candidate["status"] == "measured"
    }
    total = len(candidates)
    complete = (
        total >= 2
        and len(calendars) == 1
        and all(
            candidate["status"] == "measured"
            and candidate["metrics"]["sharpe_periodic"] is not None
            for candidate in candidates
        )
    )
    trials = [
        c["metrics"]["sharpe_periodic"]
        for c in candidates
        if c["status"] == "measured"
        and c["metrics"]["sharpe_periodic"] is not None
    ]
    stress_trials = [
        c["cost_stress_2x_metrics"]["sharpe_periodic"]
        for c in candidates
        if c["status"] == "measured"
        and c["cost_stress_2x_metrics"]["sharpe_periodic"] is not None
    ]
    count_unmeasured = (
        policy["deflation"] == "count_unmeasured" and len(calendars) == 1
    )
    for candidate in candidates:
        if candidate["status"] != "measured":
            candidate["eligible_for_validation"] = False
            continue
        for key, field, distribution in (
            ("metrics", "net", trials),
            ("cost_stress_2x_metrics", "stress_net", stress_trials),
        ):
            candidate[key] = compute_desk_metrics(
                [row[field] for row in candidate["series"]],
                periods_per_year=policy["periods_per_year"],
                trial_sharpes=distribution,
                total_attempts=total,
                trial_periods_per_year=policy["periods_per_year"],
                attempts_complete=count_unmeasured
                or (complete and len(distribution) == total),
                unmeasured_attempts=(
                    total - len(distribution) if count_unmeasured else 0
                ),
            )
        metrics = candidate["metrics"]
        if policy["training_gate"] == "dsr":
            score = metrics["deflated_sharpe_ratio"]
            threshold = policy["min_training_dsr"]
        else:
            score = metrics["probabilistic_sharpe_ratio"]
            threshold = policy["min_training_psr"]
        candidate["eligible_for_validation"] = (
            "candidate" in candidate["roles"]
            and metrics["periods"] >= policy["min_training_periods"]
            and score is not None
            and score >= threshold
            and metrics["mean_periodic_net_return"]
            > policy["min_training_mean_net"]
        )
    return complete or count_unmeasured


def evaluate_training(workflow, plans) -> dict:
    """Reserves all fixed variants, then measures actual training return books.

    Args:
        workflow: Host workflow with settings, clock and composition services.
        plans: Registered plans; their trusted service copies are reloaded.

    Returns:
        Research-only evidence, actual return series, metrics and frozen
        validation eligibility. Failed variants remain counted attempts.
    """
    policy = _policy(workflow)
    if not policy["enabled"]:
        return {"status": "disabled", "admitted": False}
    run_id = workflow.application["run_id"]
    unique = sorted(
        {(plan["namespace"], plan["composition_id"]) for plan in plans}
    )
    registered = []
    reservations = []
    trusted = {}
    # No loader is created until every planned book has an immutable identity.
    for namespace, identifier in unique:
        service = workflow.loop.compositions[namespace]
        plan = service.integrated_plan(identifier)
        trusted[namespace, identifier] = plan
        for variant in _variants(plan, policy):
            reservation = {
                **variant,
                "run_id": run_id,
                "policy": policy,
                "plan_digest": plan["plan_digest"],
                "registered_at": plan["registered_at"],
                "formation_dates": plan["grant"]["formation_dates"],
                "security_ids": plan["grant"]["security_ids"],
                "scope": "research_only",
            }
            reservations.append((service, reservation))
            registered.append(variant)
    # This prefix is determined solely by all registered feature recipes.
    # Every trial uses it, including simple inputs and the baseline controls.
    required_history = max(
        (_required_history(plan) for plan in trusted.values()), default=1
    )
    warmup = required_history - 1
    calendars = {
        tuple(sorted(plan["grant"]["formation_dates"]))
        for plan in trusted.values()
    }
    common_calendar = list(next(iter(calendars))) if len(calendars) == 1 else []
    for _, reservation in reservations:
        original = sorted(reservation["formation_dates"])
        reservation.update(
            declared_formation_dates=original,
            formation_dates=original[warmup:],
            warmup_periods=warmup,
            required_history_periods=required_history,
            actual_training_start=(
                original[warmup] if len(original) > warmup else None
            ),
        )
    unmeasured = [
        {
            "variant_id": identifier,
            "namespace": "prior_research_revision",
            "composition_id": None,
            "output_node": None,
            "roles": ["unmeasured_hypothesis"],
            "native_measurement_status": None,
            "status": "failed",
            "error": "Prior attempted hypothesis has no measured return book",
        }
        for identifier in policy.get("prior_unmeasured_attempt_ids", [])
    ]
    for namespace, service in sorted(workflow.loop.compositions.items()):
        profile = service._get("desk_profile")
        if profile is not None and profile["run_id"] != run_id:
            raise ValueError("Research proposal ledger belongs to another run")
        with open_sqlite(service.path) as connection:
            requests = connection.execute(
                "SELECT id FROM compositions WHERE kind='desk_request' "
                "ORDER BY id"
            ).fetchall()
        for (key,) in requests:
            result = service._get("desk_result:" + key) or {}
            if (namespace, result.get("composition_id")) in trusted:
                continue
            unmeasured.append(
                {
                    "variant_id": namespace + ":proposal:" + key,
                    "namespace": namespace,
                    "composition_id": result.get("composition_id"),
                    "output_node": None,
                    "roles": ["unmeasured_hypothesis"],
                    "native_measurement_status": None,
                    "status": "failed",
                    "proposal_status": result.get("status", "pending"),
                    "error": "Proposed hypothesis has no measured return book",
                }
            )
    anchor = workflow.loop.compositions[min(workflow.loop.compositions)]
    anchor._put(
        "desk_training_registry:" + run_id,
        "desk_training_registry",
        {
            "run_id": run_id,
            "policy": policy,
            "warmup_periods": warmup,
            "required_history_periods": required_history,
            "expected_formation_dates": common_calendar[warmup:],
            "variants": [r for _, r in reservations],
            "unmeasured_hypotheses": unmeasured,
        },
    )
    for service, reservation in reservations:
        service._put(
            "desk_trial:" + run_id + ":" + reservation["variant_id"],
            "desk_trial",
            reservation,
        )
    candidates = list(unmeasured)
    for namespace, identifier in unique:
        service = workflow.loop.compositions[namespace]
        plan = trusted[namespace, identifier]
        variants = [
            row
            for row in registered
            if row["namespace"] == namespace
            and row["composition_id"] == identifier
        ]
        measurement = plan.get("measurement") or {}
        error = None
        try:
            if measurement.get("status") != "measured":
                raise ValueError("Native graph has no measured outcomes")
            if len(calendars) != 1:
                raise ValueError("Registered monthly training calendars differ")
            # Validate the original calendar before trimming its known prefix.
            dates = _calendar(plan["grant"]["formation_dates"])[warmup:]
            if not dates:
                raise ValueError(
                    "Declared feature warmup consumes the calendar"
                )
            loader = workflow.prepare_composition(namespace, identifier)
            if loader is None:
                raise ValueError("No trusted observation loader is bound")
            snapshots = service.integrated_plan(identifier)["input_snapshots"]
            if not snapshots:
                raise ValueError("Actual source snapshot digests are required")
            observations = CompositionBridge._read(loader, snapshots)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            error = str(exc)
        for variant in variants:
            result = {
                key: variant[key]
                for key in (
                    "variant_id",
                    "namespace",
                    "composition_id",
                    "output_node",
                    "roles",
                )
            }
            result["native_measurement_status"] = measurement.get("status")
            result["warmup_periods"] = warmup
            result["actual_training_start"] = (
                dates[0].isoformat() if error is None else None
            )
            try:
                if error is not None:
                    raise ValueError(error)
                result.update(
                    _measure(
                        plan,
                        variant,
                        observations,
                        dates,
                        workflow.clock(),
                        policy,
                    )
                )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                result.update(status="failed", error=str(exc)[:600])
            candidates.append(result)
    complete = _selection_metrics(candidates, policy)
    report = {
        "status": "measured"
        if any(c["status"] == "measured" for c in candidates)
        else "inconclusive",
        "run_id": run_id,
        "policy": policy,
        "warmup_periods": warmup,
        "required_history_periods": required_history,
        "actual_training_start": min(
            (
                c["actual_training_start"]
                for c in candidates
                if c.get("actual_training_start") is not None
            ),
            default=None,
        ),
        "registered_variants": [v["variant_id"] for v in registered],
        "registered_attempt_ids": (
            [v["variant_id"] for v in registered]
            + [v["variant_id"] for v in unmeasured]
        ),
        "total_attempts": len(registered) + len(unmeasured),
        "unmeasured_hypotheses": unmeasured,
        "attempt_history_scope": "current_run_ledger_only",
        "attempts_complete": complete,
        "candidates": candidates,
        "eligible_candidate_ids": [
            c["variant_id"] for c in candidates if c["eligible_for_validation"]
        ],
        "short_book_assumption": "hypothetical_research_short_book",
        "limitations": [
            "Short eligibility and borrow availability are unverified.",
            "Cost is a configured research assumption, not execution proof.",
            "Training selection is retrospective; no admission is issued.",
            "Historical research exposure outside this run is unresolved.",
        ],
        "research_only": True,
        "admitted": False,
        "orders": 0,
    }
    report["training_hash"] = canonical_digest(report)
    for namespace, _ in unique:
        workflow.loop.compositions[namespace]._put(
            "desk_training:" + run_id, "desk_training", report
        )
    return report


def _reserve_validation(service, run_id, freeze_digest, now):
    """Atomically consume the run's only validation access before any loader."""
    key = "validation_access:" + run_id
    reservation = {
        "reservation_id": key,
        "run_id": run_id,
        "freeze_digest": freeze_digest,
        "reserved_at": require_aware(now, "clock").isoformat(),
    }
    with open_sqlite(service.path, immediate=True) as connection:
        previous = connection.execute(
            "SELECT id FROM compositions WHERE id=?", (key,)
        ).fetchone()
        if previous is None:
            connection.execute(
                "INSERT INTO compositions VALUES(?,?,?,?)",
                (
                    key,
                    "validation_access",
                    json.dumps(reservation, sort_keys=True, allow_nan=False),
                    canonical_digest(["validation_access", reservation]),
                ),
            )
    if previous is not None:
        reserved = service._get(key)
        if reserved["freeze_digest"] != freeze_digest:
            raise ValueError("Validation access belongs to another freeze")
        cached = service._get("desk_validation_result:" + run_id)
        if cached is None:
            raise ValueError(
                "Validation was already reserved without a final result; "
                "access is consumed and will not be reopened"
            )
        return cached
    return None


def _validation_dates(workflow, target):
    periods = workflow.settings["research"]["periods"]
    start = dt.date.fromisoformat(periods["validation_start"])
    end = dt.date.fromisoformat(periods["validation_end"])
    if start != dt.date(2020, 1, 1) or end != dt.date(2020, 12, 31):
        raise ValueError("This research validation access is limited to 2020")
    cutoff = dt.datetime.combine(end, dt.time.max, dt.timezone.utc)
    return [
        month_end_known_at(month)
        for month in range(month_index(start), month_index(end) + 1)
        if max(
            month_end_known_at(
                month + target["horizon_months"],
                target.get("resolve_lag_days", 0),
            ),
            month_end_known_at(month + 1),
        )
        <= cutoff
    ]


def evaluate_validation(workflow, plans, training_report) -> dict:
    """Freezes selected training outputs and consumes one validation access.

    Args:
        workflow: Trusted host workflow and its immutable composition store.
        plans: Registered plans already included in the training report.
        training_report: Exact persisted result returned by evaluate_training.

    Returns:
        Actual held-out research results with no admission authority. Repeated
        calls reuse the final stored result. An interrupted reservation cannot
        reopen the loader. Selected-winner validation never estimates a full
        trial variance, so its DSR remains explicitly unavailable.
    """
    policy = _policy(workflow)
    if not policy["enabled"]:
        return {"status": "disabled", "admitted": False}
    run_id = workflow.application["run_id"]
    body = {k: v for k, v in training_report.items() if k != "training_hash"}
    if (
        canonical_digest(body) != training_report.get("training_hash")
        or training_report.get("policy") != policy
        or training_report.get("run_id") != run_id
    ):
        raise ValueError("Training evidence or frozen policy changed")
    eligible = sorted(
        (
            candidate
            for candidate in training_report["candidates"]
            if candidate.get("eligible_for_validation")
            and "candidate" in candidate["roles"]
        ),
        key=lambda c: (
            -c["metrics"]["probabilistic_sharpe_ratio"],
            -c["metrics"]["mean_periodic_net_return"],
            c["variant_id"],
        ),
    )[: policy["max_validation_candidates"]]
    if not eligible:
        return {
            "status": "no_training_candidates",
            "training_hash": training_report["training_hash"],
            "candidates": [],
            "validation_accessed": False,
            "admitted": False,
            "research_only": True,
        }
    supplied = {(p["namespace"], p["composition_id"]) for p in plans}
    anchor_name = min(workflow.loop.compositions)
    anchor = workflow.loop.compositions[anchor_name]
    selected = []
    universes = set()
    for candidate in eligible:
        identity = candidate["namespace"], candidate["composition_id"]
        if identity not in supplied:
            raise ValueError("Selected training plan is absent")
        service = workflow.loop.compositions[identity[0]]
        if service._get("desk_training:" + run_id) != training_report:
            raise ValueError("Training report is not the persisted evidence")
        plan = service.integrated_plan(identity[1])
        reserved = service._get(
            "desk_trial:" + run_id + ":" + candidate["variant_id"]
        )
        if reserved is None or reserved["plan_digest"] != plan["plan_digest"]:
            raise ValueError("Training graph differs from its reservation")
        variant = next(
            row
            for row in _variants(plan, policy)
            if row["variant_id"] == candidate["variant_id"]
        )
        dates = _validation_dates(workflow, variant["target"])
        universes.add(tuple(plan["grant"]["security_ids"]))
        selected.append(
            {
                "plan": plan,
                "variant": variant,
                "formation_dates": [day.isoformat() for day in dates],
            }
        )
    if len(universes) != 1:
        raise ValueError("Validation must retain one frozen training universe")
    freeze = {
        "run_id": run_id,
        "training_hash": training_report["training_hash"],
        "policy": policy,
        "selected": selected,
        "total_training_attempts": training_report["total_attempts"],
        "security_ids": list(next(iter(universes))),
        "research_only": True,
    }
    freeze_digest = canonical_digest(freeze)
    anchor._put("desk_validation_freeze:" + run_id, "validation_freeze", freeze)
    cached = _reserve_validation(
        anchor, run_id, freeze_digest, workflow.clock()
    )
    if cached is not None:
        return cached
    report = {
        "status": "failed",
        "run_id": run_id,
        "training_hash": training_report["training_hash"],
        "freeze_digest": freeze_digest,
        "validation_accessed": True,
        "evaluation_accessed": False,
        "research_only": True,
        "admitted": False,
        "orders": 0,
        "candidates": [],
        "short_book_assumption": "hypothetical_research_short_book",
        "limitations": [
            "Selected validation winners cannot identify full trial variance.",
            "No short availability, borrow or execution proof is established.",
            "2021 and later evaluation data remain unopened.",
        ],
    }
    try:
        prepared = prepare_validation_inputs(
            workflow.settings,
            freeze["security_ids"],
            workflow.clock,
            reservation_id="validation_access:" + run_id,
            freeze_digest=freeze_digest,
        )
        observations = workflow._load_observations(
            [prepared["observation_source"]]
        )
        report["input_snapshot"] = prepared["snapshot"]
        for entry in selected:
            variant = entry["variant"]
            candidate = {
                key: variant[key]
                for key in (
                    "variant_id",
                    "namespace",
                    "composition_id",
                    "output_node",
                    "roles",
                )
            }
            try:
                dates = _calendar(entry["formation_dates"])
                candidate.update(
                    _measure(
                        entry["plan"],
                        variant,
                        observations,
                        dates,
                        workflow.clock(),
                        policy,
                    )
                )
                for key, field in (
                    ("metrics", "net"),
                    ("cost_stress_2x_metrics", "stress_net"),
                ):
                    candidate[key] = compute_desk_metrics(
                        [row[field] for row in candidate["series"]],
                        periods_per_year=policy["periods_per_year"],
                        total_attempts=training_report["total_attempts"],
                        attempts_complete=False,
                    )
                metrics = candidate["metrics"]
                psr = metrics["probabilistic_sharpe_ratio"]
                if policy["validation_rule"] == "pooled":
                    training = next(
                        c
                        for c in training_report["candidates"]
                        if c["variant_id"] == candidate["variant_id"]
                    )
                    candidate["pooled_metrics"] = compute_desk_metrics(
                        [row["net"] for row in training["series"]]
                        + [row["net"] for row in candidate["series"]],
                        periods_per_year=policy["periods_per_year"],
                    )
                    psr = candidate["pooled_metrics"][
                        "probabilistic_sharpe_ratio"
                    ]
                candidate["passed_research_validation"] = (
                    metrics["periods"] >= policy["min_validation_periods"]
                    and psr is not None
                    and psr >= policy["min_validation_psr"]
                    and metrics["mean_periodic_net_return"]
                    > policy["min_validation_mean_net"]
                )
            except (KeyError, TypeError, ValueError, OSError) as exc:
                candidate.update(
                    status="failed",
                    error=str(exc)[:600],
                    passed_research_validation=False,
                )
            report["candidates"].append(candidate)
        report["status"] = "measured"
    except Exception as exc:
        # The reservation remains consumed even when the authorized load fails.
        report["error"] = str(exc)[:600]
    return anchor._put(
        "desk_validation_result:" + run_id, "desk_validation_result", report
    )
