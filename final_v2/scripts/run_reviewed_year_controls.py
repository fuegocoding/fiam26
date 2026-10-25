#!/usr/bin/env python3
"""Operator-controlled, preregistered design follow-ups. No model calls or admissions.

These controls test already reviewed comparisons separately from a pod's
advisory review disposition. They never pretend that a paper was replicated,
that the pod autonomously approved execution, or that design is a holdout.
"""

import datetime as dt
import json
import sqlite3

from pod_platform.application.common import (
    atomic_write_json,
    now_utc,
    read_json,
)
from pod_platform.application.configuration import load_config
from pod_platform.application.integrity import (
    IntegrityPolicy,
    classify_research_candidate,
)
from pod_platform.application.script_settings import load_script_settings
from pod_platform.papers import PaperService, PaperStore
from pod_platform.papers.adaptations import PaperAdapter
from pod_platform.pods.runner import ExecutorRegistry
from pod_platform.research.builtins import register_executors
from pod_platform.research.memory import PodMemory
from pod_platform.research.registration import ResearchLedger
from pod_platform.research.registry import Registry

SCRIPT_SETTINGS = load_config("_utility_schemas")["run_reviewed_year_controls"]


def main():

    a = load_script_settings(
        "run_reviewed_year_controls", SCRIPT_SETTINGS, None
    )
    a.state_dir.mkdir(parents=True, exist_ok=True)
    clock_path = a.state_dir / "clock.json"
    if not clock_path.exists():
        atomic_write_json(clock_path, {"started_at": now_utc().isoformat()})
    at = dt.datetime.fromisoformat(read_json(clock_path)["started_at"])

    def clock():
        return at

    with sqlite3.connect(a.from_run / "research/runtime.sqlite") as conn:
        reviews = [
            json.loads(x[0])
            for x in conn.execute(
                "SELECT payload FROM runtime_events WHERE kind='review' ORDER BY seq"
            )
        ]
    registry = Registry(a.state_dir / "registry.sqlite")
    ledger = ResearchLedger(registry)
    executors = ExecutorRegistry()
    context = register_executors(
        executors,
        manifest_path=a.manifest,
        state_dir=a.state_dir / "builtin",
        clock=clock,
        research_asof=at,
        variant_budget=4,
    )
    papers = PaperService(PaperStore(a.papers_db), clock=clock)
    results = []
    for review in reviews:
        review = {
            **review,
            "reviewed_at": at.isoformat(),
            "reviewed_by": "operator_design_control",
        }
        adapter = PaperAdapter(
            a.state_dir / "adaptations.sqlite",
            papers=papers,
            registry=registry,
            ledger=ledger,
            memory=PodMemory(
                a.state_dir / "memory", review["namespace"], clock=clock
            ),
            context=context,
            clock=clock,
        )
        row = {
            "namespace": review["namespace"],
            "params": {
                k: review[k]
                for k in ("recipe_id", "feature_set", "ridge_lambda")
            },
            "autonomous_pod_execution": False,
            "mode": "operator_preregistered_design_control",
            "admitted": False,
        }
        try:
            record = adapter.adapt(review)
            measured = adapter.execute_design(
                record["adaptation_id"], executors
            )
            report = context.report(
                record["registration_id"], record["variant"]
            )
            dataset = context.manifest.dataset(review["dataset_id"])
            decision = classify_research_candidate(
                {
                    "candidate_id": record["adaptation_id"],
                    "charter": report["charter"],
                    "status": "evaluated",
                    "metrics": report["metrics"],
                    "source_ids": [report["source_id"]],
                },
                IntegrityPolicy(policy_id="year_controls_v1", frozen_at=at),
                {report["source_id"]: dataset.source_card},
            )
            row.update(
                status="measured",
                outcome=decision.outcome.value,
                metrics=report["metrics"],
                report_hash=report["report_hash"],
                lineage=measured["lineage"],
                reasons=decision.reasons,
                net_bridge_passed=report["metrics"][
                    "bridge_net_spread_candidate"
                ]
                > 0,
                net_bridge_threshold=0,
            )
        except Exception as exc:
            row.update(
                status="failed",
                error_type=type(exc).__name__,
                reason=str(exc)[:500],
            )
        results.append(row)
        atomic_write_json(
            a.state_dir / "summary.json",
            {
                "mode": "operator_preregistered_design_controls",
                "model_calls": 0,
                "protected_check_accesses": 0,
                "admissions": 0,
                "orders": 0,
                "comparisons": results,
            },
        )
        print(
            review["namespace"],
            row["status"],
            row.get("outcome", row.get("error_type")),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
