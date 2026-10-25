"""Recurring research allocator: configure / add-trigger / tick / status (explicit ticks only; no daemon, no model calls).

A tick opens the current weekly cycle and the matured-cohort monthly cycle per configured pod and routes at most the
per-lane quota of ranked triggers into <state-dir>/queue.sqlite (worker kinds 'propose'/'experiment'); items nothing can
execute are recorded as review_only. Process the queue with run_pod_research.py.

Examples:
  research_scheduler.py configure --state-dir /tmp/pods --pod valuation --executor-id builtin_valuation_relative_value_v1
  research_scheduler.py add-trigger --state-dir /tmp/pods --pod valuation --kind source --subject snap_2026_10 \\
      --question "Does the new snapshot change coverage?" --scores '{"economic_importance":0.6,"evidence":0.5,"feasibility":0.7,"cost":0.2,"redundancy":0.1}' \\
      --payload-file cards.json
  research_scheduler.py tick --state-dir /tmp/pods
  research_scheduler.py status --state-dir /tmp/pods --pod valuation
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pod_platform.pods.charters import POD_SPECS, pod_spec  # noqa: E402
from pod_platform.research.jobs import QueueError, ResearchQueue  # noqa: E402
from pod_platform.research.scheduler import KIND_LANE, PodPolicy, ResearchScheduler, SchedulerError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pods = sorted(s.pod_id for s in POD_SPECS.values())
    for name in ("configure", "add-trigger", "tick", "status", "pause"):
        p = sub.add_parser(name)
        p.add_argument("--state-dir", required=True)
        p.add_argument("--pod", choices=pods, required=name != "tick")
    c = sub.choices["configure"]
    c.add_argument("--executor-id", action="append", required=True)
    c.add_argument("--weekly-quota", help='JSON, e.g. {"candidate":2,"challenge":1,"replication":1,"review":3}')
    c.add_argument("--monthly-quota", help="JSON like --weekly-quota")
    c.add_argument("--cohort-maturity-days", type=int, default=5)
    c.add_argument("--max-failures-per-line", type=int, default=2)
    t = sub.choices["add-trigger"]
    t.add_argument("--kind", choices=sorted(KIND_LANE), required=True)
    t.add_argument("--subject", required=True)
    t.add_argument("--question", required=True)
    t.add_argument("--scores", required=True, help="JSON with economic_importance, evidence, feasibility, cost, redundancy in [0,1]")
    t.add_argument("--payload-file", help="JSON object: paper_cards/dataset_cards/source_cards/draft_entry_id/variants")
    t.add_argument("--line-id")
    t.add_argument("--depends-on", action="append", default=[])
    t.add_argument("--cohort-month")
    t.add_argument("--redundancy-key")
    p = sub.choices["pause"]
    p.add_argument("--reason", required=True)
    p.add_argument("--line-id", help="omit to pause the whole pod")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, clock: Optional[Callable[[], datetime]] = None, out=None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    clock = clock or (lambda: datetime.now(timezone.utc))
    state = Path(args.state_dir)
    try:
        state.mkdir(parents=True, exist_ok=True)
        sched = ResearchScheduler(state / "scheduler.sqlite", queue=ResearchQueue(state / "queue.sqlite", clock=clock),
                                  clock=clock, memory_root=str(state / "memory"))
        ns = pod_spec(args.pod).namespace if getattr(args, "pod", None) else None
        if args.command == "configure":
            kwargs = {k: json.loads(v) for k, v in (("weekly_quota", args.weekly_quota), ("monthly_quota", args.monthly_quota)) if v}
            pol = PodPolicy(ns, tuple(args.executor_id), cohort_maturity_days=args.cohort_maturity_days,
                            max_failures_per_line=args.max_failures_per_line, **kwargs)
            result = sched.configure_pod(pol).__dict__
        elif args.command == "add-trigger":
            payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8")) if args.payload_file else {}
            result = sched.add_trigger(ns, args.kind, args.subject, args.question, json.loads(args.scores), payload=payload,
                                       line_id=args.line_id, depends_on=args.depends_on, cohort_month=args.cohort_month,
                                       redundancy_key=args.redundancy_key)
        elif args.command == "pause":
            sched.pause(ns, args.reason, args.line_id)
            result = {"paused": args.line_id or "*"}
        elif args.command == "status":
            result = sched.status(ns)
        else:
            result = sched.tick()
        out.write(json.dumps({"ok": True, "result": result}, indent=2, default=str) + "\n")
        return 0
    except (SchedulerError, QueueError, ValueError, KeyError, OSError) as exc:
        out.write(json.dumps({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)[:500]}}) + "\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
