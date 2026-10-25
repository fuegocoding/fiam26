"""Recurring paper watches: add / list / tick (explicit ticks only; no daemon, no model calls).

Offline by default: ticks search the local paper cache. ``--online`` permits HTTP to
arXiv/OpenAlex. A tick queues bounded 'propose' jobs (with the papers actually found)
into <state-dir>/queue.sqlite; process them later with run_pod_research.py. Each
paper version is queued at most once per pod and watch.

Examples:
  research_watch.py add --state-dir /tmp/pods --pod valuation --watch-id value_spread \\
      --topic "value spread predictability" --cadence-hours 168 --executor-id value_spread_ic
  research_watch.py tick --state-dir /tmp/pods [--online] [--pod valuation]
  research_watch.py list --state-dir /tmp/pods
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pod_platform.papers import PaperService, PapersError, UrllibTransport  # noqa: E402
from pod_platform.papers.watches import PaperWatches  # noqa: E402
from pod_platform.pods.charters import POD_SPECS  # noqa: E402
from pod_platform.research.jobs import QueueError, ResearchQueue  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pods = sorted(s.pod_id for s in POD_SPECS.values())
    for name in ("add", "list", "tick"):
        p = sub.add_parser(name)
        p.add_argument("--state-dir", required=True)
        p.add_argument("--pod", choices=pods, required=name == "add")
    sub.choices["add"].add_argument("--watch-id", required=True)
    sub.choices["add"].add_argument("--topic", required=True, help="paper search query")
    sub.choices["add"].add_argument("--question", help="research question (default derived from the pod mission)")
    sub.choices["add"].add_argument("--cadence-hours", type=float, required=True)
    sub.choices["add"].add_argument("--executor-id", action="append", required=True)
    sub.choices["add"].add_argument("--search-limit", type=int, default=10)
    sub.choices["add"].add_argument("--papers-per-job", type=int, default=3)
    sub.choices["add"].add_argument("--max-jobs", type=int, default=1, help="jobs queued per tick")
    sub.choices["tick"].add_argument("--watch-id")
    sub.choices["tick"].add_argument("--online", action="store_true", help="allow HTTP to arXiv/OpenAlex")
    sub.choices["tick"].add_argument("--papers-db", help="paper cache (default <state-dir>/shared/papers.sqlite)")
    sub.choices["tick"].add_argument("--timeout", type=float, default=20.0)
    return parser


def main(argv: Optional[Sequence[str]] = None, *, transport=None,
         clock: Optional[Callable[[], datetime]] = None, out=None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    clock = clock or (lambda: datetime.now(timezone.utc))
    state = Path(args.state_dir)
    online = getattr(args, "online", False)
    papers = None
    try:
        state.mkdir(parents=True, exist_ok=True)
        papers = PaperService(getattr(args, "papers_db", None) or state / "shared" / "papers.sqlite",
                              clock=clock, online=online, timeout=getattr(args, "timeout", 20.0),
                              transport=(transport or UrllibTransport()) if online else None,
                              openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
                              openalex_mailto=os.environ.get("OPENALEX_MAILTO") or None) if args.command == "tick" else None
        watches = PaperWatches(state / "watches.sqlite", queue=ResearchQueue(state / "queue.sqlite", clock=clock),
                               papers=papers, clock=clock)
        if args.command == "add":
            result = watches.add(args.pod, args.watch_id, args.topic, cadence_hours=args.cadence_hours,
                                 executors=args.executor_id, question=args.question,
                                 search_limit=args.search_limit, papers_per_job=args.papers_per_job,
                                 max_jobs=args.max_jobs)
        elif args.command == "list":
            result = watches.list(args.pod)
        else:
            result = watches.tick(args.pod, args.watch_id)
        code = 1 if any(r.get("status") == "failed" for r in result if isinstance(r, dict)) else 0
        out.write(json.dumps({"ok": code == 0, "online": online, "result": result}, indent=2, default=str) + "\n")
        return code
    except (PapersError, QueueError, ValueError, KeyError) as exc:
        out.write(json.dumps({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)[:500]}}) + "\n")
        return 2
    finally:
        if papers is not None:
            papers.store.close()


if __name__ == "__main__":
    sys.exit(main())
