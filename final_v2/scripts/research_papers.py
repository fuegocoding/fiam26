"""Paper research CLI: search / show / ingest / claim / replicate / method (JSON output).

Offline by default: only the SQLite cache is used. Real HTTP needs ``--online``.
OpenAlex credentials come from the optional env vars OPENALEX_API_KEY and
OPENALEX_MAILTO (never written to cards). ``--as-of`` switches to historical replay.

Examples:
  research_papers.py --online search "post earnings announcement drift" --limit 5
  research_papers.py --online show arxiv:1706.03762 --as-of 2017-12-31
  research_papers.py --online ingest arxiv:1706.03762v1
  research_papers.py claim arxiv:1706.03762v1 --kind universe --quote "exact sentence"
  research_papers.py replicate arxiv:1706.03762v1 --label INCOMPLETE

Every command prints one JSON object: {"ok": true, ...} or {"ok": false, "error": {...}}.
Exit codes: 0 ok, 2 invalid request/offline miss/not found, 1 provider/other error.
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

from pod_platform.papers import (PaperService, PaperStore, PapersError,  # noqa: E402
                                 PaperValidationError, ProviderError, UrllibTransport)
from pod_platform.papers.models import parse_instant  # noqa: E402

DEFAULT_DB = ROOT / "artifacts" / "papers" / "papers.sqlite"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research_papers", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(DEFAULT_DB), help="SQLite cache path")
    p.add_argument("--online", action="store_true", help="allow real HTTP to arXiv/OpenAlex")
    p.add_argument("--timeout", type=float, default=20.0)
    sub = p.add_subparsers(dest="cmd", required=True)

    def asof(sp):
        sp.add_argument("--as-of", default=None,
                        help="historical replay cutoff (ISO date/time with tz; date => 00:00Z)")

    s = sub.add_parser("search", help="search papers"); asof(s)
    s.add_argument("query"); s.add_argument("--limit", type=int, default=10)
    s = sub.add_parser("show", help="show a paper card"); asof(s)
    s.add_argument("paper_id"); s.add_argument("--refresh", action="store_true")
    s.add_argument("--text-chars", type=int, default=0, help="include first N chars of full text")
    s = sub.add_parser("ingest", help="fetch full text (arXiv PDF) into the cache"); asof(s)
    s.add_argument("paper_id")
    s = sub.add_parser("ingest-local", help="register a team-supplied .txt/.md/.pdf")
    s.add_argument("file"); s.add_argument("--title", required=True)
    s.add_argument("--first-public", required=True, help="ISO datetime with timezone")
    s.add_argument("--version-date", default=None); s.add_argument("--doi", default=None)
    s.add_argument("--author", action="append", default=[])
    s = sub.add_parser("claim", help="add a source-grounded claim (verbatim quote)"); asof(s)
    s.add_argument("paper_id"); s.add_argument("--kind", required=True)
    s.add_argument("--quote", required=True); s.add_argument("--text", default=None)
    s.add_argument("--source", choices=["full_text", "abstract"], default="full_text")
    s = sub.add_parser("extract-claims", help="heuristic extractive claims (needs review)"); asof(s)
    s.add_argument("paper_id")
    s = sub.add_parser("replicate", help="create a replication/adaptation spec"); asof(s)
    s.add_argument("paper_id")
    s.add_argument("--label", required=True,
                   choices=["EXACT_REPLICATION", "CLOSE_REPLICATION", "MECHANISM_TRANSFER",
                            "METHOD_TRANSFER", "INCOMPLETE"])
    s.add_argument("--deviation", action="append", default=[])
    s.add_argument("--comparison-plan", default=None)
    s = sub.add_parser("method", help="create a method card")
    s.add_argument("method_id"); s.add_argument("version"); s.add_argument("--paper-id")
    s.add_argument("--target", action="append", default=[]); s.add_argument("--baseline", action="append", default=[])
    s.add_argument("--assumption", action="append", default=[]); s.add_argument("--min-history")
    s.add_argument("--failure-mode", action="append", default=[]); s.add_argument("--reference-impl")
    return p


def _card(svc, card, text_chars=0) -> dict:
    d = card.model_dump(mode="json")
    if text_chars > 0:
        t = svc.text_of(card)
        d["text_excerpt"] = t[:min(text_chars, 20000)] if t else None
    return d


def _run(args, svc: PaperService) -> dict:
    c = args.cmd
    if c == "search":
        r = svc.search(args.query, as_of=args.as_of, limit=args.limit)
        return {"ok": True, "mode": r.mode, "as_of": r.as_of.isoformat() if r.as_of else None,
                "origin": r.origin, "warnings": r.warnings, "count": len(r.cards),
                "results": [_card(svc, x) for x in r.cards]}
    if c == "show":
        return {"ok": True, "paper": _card(svc, svc.get(args.paper_id, as_of=args.as_of,
                                                         refresh=args.refresh), args.text_chars)}
    if c == "ingest":
        return {"ok": True, "paper": _card(svc, svc.ingest_full_text(args.paper_id, as_of=args.as_of))}
    if c == "ingest-local":
        card = svc.ingest_local(
            args.file, title=args.title, first_public_date=parse_instant(args.first_public),
            version_date=parse_instant(args.version_date) if args.version_date else None,
            authors=args.author, doi=args.doi)
        return {"ok": True, "paper": _card(svc, card)}
    if c == "claim":
        card = svc.add_claim(args.paper_id, kind=args.kind, quote=args.quote, text=args.text,
                             source=args.source, as_of=args.as_of)
        return {"ok": True, "paper": _card(svc, card)}
    if c == "extract-claims":
        return {"ok": True, "paper": _card(svc, svc.extract_claims(args.paper_id, as_of=args.as_of))}
    if c == "replicate":
        return {"ok": True, "spec": svc.create_replication_spec(
            args.paper_id, requested_label=args.label, as_of=args.as_of,
            deviations=args.deviation, comparison_plan=args.comparison_plan)}
    if c == "method":
        return {"ok": True, "method_card": svc.create_method_card(
            args.method_id, args.version, paper_id=args.paper_id, supported_targets=args.target,
            min_history=args.min_history, assumptions=args.assumption, baselines=args.baseline,
            failure_modes=args.failure_mode, reference_implementation=args.reference_impl)}
    raise PaperValidationError(f"unknown command {c!r}")


def main(argv: Optional[Sequence[str]] = None, *, transport=None,
         clock: Optional[Callable[[], datetime]] = None, out=None) -> int:
    out = out or sys.stdout
    args = _parser().parse_args(argv)
    store = None
    try:
        store = PaperStore(args.db)
        svc = PaperService(
            store, clock=clock or (lambda: datetime.now(timezone.utc)),
            transport=(transport or UrllibTransport()) if args.online else None,
            online=args.online, timeout=args.timeout,
            openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
            openalex_mailto=os.environ.get("OPENALEX_MAILTO") or None)
        result, code = _run(args, svc), 0
    except (PapersError, ValueError) as exc:
        kind = type(exc).__name__
        result = {"ok": False, "error": {"type": kind, "message": str(exc)[:1000]}}
        code = 1 if isinstance(exc, ProviderError) else 2
    finally:
        if store is not None:
            store.close()
    out.write(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
