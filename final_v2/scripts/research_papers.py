"""Paper research utility: search / show / ingest / claim / replicate / method (JSON output).

Settings are read from the matching section of final_v2/config.json.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.papers import (  # noqa: E402
    PapersError,
    PaperService,
    PaperStore,
    PaperValidationError,
    ProviderError,
    UrllibTransport,
)
from pod_platform.papers.models import parse_instant

ROOT = Path(__file__).resolve().parent.parent


DEFAULT_DB = ROOT / "artifacts" / "papers" / "papers.sqlite"


def _card(svc, card, text_chars=0) -> dict:
    d = card.model_dump(mode="json")
    if text_chars > 0:
        t = svc.text_of(card)
        d["text_excerpt"] = t[: min(text_chars, 20000)] if t else None
    return d


def _run(args, svc: PaperService) -> dict:
    c = args.cmd
    if c == "search":
        r = svc.search(args.query, as_of=args.as_of, limit=args.limit)
        return {
            "ok": True,
            "mode": r.mode,
            "as_of": r.as_of.isoformat() if r.as_of else None,
            "origin": r.origin,
            "warnings": r.warnings,
            "count": len(r.cards),
            "results": [_card(svc, x) for x in r.cards],
        }
    if c == "show":
        return {
            "ok": True,
            "paper": _card(
                svc,
                svc.get(args.paper_id, as_of=args.as_of, refresh=args.refresh),
                args.text_chars,
            ),
        }
    if c == "ingest":
        return {
            "ok": True,
            "paper": _card(
                svc, svc.ingest_full_text(args.paper_id, as_of=args.as_of)
            ),
        }
    if c == "ingest-local":
        card = svc.ingest_local(
            args.file,
            title=args.title,
            first_public_date=parse_instant(args.first_public),
            version_date=parse_instant(args.version_date)
            if args.version_date
            else None,
            authors=args.author,
            doi=args.doi,
        )
        return {"ok": True, "paper": _card(svc, card)}
    if c == "claim":
        card = svc.add_claim(
            args.paper_id,
            kind=args.kind,
            quote=args.quote,
            text=args.text,
            source=args.source,
            as_of=args.as_of,
        )
        return {"ok": True, "paper": _card(svc, card)}
    if c == "extract-claims":
        return {
            "ok": True,
            "paper": _card(
                svc, svc.extract_claims(args.paper_id, as_of=args.as_of)
            ),
        }
    if c == "replicate":
        return {
            "ok": True,
            "spec": svc.create_replication_spec(
                args.paper_id,
                requested_label=args.label,
                as_of=args.as_of,
                deviations=args.deviation,
                comparison_plan=args.comparison_plan,
            ),
        }
    if c == "method":
        return {
            "ok": True,
            "method_card": svc.create_method_card(
                args.method_id,
                args.version,
                paper_id=args.paper_id,
                supported_targets=args.target,
                min_history=args.min_history,
                assumptions=args.assumption,
                baselines=args.baseline,
                failure_modes=args.failure_mode,
                reference_implementation=args.reference_impl,
            ),
        }
    raise PaperValidationError(f"unknown command {c!r}")


SCRIPT_SETTINGS = load_config("_utility_schemas")["research_papers"]


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    transport=None,
    clock: Optional[Callable[[], datetime]] = None,
    out=None,
) -> int:
    out = out or sys.stdout
    args = load_script_settings("research_papers", SCRIPT_SETTINGS, argv)
    store = None
    try:
        store = PaperStore(args.db)
        svc = PaperService(
            store,
            clock=clock or (lambda: datetime.now(timezone.utc)),
            transport=(transport or UrllibTransport()) if args.online else None,
            online=args.online,
            timeout=args.timeout,
            openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
            openalex_mailto=os.environ.get("OPENALEX_MAILTO") or None,
        )
        result, code = _run(args, svc), 0
    except (PapersError, ValueError) as exc:
        kind = type(exc).__name__
        result = {
            "ok": False,
            "error": {"type": kind, "message": str(exc)[:1000]},
        }
        code = 1 if isinstance(exc, ProviderError) else 2
    finally:
        if store is not None:
            store.close()
    out.write(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
