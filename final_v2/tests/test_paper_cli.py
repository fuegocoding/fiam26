"""Offline tests for scripts/research_papers.py (injected transport and clock)."""

from __future__ import annotations

import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from pod_platform.papers import TransportResponse

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "research_papers.py"
spec = importlib.util.spec_from_file_location("research_papers_cli", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

NOW = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
FEED = (b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        b"<id>http://arxiv.org/abs/1706.00001v2</id><updated>2018-03-01T10:00:00Z</updated>"
        b"<published>2017-06-01T10:00:00Z</published><title>Drift news</title>"
        b"<summary>Abstract words here</summary><author><name>Ann Lee</name></author></entry></feed>")


class T:
    def __init__(self):
        self.calls = []

    def get(self, url, *, headers, timeout, max_bytes):
        self.calls.append(url)
        if "openalex" in url:
            return TransportResponse(200, b'{"results": []}')
        return TransportResponse(200, FEED)


def run(db, *argv, transport=None):
    out = io.StringIO()
    code = cli.main(["--db", str(db), *argv], transport=transport, clock=lambda: NOW, out=out)
    return code, json.loads(out.getvalue())


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    text = capsys.readouterr().out
    for word in ("search", "show", "ingest", "replicate", "--online", "OPENALEX_API_KEY"):
        assert word in text
    with pytest.raises(SystemExit) as e:
        cli.main(["search", "--help"])
    assert e.value.code == 0


def test_offline_by_default_never_uses_network(tmp_path):
    db = tmp_path / "c.sqlite"
    code, r = run(db, "search", "drift")
    assert code == 0 and r["ok"] and r["count"] == 0 and r["origin"] == "local_cache"
    code, r = run(db, "show", "arxiv:1706.00001")
    assert code == 2 and r["ok"] is False and r["error"]["type"] == "OfflineError"


def test_online_search_show_then_offline_cache(tmp_path):
    db, t = tmp_path / "c.sqlite", T()
    code, r = run(db, "--online", "search", "drift news", transport=t)
    assert code == 0 and r["mode"] == "contemporary" and r["count"] == 1
    assert r["results"][0]["paper_id"] == "arxiv:1706.00001" and r["results"][0]["version"] == 2
    n = len(t.calls)
    assert n >= 1 and all(u.startswith("https://") for u in t.calls)
    code, r = run(db, "show", "arxiv:1706.00001")  # offline
    assert code == 0 and r["paper"]["title"] == "Drift news" and len(t.calls) == n
    code, r = run(db, "search", "drift news")
    assert r["origin"] == "query_cache"
    code, r = run(db, "show", "arxiv:1706.00001", "--as-of", "2017-12-01")  # offline, unprovable
    assert code == 2 and r["error"]["type"] == "NotFound"


def test_cli_errors_are_structured(tmp_path):
    db = tmp_path / "c.sqlite"
    code, r = run(db, "show", "../../etc/passwd")
    assert code == 2 and r["error"]["type"] == "PaperValidationError"
    code, r = run(db, "search", "drift", "--as-of", "2999-01-01")
    assert code == 2 and r["error"]["type"] == "TemporalError"
    code, r = run(db, "search", "drift", "--as-of", "2020-01-01T00:00:00")
    assert code == 2 and r["error"]["type"] == "TemporalError"
    bad = type("B", (), {"get": lambda self, *a, **k: TransportResponse(500, b"")})()
    code, r = run(db, "--online", "search", "drift", transport=bad)
    assert code == 1 and r["error"]["type"] == "ProviderError"


def test_local_ingest_claim_and_replicate(tmp_path):
    db = tmp_path / "c.sqlite"
    f = tmp_path / "p.txt"
    f.write_text("1. Data\nThe sample consists of all US stocks.\n", encoding="utf-8", newline="\n")
    code, r = run(db, "ingest-local", str(f), "--title", "Local", "--first-public",
                  "2020-01-01T00:00:00+00:00", "--author", "Ann Lee")
    assert code == 0 and r["paper"]["text_status"] == "full_text"
    pid = r["paper"]["paper_id"]
    code, r = run(db, "claim", pid, "--kind", "universe", "--quote", "all US stocks")
    assert code == 0 and r["paper"]["claims"][0]["section"] == "1. Data"
    code, r = run(db, "claim", pid, "--kind", "universe", "--quote", "invented")
    assert code == 2 and r["error"]["type"] == "ClaimError"
    code, r = run(db, "show", pid, "--text-chars", "10")
    assert r["paper"]["text_excerpt"] == "1. Data\nTh"
    code, r = run(db, "replicate", pid, "--label", "EXACT_REPLICATION")
    assert code == 0 and r["spec"]["label"] == "INCOMPLETE" and r["spec"]["llm_called"] is False
    code, r = run(db, "method", "ridge_reg", "1", "--paper-id", pid, "--baseline", "seasonal")
    assert code == 0 and r["method_card"]["source"]["paper_id"] == pid
    code, r = run(db, "method", "ridge_reg", "1")
    assert code == 2
