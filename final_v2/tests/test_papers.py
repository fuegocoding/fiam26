"""Offline regression tests for pod_platform.papers (no real network)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from pod_platform.papers import (AuthError, ClaimError, NotFound, OfflineError, PaperService,
                                 PaperStore, PaperValidationError, ProviderError, TemporalError,
                                 TransportResponse, UrlNotAllowed, canonical_paper_id,
                                 validate_url)
from pod_platform.papers.providers import parse_arxiv_feed, parse_openalex_works
from pod_platform.papers.transport import check_status

NOW = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
BASE = "1706.00001"
DATES = {1: "2017-06-01T10:00:00Z", 2: "2018-03-01T10:00:00Z", 3: "2025-01-01T00:00:00Z"}


def clock():
    return NOW


def entry(v, doi=None, base=BASE):
    d = f"<arxiv:doi>{doi}</arxiv:doi>" if doi else ""
    return (f"<entry><id>http://arxiv.org/abs/{base}v{v}</id><updated>{DATES[v]}</updated>"
            f"<published>{DATES[1]}</published><title>Drift  after\n news</title>"
            f"<summary>abstract text v{v}</summary><author><name>Ann Lee</name></author>{d}</entry>")


def feed(*entries):
    return ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:arxiv="http://arxiv.org/schemas/atom">' + "".join(entries) + "</feed>").encode()


def oa_json(*works):
    return json.dumps({"results": list(works)}).encode()


def oa_work(wid="W1", doi="https://doi.org/10.1000/xyz", date="2019-01-01", typ="article",
            version="publishedVersion", title="Journal drift"):
    return {"id": f"https://openalex.org/{wid}", "display_name": title, "doi": doi,
            "publication_date": date, "type": typ,
            "primary_location": {"version": version, "landing_page_url": "https://example.org/p"},
            "abstract_inverted_index": {"Hello": [0], "world": [1]},
            "authorships": [{"author": {"display_name": "Ann Lee"}}]}


class Fake:
    def __init__(self, handler=None):
        self.calls, self.kw = [], []
        self.handler = handler or self.default

    def default(self, url):
        p = urlsplit(url)
        q = parse_qs(p.query)
        if p.hostname == "export.arxiv.org":
            ident = q.get("id_list", [BASE])[0]
            m = re.fullmatch(re.escape(BASE) + r"(?:v(\d))?", ident)
            if not m:
                return TransportResponse(200, feed())
            return TransportResponse(200, feed(entry(int(m.group(1) or 3), doi="10.1000/xyz")))
        if p.hostname == "api.openalex.org":
            return TransportResponse(200, oa_json(oa_work()))
        return TransportResponse(200, b"not a pdf")

    def get(self, url, *, headers, timeout, max_bytes):
        self.calls.append(url)
        self.kw.append((timeout, max_bytes))
        return self.handler(url)


def svc_for(tmp_path, transport=None, online=True, **kw):
    return PaperService(PaperStore(tmp_path / "p.sqlite"), clock=clock, transport=transport,
                        online=online, **kw)


# ------------------------------------------------------------------ safety
@pytest.mark.parametrize("url", [
    "http://export.arxiv.org/api/query", "https://localhost/x", "https://127.0.0.1/x",
    "https://evil.com/export.arxiv.org", "https://export.arxiv.org@evil.com/", "file:///etc/passwd",
    "https://export.arxiv.org:8443/x", "https://169.254.169.254/", "https://arxiv.org/a b",
    "https://api.openalex.org.evil.com/works"])
def test_validate_url_rejects(url):
    with pytest.raises(UrlNotAllowed):
        validate_url(url)


def test_validate_url_accepts_providers():
    for u in ("https://export.arxiv.org/api/query?x=1", "https://api.openalex.org/works",
              "https://arxiv.org/pdf/1706.00001v1"):
        assert validate_url(u) == u


@pytest.mark.parametrize("bad", ["../../etc/passwd", "arxiv:../x", "doi:10.1000/a b", "", "openalex:X1",
                                 "local:zz", "arxiv:1706.00001v", "a" * 400])
def test_canonical_paper_id_rejects(bad):
    with pytest.raises(PaperValidationError):
        canonical_paper_id(bad)


def test_canonical_paper_id_forms():
    assert canonical_paper_id("arXiv:1706.00001v2") == ("arxiv:1706.00001", 2)
    assert canonical_paper_id("https://doi.org/10.1000/ABC") == ("doi:10.1000/abc", None)
    assert canonical_paper_id("hep-th/9901001") == ("arxiv:hep-th/9901001", None)


def test_service_never_requests_non_allowlisted_url(tmp_path):
    t = Fake()
    s = svc_for(tmp_path, t)
    with pytest.raises(UrlNotAllowed):
        s._fetch("https://localhost/secret", "arxiv", "x")
    assert t.calls == []


# ------------------------------------------------------------------ parsing
def test_arxiv_parse_normalises():
    c = parse_arxiv_feed(feed(entry(2, doi="10.1000/XYZ")), NOW)[0]
    assert (c.paper_id, c.version, c.title) == (f"arxiv:{BASE}", 2, "Drift after news")
    assert c.doi == "10.1000/xyz" and c.version_key == "v2"
    assert c.first_public_date.tzinfo and c.version_date > c.first_public_date
    assert c.urls["pdf"] == f"https://arxiv.org/pdf/{BASE}v2" and c.card_hash is None
    assert c.sealed().card_hash == c.sealed().compute_hash()


@pytest.mark.parametrize("body", [
    b"<feed", b"<html/>", b'<!DOCTYPE feed [<!ENTITY a "b">]><feed xmlns="http://www.w3.org/2005/Atom"/>',
    feed("<entry><id>http://arxiv.org/api/errors#x</id><summary>bad</summary></entry>"),
    feed("<entry><id>http://arxiv.org/abs/nonsense</id></entry>"),
    feed(entry(1).replace("<published>2017-06-01T10:00:00Z", "<published>2030-01-01T00:00:00Z")),
    feed(entry(1).replace("<updated>2017-06-01T10:00:00Z</updated>", "<updated>junk</updated>")),
])
def test_arxiv_parse_malformed(body):
    with pytest.raises(ProviderError):
        parse_arxiv_feed(body, NOW)


def test_openalex_parse_and_malformed():
    c = parse_openalex_works(oa_json(oa_work()), NOW)[0]
    assert c.abstract == "Hello world" and c.openalex_id == "W1" and c.version_key == "pub"
    assert c.version_date.isoformat().startswith("2019-01-01T23:59:59")  # date-only => end of day
    pre = parse_openalex_works(oa_json(oa_work(typ="preprint", version="submittedVersion")), NOW)[0]
    assert pre.version_date is None and pre.first_public_date is not None
    ax = parse_openalex_works(oa_json(oa_work(doi="https://doi.org/10.48550/arXiv.1706.00001")), NOW)[0]
    assert ax.arxiv_id == BASE and ax.paper_id == f"arxiv:{BASE}"
    for bad in (b"{", b"[]", b'{"results": [1]}', oa_json({"id": "x", "display_name": "t"}),
                oa_json(oa_work(date="2019-13-45"))):
        with pytest.raises(ProviderError):
            parse_openalex_works(bad, NOW)


def test_check_status_errors():
    with pytest.raises(AuthError, match="OPENALEX_API_KEY"):
        check_status(TransportResponse(403, b""), "https://api.openalex.org/works?api_key=SECRET", "openalex")
    try:
        check_status(TransportResponse(401, b""), "https://api.openalex.org/works?api_key=SECRET", "openalex")
    except AuthError as e:
        assert "SECRET" not in str(e)
    with pytest.raises(ProviderError):
        check_status(TransportResponse(500, b""), "https://export.arxiv.org/x", "arxiv")


# ------------------------------------------------------- get / as-of / cache
def test_get_contemporary_then_offline_cache(tmp_path):
    t = Fake()
    card = svc_for(tmp_path, t).get(f"arxiv:{BASE}")
    assert card.version == 3 and card.latest_version == 3 and card.card_hash
    assert t.kw[0][0] > 0 and t.kw[0][1] > 0  # finite timeout and body bound passed
    n = len(t.calls)
    off = PaperService(PaperStore(tmp_path / "p.sqlite"), clock=clock, transport=t, online=False)
    assert off.get(f"arxiv:{BASE}").card_hash == card.card_hash
    assert len(t.calls) == n
    with pytest.raises(OfflineError):
        off.get("arxiv:1706.99999")
    assert len(t.calls) == n


def test_offline_service_never_calls_transport(tmp_path):
    t = Fake()
    s = svc_for(tmp_path, t, online=False)
    with pytest.raises(OfflineError):
        s.get(f"arxiv:{BASE}")
    assert s.search("drift").cards == [] and t.calls == []


def test_as_of_picks_version_not_modern_revision(tmp_path):
    t = Fake()
    s = svc_for(tmp_path, t)
    assert s.get(f"arxiv:{BASE}", as_of="2018-06-01").version == 2
    assert len(t.calls) == 2  # latest + one bisection step
    assert s.get(f"arxiv:{BASE}", as_of="2017-12-01").version == 1
    assert s.get(f"arxiv:{BASE}", as_of="2026-01-01").version == 3
    for early in ("2017-05-01", "2017-06-01"):  # before v1's own timestamp
        with pytest.raises(NotFound):
            s.get(f"arxiv:{BASE}", as_of=early)
    with pytest.raises(NotFound):  # explicit modern version under an old cutoff
        s.get(f"arxiv:{BASE}v3", as_of="2018-06-01")
    assert s.get(f"arxiv:{BASE}", as_of="2018-06-01").abstract == "abstract text v2"


def test_offline_revision_lookahead_fails_closed(tmp_path):
    t = Fake()
    svc_for(tmp_path, t).get(f"arxiv:{BASE}v1")  # only v1 cached, newer versions unknown
    off = PaperService(PaperStore(tmp_path / "p.sqlite"), clock=clock, online=False)
    with pytest.raises(NotFound):  # v2 (2018-03) may have existed by 2018-06: cannot prove
        off.get(f"arxiv:{BASE}", as_of="2018-06-01")
    on = svc_for(tmp_path, t)
    on.get(f"arxiv:{BASE}", refresh=True)  # latest (v3) now known, v2 still unknown
    assert off.get(f"arxiv:{BASE}", as_of="2026-01-01").version == 3  # latest known at/after cutoff
    with pytest.raises(NotFound):  # v2 could be <= 2017-12-01 for all the cache can prove
        off.get(f"arxiv:{BASE}", as_of="2017-12-01")
    on.get(f"arxiv:{BASE}v2")
    n = len(t.calls)
    assert off.get(f"arxiv:{BASE}", as_of="2017-12-01").version == 1  # v2 cached and later
    assert off.get(f"arxiv:{BASE}", as_of="2018-06-01").version == 2
    assert len(t.calls) == n


def test_bisection_budget_excludes(tmp_path):
    dates = {1: DATES[1], 2: "2018-01-01T00:00:00Z", 3: "2018-02-01T00:00:00Z", 4: "2025-01-01T00:00:00Z"}

    def handler(url):
        ident = parse_qs(urlsplit(url).query)["id_list"][0]
        v = int(ident.split("v")[1]) if "v" in ident else 4
        e = entry(1).replace("v1</id>", f"v{v}</id>").replace(DATES[1] + "</updated>", dates[v] + "</updated>")
        return TransportResponse(200, feed(e))

    s = svc_for(tmp_path, Fake(handler), max_version_lookups=0)
    with pytest.raises(NotFound):
        s.get(f"arxiv:{BASE}", as_of="2018-01-15")
    assert any("budget" in w for w in s.warnings)


def test_temporal_validation(tmp_path):
    s = svc_for(tmp_path, Fake())
    with pytest.raises(TemporalError):
        s.get(f"arxiv:{BASE}", as_of=datetime(2018, 1, 1))  # naive
    with pytest.raises(TemporalError):
        s.get(f"arxiv:{BASE}", as_of="2030-01-01")  # future
    with pytest.raises(TemporalError):
        s.get(f"arxiv:{BASE}", as_of="2018-13-01")
    bad = PaperService(PaperStore(":memory:"), clock=lambda: datetime(2026, 1, 1), online=False)
    with pytest.raises(TemporalError):
        bad.search("drift", as_of="2020-01-01")


# ------------------------------------------------------------------- search
def test_search_dedups_duplicate_doi_and_arxiv_versions(tmp_path):
    def handler(url):
        if urlsplit(url).hostname == "export.arxiv.org":
            return TransportResponse(200, feed(entry(3, doi="10.1000/xyz")))
        return TransportResponse(200, oa_json(oa_work(), oa_work("W2", doi="https://doi.org/10.48550/arXiv." + BASE)))

    s = svc_for(tmp_path, Fake(handler))
    r = s.search("drift after news", limit=5)
    assert r.mode == "contemporary" and r.origin == "network"
    assert [c.paper_id for c in r.cards] == [f"arxiv:{BASE}"]
    assert sorted(c.version_key for c in s.store.rows(f"arxiv:{BASE}")) == ["pub", "v3"]
    assert s.store.resolve("doi:10.1000/xyz") == f"arxiv:{BASE}"


def test_historical_search_filters_revisions_and_unknown_dates(tmp_path):
    def handler(url):
        p = urlsplit(url)
        if p.hostname == "export.arxiv.org" and "search_query" in p.query:
            return TransportResponse(200, feed(entry(3, doi="10.1000/xyz")))
        if p.hostname == "api.openalex.org":
            return TransportResponse(200, oa_json(
                oa_work("W9", doi="https://doi.org/10.5555/pre", typ="preprint", version="submittedVersion",
                        date="2016-01-01", title="Unversioned preprint")))
        return Fake().default(url)

    t = Fake(handler)
    s = svc_for(tmp_path, t)
    r = s.search("drift", as_of="2018-06-01")
    assert r.mode == "historical_replay"
    assert [(c.paper_id, c.version) for c in r.cards] == [(f"arxiv:{BASE}", 2)]  # W9 unknown date: excluded
    assert any("submittedDate" in u for u in t.calls)
    assert any("to_publication_date" in u for u in t.calls)
    contemporary = s.search("drift")
    assert {c.paper_id for c in contemporary.cards} == {f"arxiv:{BASE}", "doi:10.5555/pre"}


def test_search_partial_failure_auth_and_secret_hygiene(tmp_path):
    def handler(url):
        if urlsplit(url).hostname == "api.openalex.org":
            return TransportResponse(403, b"")
        return TransportResponse(200, feed(entry(3)))

    t = Fake(handler)
    s = svc_for(tmp_path, t, openalex_api_key="SECRETKEY123")
    r = s.search("drift")
    assert len(r.cards) == 1 and any("OPENALEX_API_KEY" in w for w in r.warnings)
    assert any("api_key=SECRETKEY123" in u for u in t.calls)
    assert all("SECRETKEY123" not in w for w in r.warnings)
    s.store.close()
    assert b"SECRETKEY123" not in (tmp_path / "p.sqlite").read_bytes()
    only = svc_for(tmp_path, Fake(lambda u: TransportResponse(403, b"")),
                   providers=("openalex",), openalex_api_key="SECRETKEY123")
    with pytest.raises(AuthError) as ei:
        only.search("drift")
    assert "SECRETKEY123" not in str(ei.value)


def test_search_errors_and_limits(tmp_path):
    s = svc_for(tmp_path, Fake(lambda u: TransportResponse(200, b"<nope")), providers=("arxiv",))
    with pytest.raises(ProviderError):
        s.search("drift")
    for bad in (0, 51, "3"):
        with pytest.raises(PaperValidationError):
            s.search("drift", limit=bad)
    for q in ("", "   ", "!!!", "x" * 301):
        with pytest.raises(PaperValidationError):
            s.search(q)


def test_oversized_body_rejected(tmp_path):
    s = svc_for(tmp_path, Fake(lambda u: TransportResponse(200, b"x" * 100)), max_body_bytes=10)
    with pytest.raises(ProviderError):
        s.search("drift")


def test_query_cache_and_local_search_offline(tmp_path):
    t = Fake()
    svc_for(tmp_path, t).search("drift after news", limit=3)
    off = PaperService(PaperStore(tmp_path / "p.sqlite"), clock=clock, online=False)
    r = off.search("drift after news", limit=3)
    assert r.origin == "query_cache" and r.cards
    r2 = off.search("news", limit=5)  # different query: token scan of cache
    assert r2.origin == "local_cache" and r2.cards
    assert off.search("zzzunmatched").cards == []
    assert off.search("news", as_of="2017-01-01").cards == []


# ------------------------------------------------------- content and claims
TEXT = ("Abstract\nWe study drift.\n\f1. Introduction\nWe find that stocks drift after earnings news announcements.\n"
        "\f3. Data\nThe sample consists of all US common stocks from 1990 to 2010.\n"
        "Momentum is defined as the past twelve month return.\n")
FIRST = datetime(2020, 1, 1, tzinfo=timezone.utc)


def local_card(tmp_path, s, text=TEXT, name="paper.txt"):
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return s.ingest_local(f, title="Local drift", first_public_date=FIRST, version_date=FIRST,
                          authors=["Ann Lee"])


def test_ingest_local_and_grounded_claims(tmp_path):
    s = svc_for(tmp_path, None, online=False)
    card = local_card(tmp_path, s)
    assert card.text_status == "full_text" and card.access_status.value == "full_text"
    assert card.content_hash and card.original_hash and not card.content_truncated
    q = "The sample consists of all US common stocks from 1990 to 2010."
    c2 = s.add_claim(card.paper_id, kind="universe", quote=q)
    cl = c2.claims[0]
    assert (cl.page, cl.section, cl.extraction) == (3, "3. Data", "manual")
    assert s.text_of(c2)[cl.char_start:cl.char_end] == q and cl.source_hash == c2.content_hash
    assert s.verify_claims(c2) == []
    assert local_card(tmp_path, s).claims[0].claim_id == cl.claim_id  # re-ingest keeps claims
    with pytest.raises(ClaimError):
        s.add_claim(card.paper_id, kind="universe", quote="not in the paper")
    with pytest.raises(ClaimError):
        s.add_claim(card.paper_id, kind="bogus", quote=q)
    # tampering with stored text is detected
    s.store._db.execute("UPDATE contents SET text=?", (TEXT.replace("1990", "1991"),))
    assert s.verify_claims(s.get(card.paper_id))


def test_metadata_only_cannot_ground_claims_and_abstract_can(tmp_path):
    def handler(url):
        return TransportResponse(200, feed(entry(3).replace("<summary>abstract text v3</summary>", "")))

    s = svc_for(tmp_path, Fake(handler))
    with pytest.raises(ClaimError):
        s.add_claim(f"arxiv:{BASE}", kind="economic_claim", quote="anything")
    with pytest.raises(ClaimError):
        s.extract_claims(f"arxiv:{BASE}")
    s2 = svc_for(tmp_path / "b", Fake())
    c = s2.add_claim(f"arxiv:{BASE}", kind="economic_claim", quote="abstract text", source="abstract")
    assert c.claims[0].source == "abstract" and c.claims[0].page is None and s2.verify_claims(c) == []


def test_extract_claims_heuristic_is_extractive(tmp_path):
    s = svc_for(tmp_path, None, online=False)
    card = s.extract_claims(local_card(tmp_path, s).paper_id)
    kinds = {c.kind for c in card.claims}
    assert {"economic_claim", "universe"} <= kinds
    assert all(c.extraction == "heuristic" and c.text == c.quote for c in card.claims)
    assert s.verify_claims(card) == []


def test_ingest_local_validation(tmp_path):
    s = svc_for(tmp_path, None, online=False, max_body_bytes=50)
    big = tmp_path / "big.txt"
    big.write_text("x" * 100)
    with pytest.raises(PaperValidationError):
        s.ingest_local(big, title="t", first_public_date=FIRST)
    ok = tmp_path / "ok.txt"
    ok.write_text("hello")
    link = tmp_path / "link.txt"
    try:
        os.symlink(ok, link)
    except OSError:
        pass
    else:
        with pytest.raises(PaperValidationError):
            s.ingest_local(link, title="t", first_public_date=FIRST)
    with pytest.raises(PaperValidationError):
        s.ingest_local(tmp_path / "missing.txt", title="t", first_public_date=FIRST)
    with pytest.raises(Exception):
        s.ingest_local(ok, title="t", first_public_date=datetime(2020, 1, 1))  # naive
    binf = tmp_path / "b.bin"
    binf.write_bytes(b"\xff\xfe\x00\x01")
    with pytest.raises(PaperValidationError):
        s.ingest_local(binf, title="t", first_public_date=FIRST)


def test_text_truncation_bounded_and_flagged(tmp_path):
    s = svc_for(tmp_path, None, online=False, max_text_chars=20)
    card = local_card(tmp_path, s)
    assert card.content_truncated and card.content_chars == 20
    assert card.original_hash != card.content_hash
    spec = s.create_replication_spec(card.paper_id, requested_label="MECHANISM_TRANSFER")
    assert spec["label"] == "INCOMPLETE" and any("truncated" in r for r in spec["reasons"])


def test_arxiv_full_text_incomplete_when_not_pdf(tmp_path):
    t = Fake()
    s = svc_for(tmp_path, t)
    card = s.ingest_full_text(f"arxiv:{BASE}v2")
    assert any(u.startswith(f"https://arxiv.org/pdf/{BASE}v2") for u in t.calls)
    assert card.text_status == "incomplete:not_a_pdf" and card.access_status.value == "abstract"
    assert card.content_hash is None and card.original_hash


def test_openalex_only_has_no_full_text_source(tmp_path):
    s = svc_for(tmp_path, Fake(lambda u: TransportResponse(200, oa_json(oa_work(doi="https://doi.org/10.5555/j")))))
    card = s.get("doi:10.5555/j")
    assert card.openalex_id == "W1" and card.urls["doi"].startswith("https://doi.org/")
    assert s.ingest_full_text("doi:10.5555/j").text_status == "incomplete:no_permitted_full_text_source"


# ----------------------------------------------- replication and method cards
FULL = ("The universe is all NYSE stocks. The period is 1990 to 2010. The target is next month return. "
        "The horizon is one month. Signal is defined as the lagged ratio. Timing is end of month. "
        "We run Fama-MacBeth regressions. The baseline is a seasonal random walk. We find drift exists. "
        "The mechanism is slow information diffusion.")


def grounded(tmp_path, kinds):
    tmp_path.mkdir(parents=True, exist_ok=True)
    s = svc_for(tmp_path, None, online=False)
    card = local_card(tmp_path, s, FULL)
    quotes = {"universe": "all NYSE stocks", "period": "1990 to 2010", "target": "next month return",
              "horizon": "one month", "variable_definition": "defined as the lagged ratio",
              "timing": "end of month", "statistical_design": "Fama-MacBeth regressions",
              "baseline": "seasonal random walk", "economic_claim": "drift exists",
              "mechanism": "slow information diffusion"}
    for k in kinds:
        card = s.add_claim(card.paper_id, kind=k, quote=quotes[k])
    return s, card


def test_replication_spec_labels(tmp_path):
    allk = ["universe", "period", "target", "horizon", "variable_definition", "timing",
            "statistical_design", "baseline", "economic_claim"]
    s, card = grounded(tmp_path, allk)
    spec = s.create_replication_spec(card.paper_id, requested_label="EXACT_REPLICATION")
    assert spec["label"] == "EXACT_REPLICATION" and spec["llm_called"] is False
    assert spec["evidence"]["universe"][0]["page"] == 1 and spec["status"].startswith("draft")
    assert s.store.get_artifact("replication_spec", spec["spec_id"])["label"] == spec["label"]
    with pytest.raises(PaperValidationError):
        s.create_replication_spec(card.paper_id, requested_label="EXACT_REPLICATION", deviations=["x"])
    with pytest.raises(PaperValidationError):
        s.create_replication_spec(card.paper_id, requested_label="CLOSE_REPLICATION", deviations=["x"])
    close = s.create_replication_spec(card.paper_id, requested_label="CLOSE_REPLICATION",
                                      deviations=["seasonal baseline replaced"], comparison_plan="run both")
    assert close["label"] == "CLOSE_REPLICATION" and close["deviations"]
    s2, c2 = grounded(tmp_path / "m", allk[:-2])
    inc = s2.create_replication_spec(c2.paper_id, requested_label="EXACT_REPLICATION")
    assert inc["label"] == "INCOMPLETE" and set(inc["missing_claim_kinds"]) == {"baseline", "economic_claim"}
    mt2 = s2.create_replication_spec(c2.paper_id, requested_label="METHOD_TRANSFER")
    assert mt2["label"] == "INCOMPLETE" and mt2["missing_claim_kinds"] == ["baseline"]
    mt = s.create_replication_spec(card.paper_id, requested_label="METHOD_TRANSFER")
    assert mt["label"] == "METHOD_TRANSFER"


def test_replication_spec_incomplete_for_metadata_only_and_as_of(tmp_path):
    s = svc_for(tmp_path, Fake())
    spec = s.create_replication_spec(f"arxiv:{BASE}", requested_label="MECHANISM_TRANSFER",
                                     as_of="2018-06-01")
    assert spec["label"] == "INCOMPLETE" and spec["research_mode"] == "historical_replay"
    assert spec["evidence"] == {} and spec["version_key"] == "v2"


def test_method_card_immutable_and_validated(tmp_path):
    s, card = grounded(tmp_path, ["universe"])
    m = s.create_method_card("ridge_reg", "1", paper_id=card.paper_id, baselines=["seasonal"])
    assert m["source"]["card_hash"] == s.get(card.paper_id).card_hash
    with pytest.raises(PaperValidationError):
        s.create_method_card("ridge_reg", "1")
    with pytest.raises(PaperValidationError):
        s.create_method_card("../x", "1")
