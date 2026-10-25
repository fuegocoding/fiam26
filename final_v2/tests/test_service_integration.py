import datetime as dt
import json

import pytest

from pod_platform.agents.durable_cache import DurableExtractionCache
from pod_platform.papers.service import PaperService
from pod_platform.pods.charters import pod_spec
from pod_platform.pods.demo import demo_provider
from pod_platform.pods.runner import PodResearchRunner, PodWorker
from pod_platform.research.jobs import ResearchQueue
from pod_platform.research.memory import PodMemory
from pod_platform.services.app import build_server

NOW = dt.datetime(2026, 1, 15, 12, tzinfo=dt.timezone.utc)
EVIDENCE = {"question": "Does value spread predict returns?", "executors": ["value_spread_ic"],
            "source_ids": ["comp_chars_panel"]}


def clock():
    return NOW


def call(server, tool, args=None):
    response = server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                "params": {"name": tool, "arguments": args or {}}})["result"]
    return response, response["content"][0]["text"]


def start(tmp_path, pod, **kw):
    server = build_server(tmp_path, pod, clock=clock, **kw)
    server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    return server


def submit(server, kind="propose", payload=None, **extra):
    return call(server, "research.submit", {"kind": kind, "payload": payload or EVIDENCE, **extra})


def seed_paper(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("We study value spreads. The universe is US common stocks.", encoding="utf-8")
    papers = PaperService(tmp_path / "shared" / "papers.sqlite", clock=clock)
    try:
        card = papers.ingest_local(source, title="Value spreads and returns",
                                   first_public_date=NOW - dt.timedelta(days=90))
    finally:
        papers.store.close()
    return card.paper_id


def test_private_memory_and_jobs_are_isolated_across_pods(tmp_path):
    first, second = start(tmp_path, "fundamentals"), start(tmp_path, "events")
    try:
        result, body = call(first, "memory.note", {
            "content": {"idea": "a private failed hypothesis"}, "idempotency_key": "idea1"})
        assert not result["isError"]
        assert len(json.loads(call(first, "memory.list")[1])) == 1
        assert json.loads(call(second, "memory.list")[1]) == []
        result, body = submit(first)
        assert not result["isError"], body
        job = json.loads(body)
        assert job["namespace"] == "pod_fundamentals" and job["kind"] == "propose"
        assert len(json.loads(call(first, "research.list")[1])) == 1
        assert json.loads(call(second, "research.list")[1]) == []
        assert call(second, "research.show", {"job_id": job["job_id"]})[0]["isError"]
        assert call(first, "research.show", {"job_id": job["job_id"]})[0]["isError"] is False
    finally:
        first.close()
        second.close()


def test_mcp_and_cli_share_one_queue_with_namespace_enforced(tmp_path):
    server = start(tmp_path, "valuation")
    try:
        job = json.loads(submit(server)[1])
        queue = ResearchQueue(tmp_path / "queue.sqlite")  # the file run_pod_research.py uses
        assert [j.job_id for j in queue.list(namespace="pod_valuation")] == [job["job_id"]]
        assert not (tmp_path / "pods" / "pod_valuation" / "jobs.sqlite").exists()
        other = queue.submit("pod_events", "propose", {"question": "q"})
        cli_job = queue.submit("pod_valuation", "propose", {"question": "cli job"})
        listed = {j["job_id"] for j in json.loads(call(server, "research.list")[1])}
        assert listed == {job["job_id"], cli_job.job_id} and other.job_id not in listed
        assert call(server, "research.show", {"job_id": other.job_id})[0]["isError"]
        assert json.loads(submit(server)[1])["job_id"] == job["job_id"]  # idempotent resubmission
    finally:
        server.close()


@pytest.mark.parametrize("kind,payload", [
    ("proposal", EVIDENCE),                                        # not a worker kind
    ("propose", {**EVIDENCE, "namespace": "pod_events"}),          # no namespace override
    ("propose", {"question": "q", "executors": ["x"]}),            # no evidence at all
    ("propose", {**EVIDENCE, "source_ids": ["no_such_source"]}),
    ("propose", {**EVIDENCE, "paper_ids": ["arxiv:0000.00000"]}),  # unknown/uncached paper, offline
    ("propose", {**EVIDENCE, "paper_cards": [{"paper_id": "forged"}]}),  # callers cannot supply cards
    ("propose", {**EVIDENCE, "mode": "historical"}),               # needs a registered checkpoint
    ("propose", {**EVIDENCE, "research_asof": "2030-01-01T00:00:00+00:00"}),
    ("propose", {**EVIDENCE, "executors": ["Bad Code; rm -rf"]}),
    ("experiment", {"draft_entry_id": "mem_missing"}),
    ("experiment", {"draft_entry_id": "mem_x", "variants": [f"v{i}" for i in range(9)]}),
])
def test_invalid_kinds_and_payloads_are_rejected_and_never_queued(tmp_path, kind, payload):
    server = start(tmp_path, "fundamentals")
    try:
        assert submit(server, kind, payload)[0]["isError"]
        assert json.loads(call(server, "research.list")[1]) == []
    finally:
        server.close()


def test_cli_worker_processes_mcp_job_and_experiments_stay_in_own_pod(tmp_path):
    server, other = start(tmp_path, "fundamentals"), start(tmp_path, "events")
    try:
        job = json.loads(submit(server)[1])
        spec = pod_spec("fundamentals")
        runner = PodResearchRunner(spec, demo_provider(), DurableExtractionCache(tmp_path / "cache.sqlite"),
                                   PodMemory(tmp_path / "memory", spec.namespace), clock=clock)
        outcome = PodWorker(ResearchQueue(tmp_path / "queue.sqlite", clock=clock),
                            {spec.namespace: runner}).run_once()
        assert outcome.state == "succeeded", outcome.error
        shown = json.loads(call(server, "research.show", {"job_id": job["job_id"]})[1])
        assert shown["job"]["state"] == "succeeded"
        assert shown["result"]["provider_is_mock"] is True and shown["result"]["admitted"] is False
        drafts = json.loads(call(server, "memory.list")[1])
        assert [d["kind"] for d in drafts] == ["draft_proposal"] and drafts[0]["evaluated"] is False
        assert json.loads(call(other, "memory.list")[1]) == []
        draft_id = shown["result"]["memory_entry_id"]
        assert submit(other, "experiment", {"draft_entry_id": draft_id})[0]["isError"]
        result, body = submit(server, "experiment", {"draft_entry_id": draft_id, "variants": ["base", "base"]})
        assert not result["isError"], body
        assert json.loads(body)["payload"] == {"draft_entry_id": draft_id, "variants": ["base"]}
        assert submit(server, "experiment", {"draft_entry_id": draft_id, "variants": ["nope"]})[0]["isError"]
    finally:
        server.close()
        other.close()


def test_paper_tools_use_cache_offline_and_feed_real_evidence(tmp_path):
    paper_id = seed_paper(tmp_path)
    server = start(tmp_path, "valuation")
    try:
        found = json.loads(call(server, "papers.search", {"query": "value spreads"})[1])
        assert [c["paper_id"] for c in found["cards"]] == [paper_id]
        assert json.loads(call(server, "papers.get", {"paper_id": paper_id})[1])["title"] == "Value spreads and returns"
        read = json.loads(call(server, "papers.read", {"paper_id": paper_id, "max_chars": 100})[1])
        assert read["text"].startswith("We study value spreads") and read["limitations"] == []
        assert call(server, "papers.read", {"paper_id": "arxiv:0000.00000"})[0]["isError"]
        assert call(server, "papers.ingest", {"paper_id": "arxiv:0000.00000"})[0]["isError"]  # offline
        spec = json.loads(call(server, "papers.replication", {"paper_id": paper_id})[1])
        assert spec["label"] == "INCOMPLETE" and spec["llm_called"] is False
        queued = json.loads(submit(server, payload={**EVIDENCE, "paper_ids": [paper_id]})[1])
        cards = queued["payload"]["paper_cards"]
        assert cards == [f"{paper_id}.unversioned"] and queued["payload"]["source_cards"] == ["comp_chars_panel"]
        stored = ResearchQueue(tmp_path / "queue.sqlite").get(queued["job_id"]).payload
        assert stored["paper_cards"][0]["title"] == "Value spreads and returns"
        assert stored["mode"] == "contemporary" and stored["executors"] == ["value_spread_ic"]
    finally:
        server.close()


def test_registered_experiments_and_features_are_declared_and_not_admitting(tmp_path):
    server = start(tmp_path, "dynamics")  # pod alias resolves to the pod namespace
    try:
        plan = {"experiment_id": "e1", "owner": "pod_dynamics", "hypothesis_id": "h",
                "registered_at": (NOW - dt.timedelta(hours=1)).isoformat(), "target_id": "growth",
                "source_ids": ["comp_chars_panel"], "variant_budget": 1}
        first = json.loads(call(server, "experiments.register", {"plan": plan})[1])
        assert first["admitted"] is False
        assert json.loads(call(server, "experiments.register", {"plan": plan})[1]) == first
        assert call(server, "experiments.register", {"plan": {**plan, "target_id": "other"}})[0]["isError"]
        future = {**plan, "experiment_id": "e2", "registered_at": (NOW + dt.timedelta(hours=1)).isoformat()}
        assert call(server, "experiments.register", {"plan": future})[0]["isError"]
        result, body = call(server, "features.materialize", {
            "recipe": {"feature_id": "f", "source_id": "comp_chars_panel", "metric": "roe", "operation": "latest"},
            "observations": [{"observation_id": "o1", "security_id": "A", "source_id": "comp_chars_panel",
                              "metric": "roe", "period_end": "2025-12-31T00:00:00+00:00",
                              "available_at": "2026-01-05T00:00:00+00:00",
                              "version_known_at": "2026-01-05T00:00:00+00:00", "value": 0.1}],
            "formation_at": "2026-01-10T00:00:00+00:00", "security_ids": ["A"]})
        assert not result["isError"], body
    finally:
        server.close()


def test_services_cannot_override_pod_or_claim_evaluation(tmp_path):
    server = start(tmp_path, "fundamentals")
    try:
        assert call(server, "memory.note", {
            "content": {"admitted": True}, "idempotency_key": "n1", "namespace": "pod_events"})[0]["isError"]
        result, body = call(server, "memory.note", {
            "content": {"admitted": True}, "idempotency_key": "n2"})
        assert not result["isError"]
        assert json.loads(body)["evaluated"] is False
        result, _ = call(server, "experiments.register", {"plan": {
            "experiment_id": "x", "owner": "pod_events", "hypothesis_id": "h",
            "registered_at": (NOW - dt.timedelta(hours=1)).isoformat(),
            "target_id": "growth", "source_ids": ["comp_chars_panel"], "variant_budget": 1}})
        assert result["isError"]
        status = json.loads(call(server, "platform.status")[1])
        assert status["namespace"] == "pod_fundamentals"
        assert status["automatic_admission"] is False and status["online_papers"] is False
        names = set(server.tools)
        assert not any(bad in name for name in names for bad in ("shell", "admit", "key", "exec", "bash"))
        assert {"papers.search", "papers.read", "papers.ingest", "papers.replication", "sources.list",
                "memory.note", "research.submit", "experiments.evaluate", "features.materialize"} <= names
    finally:
        server.close()


def test_paper_search_is_offline_by_default_and_sources_visible(tmp_path):
    server = start(tmp_path, "valuation")
    try:
        result, body = call(server, "papers.search", {"query": "earnings announcement drift"})
        assert not result["isError"]
        assert json.loads(body)["cards"] == []
        cards = json.loads(call(server, "sources.list")[1])
        assert any(card["source_id"] == "comp_chars_panel" and card["formation_eligible"] for card in cards)
        assert any(card["dataset_id"] == "CA0056" and not card["formation_eligible"] for card in cards)
    finally:
        server.close()
