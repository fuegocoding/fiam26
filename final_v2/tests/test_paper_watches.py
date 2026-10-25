import datetime as dt
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pod_platform.agents.durable_cache import DurableExtractionCache
from pod_platform.papers.models import OfflineError, PaperCard
from pod_platform.papers.watches import PaperWatches, paper_evidence
from pod_platform.pods.charters import pod_spec
from pod_platform.pods.demo import demo_provider
from pod_platform.pods.runner import PodResearchRunner, PodWorker
from pod_platform.pods.schema import normalise_evidence
from pod_platform.research.jobs import ResearchQueue
from pod_platform.research.memory import PodMemory

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "research_watch.py"
_spec = importlib.util.spec_from_file_location("research_watch_cli", SCRIPT)
research_watch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(research_watch)

T0 = dt.datetime(2026, 3, 1, 12, tzinfo=dt.timezone.utc)


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += dt.timedelta(**kw)


def card(n, version="v1", **kw):
    return PaperCard(paper_id=f"arxiv:2001.{n:05d}", title=f"Paper {n} on value spreads",
                     abstract=f"Abstract {n}", version_key=version, source="arxiv",
                     first_public_date=T0 - dt.timedelta(days=30), version_date=T0 - dt.timedelta(days=10),
                     retrieved_at=T0 - dt.timedelta(days=1), **kw).sealed()


class FakePapers:
    def __init__(self, cards=()):
        self.cards, self.error, self.queries = list(cards), None, []

    def search(self, query, *, limit=10, as_of=None):
        self.queries.append((query, limit, as_of))
        if self.error:
            raise self.error
        return SimpleNamespace(cards=self.cards[:limit], origin="query_cache", warnings=[])


@pytest.fixture
def env(tmp_path):
    clock, papers = Clock(), FakePapers([card(1), card(2)])
    queue = ResearchQueue(tmp_path / "queue.sqlite", clock=clock)
    watches = PaperWatches(tmp_path / "watches.sqlite", queue=queue, papers=papers, clock=clock)
    return SimpleNamespace(clock=clock, papers=papers, queue=queue, watches=watches, path=tmp_path)


def add(env, pod="valuation", watch_id="spread", **kw):
    kw.setdefault("cadence_hours", 24)
    return env.watches.add(pod, watch_id, "value spread", executors=["value_spread_ic"], **kw)


def test_due_cadence_and_dedup_of_paper_versions(env):
    add(env)
    first = env.watches.tick()
    assert [r["status"] for r in first] == ["ok"] and len(first[0]["jobs"]) == 1
    assert env.watches.tick() == []  # not due again yet
    env.clock.advance(hours=23)
    assert env.watches.tick() == []
    env.clock.advance(hours=2)
    second = env.watches.tick()
    assert second[0]["status"] == "ok" and second[0]["jobs"] == [] and second[0]["new"] == 0
    env.papers.cards.append(card(1, version="v2"))  # a new version of a known paper is new evidence
    env.papers.cards.append(card(3))
    env.clock.advance(hours=25)
    third = env.watches.tick()
    assert third[0]["new"] == 2 and len(third[0]["jobs"]) == 1
    jobs = env.queue.list(namespace="pod_valuation")
    assert len(jobs) == 2 and all(j.kind == "propose" and j.state == "queued" for j in jobs)


def test_payload_uses_actual_returned_papers_and_runs_in_worker_demo(env):
    add(env)
    env.watches.tick()
    job = env.queue.list(namespace="pod_valuation")[0]
    ids = [c["evidence_id"] for c in job.payload["paper_cards"]]
    assert ids == ["arxiv:2001.00001.v1", "arxiv:2001.00002.v1"]
    assert job.payload["paper_cards"][0]["abstract"] == "Abstract 1"
    assert job.payload["executors"] == ["value_spread_ic"] and job.payload["mode"] == "contemporary"
    assert [i.evidence_id for i in normalise_evidence(job.payload["paper_cards"], "paper_card")] == ids
    assert job.payload["research_asof"] == T0.isoformat()  # tick clock, not the card dates
    assert {c["available_at"] for c in job.payload["paper_cards"]} == {(T0 - dt.timedelta(days=10)).isoformat()}
    spec = pod_spec("valuation")
    runner = PodResearchRunner(spec, demo_provider(), DurableExtractionCache(env.path / "cache.sqlite"),
                               PodMemory(env.path / "memory", spec.namespace), clock=env.clock)
    outcome = PodWorker(env.queue, {spec.namespace: runner}).run_once()
    assert outcome.state == "succeeded", outcome.error
    assert outcome.result["provider_is_mock"] is True and outcome.result["evaluated"] is False
    assert outcome.result["admitted"] is False


def test_interrupted_enqueue_retry_at_later_clock_reuses_job(env):
    add(env)
    w = env.watches.list()[0]
    cards = [card(1), card(2)]
    first = env.watches._enqueue(w, cards, env.clock.now)
    env.clock.advance(hours=3)
    again = env.watches._enqueue(w, cards, env.clock.now)  # crash before `seen` was marked
    assert again.job_id == first.job_id and len(env.queue.list()) == 1
    assert again.payload["research_asof"] == T0.isoformat()


def test_restart_replay_does_not_requeue(env):
    add(env)
    env.watches.tick()
    reopened = PaperWatches(env.path / "watches.sqlite", queue=env.queue, papers=env.papers, clock=env.clock)
    env.clock.advance(days=2)
    result = reopened.tick()
    assert result[0]["jobs"] == [] and len(env.queue.list()) == 1


def test_bounded_jobs_defer_the_rest_to_later_ticks(env):
    env.papers.cards = [card(n) for n in range(1, 6)]
    add(env, papers_per_job=2, max_jobs=1)
    first = env.watches.tick()[0]
    assert len(first["jobs"]) == 1 and first["new"] == 5 and first["deferred"] == 3
    env.clock.advance(days=1, minutes=1)
    assert env.watches.tick()[0]["new"] == 3
    assert len(env.queue.list()) == 2
    assert len(env.queue.list()[0].payload["paper_cards"]) == 2


def test_namespace_isolation_and_conflicting_redefinition(env):
    add(env, pod="valuation")
    add(env, pod="events")
    assert {w["namespace"] for w in env.watches.list()} == {"pod_valuation", "pod_events"}
    assert len(env.watches.list("events")) == 1
    results = env.watches.tick("events")
    assert len(results) == 1 and results[0]["namespace"] == "pod_events"
    assert [j.namespace for j in env.queue.list()] == ["pod_events"]
    env.watches.tick("valuation")
    assert len(env.queue.list(namespace="pod_valuation")) == 1
    assert add(env, pod="valuation")["watch_id"] == "spread"  # identical re-add is idempotent
    with pytest.raises(ValueError):
        add(env, pod="valuation", cadence_hours=48)
    with pytest.raises(ValueError):
        env.watches.add("valuation", "Bad Id", "q", cadence_hours=1, executors=["x"])


def test_missing_results_and_failures_are_recorded_not_claimed(env):
    env.papers.cards = []
    add(env)
    empty = env.watches.tick()[0]
    assert empty["status"] == "ok" and empty["returned"] == 0 and empty["jobs"] == []
    assert env.queue.list() == []
    env.papers.error = OfflineError("network access is disabled")
    env.clock.advance(days=2)
    failed = env.watches.tick()[0]
    assert failed["status"] == "failed" and failed["error"]["type"] == "OfflineError"
    watch = env.watches.list()[0]
    assert watch["last_success_at"] == (T0).isoformat()  # still the earlier success
    assert watch["next_due_at"] == (env.clock.now + dt.timedelta(hours=1)).isoformat()
    assert [r["status"] for r in env.watches.runs("valuation", "spread")] == ["failed", "ok"]
    env.papers.error = None
    env.papers.cards = [card(7)]
    env.clock.advance(hours=1)
    assert len(env.watches.tick()[0]["jobs"]) == 1


def test_evidence_is_only_what_the_card_contains():
    bare = PaperCard(paper_id="doi:10.1/x(y)", title="T", version_key="pub", source="openalex",
                     retrieved_at=T0).sealed()
    ev = paper_evidence(bare)
    assert "available_at" not in ev and "abstract" not in ev  # nothing invented
    assert normalise_evidence([ev], "paper_card")[0].evidence_id == "doi:10.1_x_y_.pub"


def test_cli_add_list_tick_offline_with_empty_cache(tmp_path):
    clock = Clock()

    def run(*argv):
        out = io.StringIO()
        code = research_watch.main([*argv, "--state-dir", str(tmp_path)], clock=clock, out=out)
        return code, json.loads(out.getvalue())

    code, added = run("add", "--pod", "dynamics", "--watch-id", "mom", "--topic", "momentum crashes",
                      "--cadence-hours", "12", "--executor-id", "momentum_ic")
    assert code == 0 and added["result"]["namespace"] == "pod_dynamics"
    code, ticked = run("tick")
    assert code == 0 and ticked["online"] is False
    assert ticked["result"][0]["returned"] == 0 and ticked["result"][0]["jobs"] == []
    assert run("list", "--pod", "dynamics")[1]["result"][0]["last_success_at"] == T0.isoformat()
    assert run("list", "--pod", "events")[1]["result"] == []
    assert ResearchQueue(tmp_path / "queue.sqlite").list() == []
