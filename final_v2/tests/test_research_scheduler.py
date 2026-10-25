"""Adversarial tests for the deterministic recurring research allocator (no model, no network, no executors run)."""

import datetime as dt
import sqlite3
import threading

import pytest

from pod_platform.pods import ExperimentPlan, ExperimentVariant, FeatureDefinition, ResearchProposal
from pod_platform.research.jobs import ResearchQueue, open_sqlite
from pod_platform.research.memory import PodMemory
from pod_platform.research.scheduler import (
    PodPolicy, ResearchScheduler, SchedulerError, cohort_due_at, latest_matured_cohort, weekly_cycle)

UTC = dt.timezone.utc
VAL, EVT = "pod_valuation", "pod_events"
EXEC = "builtin_valuation_relative_value_v1"
NOW = dt.datetime(2026, 10, 21, 12, 0, tzinfo=UTC)  # ISO week 43; September cohort matured, October not


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def sc(econ=0.5, evid=0.5, feas=0.5, cost=0.2, red=0.1):
    return {"economic_importance": econ, "evidence": evid, "feasibility": feas, "cost": cost, "redundancy": red}


def cards(tag):
    return {"source_cards": [{"evidence_id": f"src_{tag}", "title": f"Source {tag}", "summary": f"Coverage summary for {tag}.",
                              "available_at": "2026-10-01T00:00:00+00:00"}]}


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    queue = ResearchQueue(tmp_path / "q.sqlite", clock=clock)
    sched = ResearchScheduler(tmp_path / "s.sqlite", queue=queue, clock=clock, memory_root=str(tmp_path / "mem"))
    for ns in (VAL, EVT):
        sched.configure_pod(PodPolicy(ns, (EXEC,)))
    return sched, queue, clock, tmp_path


def draft(tmp_path, ns=VAL, executor=EXEC):
    variant = ExperimentVariant(name="v1", params={"dataset_id": "fixture_panel"})
    proposal = ResearchProposal(
        title="Stored draft proposal", mechanism="A mechanism long enough to pass the schema validation.",
        falsifiable_target="Candidate rank IC is not above the declared baseline on design.", baseline="Simple baseline",
        features=[FeatureDefinition(name="f_one", definition="Rank of a panel column", timing="Known at month end", source_ids=["src_a"])],
        evidence_refs=["src_a"],
        experiment_plan=ExperimentPlan(executor_id=executor, primary_metric="belief_ic_improvement", data_partition="design",
                                       variants=[variant], description="Run the registered built-in executor."),
        competing_explanation="Common style exposure.", limitations=["Fixture only"], evidence_quality="weak")
    return PodMemory(tmp_path / "mem", ns).append("draft_proposal", {"status": "draft", "title": proposal.title, "evaluated": False,
                                                                     "admission": "not_admitted", "proposal": proposal.model_dump(mode="json")})


def test_cycle_helpers_are_calendar_exact():
    assert weekly_cycle(NOW)[0] == "w:2026-W43" and weekly_cycle(NOW)[1] == dt.datetime(2026, 10, 19, tzinfo=UTC)
    assert cohort_due_at("2026-12", 5) == dt.datetime(2027, 1, 6, tzinfo=UTC)
    assert latest_matured_cohort(NOW, 5) == "2026-09"
    assert latest_matured_cohort(dt.datetime(2026, 10, 3, tzinfo=UTC), 5) == "2026-08"  # September matures only on Oct 6


def test_quota_priority_idempotent_ticks_and_restart(env):
    sched, queue, clock, tmp = env
    ids = {}
    for tag, econ in (("a", 0.9), ("b", 0.8), ("c", 0.3), ("d", 0.2)):
        ids[tag] = sched.add_trigger(VAL, "source", f"snap_{tag}", "Does the new snapshot change coverage?", sc(econ=econ), payload=cards(tag))["trigger_id"]
    first = sched.tick()["pods"][VAL]
    assert sorted(r["trigger_id"] for r in first["routed"]) == sorted([ids["a"], ids["b"]])  # highest priority wins the 2 slots
    assert sorted(r["trigger_id"] for r in first["deferred_quota"]) == sorted([ids["c"], ids["d"]])
    jobs = queue.list(namespace=VAL)
    assert len(jobs) == 2 and all(j.kind == "propose" and j.namespace == VAL for j in jobs)
    p = jobs[0].payload  # worker-compatible propose schema
    assert {"question", "research_asof", "mode", "policy", "executors", "paper_cards", "dataset_cards", "source_cards"} <= set(p)
    assert p["executors"] == [EXEC] and p["research_asof"] == NOW.isoformat() and p["mode"] == "contemporary"
    for _ in range(3):  # repeated ticks, then a restarted scheduler, never add jobs or reset the quota
        sched.tick()
    again = ResearchScheduler(tmp / "s.sqlite", queue=queue, clock=clock, memory_root=str(tmp / "mem"))
    again.tick()
    assert len(queue.list(namespace=VAL)) == 2
    assert again.status(VAL)["cycles"]["w:2026-W43"]["used"]["candidate"] == 2
    clock.now = NOW + dt.timedelta(days=7)  # next ISO week: fresh quota for the leftovers only
    nxt = again.tick()["pods"][VAL]
    assert sorted(r["trigger_id"] for r in nxt["routed"]) == sorted([ids["c"], ids["d"]]) and len(queue.list(namespace=VAL)) == 4


def test_lanes_are_reserved_separately_and_review_only_is_honest(env):
    sched, queue, clock, _ = env
    for tag in "abc":
        sched.add_trigger(VAL, "source", f"s_{tag}", "Question about new sources?", sc(), payload=cards(tag))
    cost = sched.add_trigger(VAL, "cost", "cost_schedule_v2", "Did the cost schedule change net results?", sc(), payload=cards("cost"))["trigger_id"]
    calib = sched.add_trigger(VAL, "calibration", "cal_sep", "Is forecast calibration stable?", sc(), cohort_month="2026-09")["trigger_id"]
    late = sched.add_trigger(VAL, "coverage", "cov_oct", "Did coverage drift in October?", sc(), cohort_month="2026-10")["trigger_id"]
    out = sched.tick()["pods"][VAL]
    assert cost in [r["trigger_id"] for r in out["routed"]]  # candidate quota being full does not starve the challenge lane
    assert len(out["deferred_quota"]) == 1
    assert [r["trigger_id"] for r in out["review_only"]] == [calib]
    assert [r["trigger_id"] for r in out["deferred_cohort"]] == [late]  # October cohort not matured before Nov 6
    item = next(i for i in sched.status(VAL)["items"] if i["trigger_id"] == calib)
    assert item["status"] == "review_only" and item["job_id"] is None and item["quota_lane"] == "review" and "review_only" in item["reason"]
    assert item["cycle_id"] == "m:2026-09" and len(queue.list(namespace=VAL)) == 3  # nothing was queued for the review item
    clock.now = dt.datetime(2026, 11, 7, tzinfo=UTC)
    assert [r["trigger_id"] for r in sched.tick()["pods"][VAL]["review_only"]] == [late]


def test_concurrent_ticks_cannot_overspend_or_duplicate(env):
    sched, queue, _, _ = env
    for i in range(8):
        sched.add_trigger(VAL, "source", f"s_{i}", "Concurrent question?", sc(econ=i / 10), payload=cards(str(i)))
    errors = []

    def run():
        try:
            sched.tick()
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert len(queue.list(namespace=VAL)) == 2
    items = sched.status(VAL)["items"]
    assert len(items) == 2 and len({i["trigger_id"] for i in items}) == 2


def test_database_itself_refuses_slot_reuse_and_quota_reset(env):
    sched, _, _, tmp = env
    sched.add_trigger(VAL, "source", "s1", "A question?", sc(), payload=cards("1"))
    sched.tick()
    with open_sqlite(tmp / "s.sqlite") as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM work_items")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE cycles SET quota_json='{}'")
        row = conn.execute("SELECT * FROM work_items").fetchone()
        with pytest.raises(sqlite3.IntegrityError):  # same (ns, cycle, lane, slot) under a different item id
            conn.execute("INSERT INTO work_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         ("si_x", row[1], row[2], "tr_other", row[4], row[5], row[6], "routed", 0.1, "{}", None, None, None, "", row[14]))
    sched.configure_pod(PodPolicy(VAL, (EXEC,), weekly_quota={"candidate": 50, "challenge": 1, "replication": 1, "review": 3}))
    assert sched.status(VAL)["cycles"]["w:2026-W43"]["quota"]["candidate"] == 2  # edits apply to new cycles only


def test_dependencies_block_then_release_or_fail(env):
    sched, queue, clock, _ = env
    a = sched.add_trigger(VAL, "source", "root", "Root question?", sc(), payload=cards("r"))["trigger_id"]
    b = sched.add_trigger(VAL, "source", "child", "Child question?", sc(econ=0.99), payload=cards("c"), depends_on=[a])["trigger_id"]
    with pytest.raises(SchedulerError):
        sched.add_trigger(VAL, "source", "bad", "Bad question?", sc(), depends_on=["tr_missing"])
    out = sched.tick()["pods"][VAL]
    assert [r["trigger_id"] for r in out["routed"]] == [a] and [r["trigger_id"] for r in out["blocked"]] == [b]
    job = queue.claim("w", namespace=VAL)
    queue.complete(job.job_id, job.lease_token, {"ok": True})
    assert [r["trigger_id"] for r in sched.tick()["pods"][VAL]["routed"]] == [b]
    clock.now = NOW + dt.timedelta(days=7)  # this week's candidate quota is spent
    c = sched.add_trigger(VAL, "source", "root2", "Second root?", sc(), payload=cards("r2"))["trigger_id"]
    d = sched.add_trigger(VAL, "source", "child2", "Second child?", sc(), payload=cards("c2"), depends_on=[c])["trigger_id"]
    sched.tick()
    failing = queue.claim("w", namespace=VAL)  # claims the oldest queued job: the child 'b'
    queue.fail(failing.job_id, failing.lease_token, {"message": "x"})
    root2 = queue.claim("w", namespace=VAL)
    queue.fail(root2.job_id, root2.lease_token, {"message": "boom"})
    out = sched.tick()["pods"][VAL]
    assert [r["detail"] for r in out["rejected"]] == [f"dependency_failed:{c}"] and d in [r["trigger_id"] for r in out["rejected"]]


def test_pause_and_failure_stopping_rule(env):
    sched, queue, clock, _ = env
    sched.add_trigger(VAL, "source", "p1", "Paused line question?", sc(), payload=cards("p"), line_id="line_a")
    sched.pause(VAL, "operator pause", line_id="line_a")
    assert [r["trigger_id"] for r in sched.tick()["pods"][VAL]["paused"]] and not queue.list(namespace=VAL)
    sched.add_trigger(VAL, "source", "f1", "First failing?", sc(), payload=cards("f1"), line_id="line_b")
    sched.add_trigger(VAL, "source", "f2", "Second failing?", sc(), payload=cards("f2"), line_id="line_b")
    sched.tick()
    for _ in range(2):
        job = queue.claim("w", namespace=VAL)
        queue.fail(job.job_id, job.lease_token, {"message": "provider down"})
    sched.add_trigger(VAL, "source", "f3", "Third on same line?", sc(), payload=cards("f3"), line_id="line_b")
    clock.now = NOW + dt.timedelta(days=7)
    out = sched.tick()["pods"][VAL]
    assert len(out["paused"]) == 2 and not out["routed"]  # the operator-paused line_a and the auto-paused line_b
    assert "stopping rule" in sched.status(VAL)["pauses"]["line_b"]
    sched.pause(VAL, "whole pod stop")
    sched.add_trigger(VAL, "source", "g1", "Other line?", sc(), payload=cards("g"))
    assert not sched.tick()["pods"][VAL]["routed"]


def test_namespace_privacy_and_no_executor_injection(env):
    sched, queue, _, tmp = env
    with pytest.raises(SchedulerError):  # a trigger cannot smuggle an executor, code or path
        sched.add_trigger(VAL, "source", "evil", "Question?", sc(), payload={**cards("e"), "executors": ["os_system"]})
    with pytest.raises(SchedulerError):
        sched.add_trigger(VAL, "source", "noscores", "Question?", {"evidence": 0.5})  # priorities are never invented
    foreign = draft(tmp, ns=EVT)  # a draft that exists only in the events pod's private memory
    t = sched.add_trigger(VAL, "replication", "rep_foreign", "Replicate?", sc(), payload={"draft_entry_id": foreign.entry_id})["trigger_id"]
    sched.add_trigger(EVT, "source", "evt_only", "Events question?", sc(), payload=cards("evt"))
    out = sched.tick()["pods"]
    assert [r["trigger_id"] for r in out[VAL]["review_only"]] == [t] and not queue.list(namespace=VAL)
    assert len(queue.list(namespace=EVT)) == 1 and sched.status(VAL)["items"][0]["trigger_id"] == t
    assert all("evt_only" not in str(i) for i in sched.status(VAL)["items"])
    with pytest.raises(SchedulerError):
        sched.status("pod_dynamics")  # unconfigured pod
    with pytest.raises(SchedulerError):  # only calibration/coverage triggers carry a cohort month
        sched.add_trigger(VAL, "source", "odd_cohort", "Q?", sc(), cohort_month="2026-09")


def test_replication_routes_the_workers_experiment_job_only_for_a_real_allowed_draft(env):
    sched, queue, _, tmp = env
    good = draft(tmp)
    bad_exec = draft(tmp, executor="builtin_market_dynamics_v1")
    t1 = sched.add_trigger(VAL, "replication", "rep_good", "Re-run registered variants?", sc(), payload={"draft_entry_id": good.entry_id})["trigger_id"]
    t2 = sched.add_trigger(VAL, "replication", "rep_bad", "Re-run foreign executor?", sc(), payload={"draft_entry_id": bad_exec.entry_id})["trigger_id"]
    t3 = sched.add_trigger(VAL, "replication", "rep_none", "Re-run missing?", sc(), payload={"draft_entry_id": "mem_nonexistent"})["trigger_id"]
    t4 = sched.add_trigger(VAL, "replication", "rep_variant", "Unknown variant?", sc(), payload={"draft_entry_id": good.entry_id, "variants": ["zzz"]})["trigger_id"]
    out = sched.tick()["pods"][VAL]
    assert [r["trigger_id"] for r in out["routed"]] == [t1] and {r["trigger_id"] for r in out["review_only"]} == {t2, t3}
    assert [r["trigger_id"] for r in out["rejected"]] == [t4]
    job = queue.claim("w", namespace=VAL, kinds=("propose", "experiment"))
    assert job.kind == "experiment" and job.payload == {"draft_entry_id": good.entry_id, "variants": ["v1"]}


def test_duplicates_redundancy_and_invalid_evidence(env):
    sched, queue, _, _ = env
    first = sched.add_trigger(VAL, "source", "dup", "Same?", sc(), payload=cards("d"))
    assert sched.add_trigger(VAL, "source", "dup", "Same?", sc(), payload=cards("d")) == {**first, "duplicate": True}
    with pytest.raises(SchedulerError):
        sched.add_trigger(VAL, "source", "dup", "Same?", sc(econ=0.9), payload=cards("d"))
    a = sched.add_trigger(VAL, "source", "ra", "Mechanism X?", sc(econ=0.9), payload=cards("ra"), redundancy_key="mech_x")["trigger_id"]
    b = sched.add_trigger(VAL, "paper", "rb", "Mechanism X again?", sc(econ=0.1), payload=cards("rb"), redundancy_key="mech_x")["trigger_id"]
    future = {"source_cards": [{"evidence_id": "src_f", "title": "Future", "summary": "Not yet public.", "available_at": "2027-01-01T00:00:00+00:00"}]}
    f = sched.add_trigger(VAL, "source", "fut", "Future evidence?", sc(econ=0.05), payload=future)["trigger_id"]
    out = sched.tick()["pods"][VAL]
    rejected = {r["trigger_id"]: r["detail"] for r in out["rejected"]}
    assert rejected[b].startswith("duplicate_of:") and rejected[f].startswith("invalid_evidence")
    assert a in [r["trigger_id"] for r in out["routed"]]


def test_review_resolution_unblocks_dependents(env):
    sched, _, _, _ = env
    r = sched.add_trigger(VAL, "cost", "rev", "Manual cost review?", sc())["trigger_id"]
    sched.add_trigger(VAL, "source", "after", "After the review?", sc(), payload=cards("z"), depends_on=[r])
    out = sched.tick()["pods"][VAL]
    assert len(out["review_only"]) == 1 and len(out["blocked"]) == 1
    item_id = out["review_only"][0]["detail"]
    with pytest.raises(SchedulerError):
        sched.resolve_review(EVT, item_id, "completed", "wrong namespace")
    sched.resolve_review(VAL, item_id, "completed", "reviewed by a human")
    assert len(sched.tick()["pods"][VAL]["routed"]) == 1


def test_new_midweek_paper_uses_persisted_trigger_clock(env):
    sched,queue,clock,tmp=env
    payload=cards('midweek')
    field=next(k for k in ('paper_cards','dataset_cards','source_cards') if payload.get(k))
    # Evidence became available after the Monday budget boundary but before discovery.
    payload[field][0]['available_at']=(NOW-dt.timedelta(hours=1)).isoformat()
    sched.add_trigger(VAL,'paper','fresh-midweek-paper','Compare the newly released method',sc(),payload=payload)
    result=sched.tick()['pods'][VAL]
    assert len(result['routed'])==1
    job=queue.list(namespace=VAL)[0]
    assert job.payload['research_asof']==NOW.isoformat()
    clock.now=NOW+dt.timedelta(hours=2)
    sched.tick()
    assert len(queue.list(namespace=VAL))==1
    assert queue.list(namespace=VAL)[0].payload['research_asof']==NOW.isoformat()
