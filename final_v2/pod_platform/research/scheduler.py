"""Deterministic recurring research allocator (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 5.5, 10.2, 12, 23.4).

Explicit ``tick`` only; no daemon, no model call, no executor of its own. A tick opens the current ISO-week cycle
and the matured-cohort monthly cycle of every configured pod, ranks pending triggers (source / paper / calibration /
coverage / cost / replication) by a deterministic categorised priority, and routes at most the cycle's per-lane quota
into the existing ``ResearchQueue`` using the worker's own ``propose`` / ``experiment`` payload schemas. Anything the
worker cannot execute is recorded as ``review_only`` (it consumes the separate ``review`` quota and never claims a result).

Safety: one ``BEGIN IMMEDIATE`` transaction per tick (concurrent ticks serialise), a unique ``(ns, cycle, quota_lane,
slot)`` key (quotas cannot be overspent even by a buggy caller), one work item per trigger (never duplicated), job
idempotency key ``sched:<item_id>`` with a payload fixed by cycle (a crashed tick re-submits the identical job), quota
snapshot per cycle and delete-blocking triggers (a restart or policy edit cannot reset spent quota). Every query is
namespace-filtered; evidence is only ever copied into the same namespace's jobs.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from ..agents.provider import ResearchMode
from ..contracts.base import canonical_digest, require_aware
from ..pods.charters import pod_spec
from ..pods.schema import ResearchProposal, filter_eligible, normalise_evidence
from .jobs import NAMESPACE_RE, ResearchQueue, open_sqlite, utc_now
from .memory import PodMemory

LANES = ("candidate", "challenge", "replication")
QUOTA_LANES = LANES + ("review",)
KIND_LANE = {"source": "candidate", "paper": "candidate", "calibration": "challenge", "coverage": "challenge",
             "cost": "challenge", "replication": "replication"}
COHORT_KINDS = ("calibration", "coverage")  # routed only in a matured monthly cohort cycle
SCORE_KEYS = ("economic_importance", "evidence", "feasibility", "cost", "redundancy")
DEFAULT_WEIGHTS = {"economic_importance": 0.35, "evidence": 0.25, "feasibility": 0.20, "cost": 0.10, "redundancy": 0.10}
CARD_FIELDS = (("paper_cards", "paper_card"), ("dataset_cards", "dataset_card"), ("source_cards", "source_card"))
PAYLOAD_KEYS = {"paper_cards", "dataset_cards", "source_cards", "draft_entry_id", "variants"}
_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_DONE = ("succeeded", "review_done")
_DEAD = ("failed", "review_abandoned")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pods (namespace TEXT PRIMARY KEY, policy_json TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS triggers (
    trigger_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, kind TEXT NOT NULL, subject TEXT NOT NULL,
    spec_json TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL DEFAULT '', UNIQUE (namespace, kind, subject));
CREATE TABLE IF NOT EXISTS cycles (
    namespace TEXT NOT NULL, cycle_id TEXT NOT NULL, cadence TEXT NOT NULL, starts_at TEXT NOT NULL,
    quota_json TEXT NOT NULL, opened_at TEXT NOT NULL, PRIMARY KEY (namespace, cycle_id));
CREATE TABLE IF NOT EXISTS work_items (
    item_id TEXT PRIMARY KEY, namespace TEXT NOT NULL, cycle_id TEXT NOT NULL, trigger_id TEXT NOT NULL,
    lane TEXT NOT NULL, quota_lane TEXT NOT NULL, slot INTEGER NOT NULL, status TEXT NOT NULL,
    priority REAL NOT NULL, priority_json TEXT NOT NULL, redundancy_key TEXT, job_id TEXT, job_kind TEXT,
    reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
    UNIQUE (namespace, cycle_id, quota_lane, slot), UNIQUE (namespace, trigger_id));
CREATE TABLE IF NOT EXISTS pauses (
    namespace TEXT NOT NULL, scope TEXT NOT NULL, reason TEXT NOT NULL, at TEXT NOT NULL, PRIMARY KEY (namespace, scope));
CREATE TRIGGER IF NOT EXISTS work_items_no_delete BEFORE DELETE ON work_items
BEGIN SELECT RAISE(ABORT, 'work items are never deleted (spent quota cannot be reset)'); END;
CREATE TRIGGER IF NOT EXISTS cycles_no_delete BEFORE DELETE ON cycles
BEGIN SELECT RAISE(ABORT, 'cycles are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS cycles_no_update BEFORE UPDATE ON cycles
BEGIN SELECT RAISE(ABORT, 'a cycle quota snapshot is immutable'); END;
CREATE TRIGGER IF NOT EXISTS pauses_no_delete BEFORE DELETE ON pauses
BEGIN SELECT RAISE(ABORT, 'pauses are never deleted'); END;
"""


class SchedulerError(ValueError):
    pass


class _Reject(Exception):
    pass


def _iso(value: dt.datetime) -> str:
    return require_aware(value, "timestamp").astimezone(dt.timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PodPolicy:
    namespace: str
    allowed_executors: tuple[str, ...]
    weekly_quota: Mapping[str, int] = field(default_factory=lambda: {"candidate": 2, "challenge": 1, "replication": 1, "review": 3})
    monthly_quota: Mapping[str, int] = field(default_factory=lambda: {"candidate": 0, "challenge": 2, "replication": 1, "review": 3})
    cohort_maturity_days: int = 5
    max_failures_per_line: int = 2
    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def validated(self) -> "PodPolicy":
        pod_spec(self.namespace)
        if not NAMESPACE_RE.match(self.namespace):
            raise SchedulerError("invalid namespace")
        if not self.allowed_executors or any(not _IDENT.match(e) for e in self.allowed_executors):
            raise SchedulerError("allowed_executors must be non-empty registered executor ids")
        for name, quota in (("weekly_quota", self.weekly_quota), ("monthly_quota", self.monthly_quota)):
            if set(quota) != set(QUOTA_LANES) or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 50 for v in quota.values()):
                raise SchedulerError(f"{name} needs integer 0..50 for exactly {QUOTA_LANES}")
        if set(self.weights) != set(SCORE_KEYS) or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise SchedulerError(f"weights need finite non-negative values for exactly {SCORE_KEYS}")
        if not 0 <= self.cohort_maturity_days <= 60 or not 1 <= self.max_failures_per_line <= 10:
            raise SchedulerError("cohort_maturity_days must be 0..60 and max_failures_per_line 1..10")
        return PodPolicy(self.namespace, tuple(sorted(set(self.allowed_executors))), dict(self.weekly_quota),
                         dict(self.monthly_quota), self.cohort_maturity_days, self.max_failures_per_line, dict(self.weights))

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "PodPolicy":
        return PodPolicy(**{**data, "allowed_executors": tuple(data["allowed_executors"])})


def weekly_cycle(now: dt.datetime) -> tuple[str, dt.datetime]:
    year, week, _ = require_aware(now, "now").astimezone(dt.timezone.utc).isocalendar()
    start = dt.datetime.combine(dt.date.fromisocalendar(year, week, 1), dt.time.min, tzinfo=dt.timezone.utc)
    return f"w:{year}-W{week:02d}", start


def cohort_due_at(cohort_month: str, maturity_days: int) -> dt.datetime:
    """A month's cohort is matured ``maturity_days`` after that month ended (UTC midnight)."""
    year, month = int(cohort_month[:4]), int(cohort_month[5:])
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return dt.datetime.combine(nxt, dt.time.min, tzinfo=dt.timezone.utc) + dt.timedelta(days=maturity_days)


def latest_matured_cohort(now: dt.datetime, maturity_days: int) -> str:
    now = require_aware(now, "now").astimezone(dt.timezone.utc)
    year, month = now.year, now.month
    for _ in range(4):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        label = f"{year:04d}-{month:02d}"
        if cohort_due_at(label, maturity_days) <= now:
            return label
    raise SchedulerError("no matured cohort")  # unreachable for maturity_days <= 60


class ResearchScheduler:
    def __init__(self, path, *, queue: ResearchQueue, clock: Callable[[], dt.datetime] = utc_now,
                 memory_root: "str | None" = None):
        self.path, self.queue, self._clock, self.memory_root = path, queue, clock, memory_root
        with open_sqlite(path) as conn:
            conn.executescript(_SCHEMA)

    def _now(self) -> dt.datetime:
        return require_aware(self._clock(), "clock")

    # -- configuration -----------------------------------------------------

    def configure_pod(self, policy: PodPolicy) -> PodPolicy:
        """Store the pod's policy. Cycles already opened keep their quota snapshot."""
        policy = policy.validated()
        with open_sqlite(self.path, immediate=True) as conn:
            conn.execute("INSERT INTO pods VALUES (?,?,?) ON CONFLICT(namespace) DO UPDATE SET policy_json=excluded.policy_json, "
                         "updated_at=excluded.updated_at", (policy.namespace, _json(asdict(policy)), _iso(self._now())))
        return policy

    def policy(self, namespace: str, conn=None) -> PodPolicy:
        def read(c):
            row = c.execute("SELECT policy_json FROM pods WHERE namespace=?", (namespace,)).fetchone()
            if row is None:
                raise SchedulerError(f"pod {namespace!r} is not configured")
            return PodPolicy.from_dict(json.loads(row[0]))
        if conn is not None:
            return read(conn)
        with open_sqlite(self.path) as c:
            return read(c)

    def pause(self, namespace: str, reason: str, line_id: Optional[str] = None) -> None:
        """Stopping rule: pause a whole pod (``line_id=None``) or one research line. Never un-paused here."""
        if not reason.strip() or (line_id is not None and not _IDENT.match(line_id)):
            raise SchedulerError("a reason (and an identifier line_id) is required")
        with open_sqlite(self.path, immediate=True) as conn:
            self.policy(namespace, conn)
            conn.execute("INSERT OR IGNORE INTO pauses VALUES (?,?,?,?)", (namespace, line_id or "*", reason[:300], _iso(self._now())))

    # -- triggers ----------------------------------------------------------

    def add_trigger(self, namespace: str, kind: str, subject: str, question: str, scores: Mapping[str, float], *,
                    payload: Optional[Mapping[str, Any]] = None, line_id: Optional[str] = None,
                    depends_on: Sequence[str] = (), cohort_month: Optional[str] = None,
                    redundancy_key: Optional[str] = None) -> dict:
        """Record a trigger once per (namespace, kind, subject). Same content is a no-op; different content raises."""
        payload = dict(payload or {})
        subject, question = (subject or "").strip(), (question or "").strip()
        if kind not in KIND_LANE or not 1 <= len(subject) <= 200 or not 1 <= len(question) <= 2000:
            raise SchedulerError(f"kind must be one of {sorted(KIND_LANE)}; subject 1-200 and question 1-2000 characters")
        if set(scores) != set(SCORE_KEYS) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in scores.values()):
            raise SchedulerError(f"scores must supply every category {SCORE_KEYS} in [0, 1]; none is invented")
        if set(payload) - PAYLOAD_KEYS:
            raise SchedulerError(f"payload may only contain {sorted(PAYLOAD_KEYS)} (no executors, code or paths)")
        if line_id is not None and not _IDENT.match(line_id):
            raise SchedulerError("line_id must be an identifier")
        if (kind in COHORT_KINDS) != (cohort_month is not None) or (cohort_month is not None and not _MONTH.match(cohort_month)):
            raise SchedulerError("calibration/coverage triggers (and only those) need cohort_month YYYY-MM")
        spec = {"question": question, "scores": {k: float(scores[k]) for k in SCORE_KEYS}, "payload": payload,
                "line_id": line_id, "depends_on": sorted(set(depends_on)), "cohort_month": cohort_month,
                "redundancy_key": redundancy_key}
        trigger_id = "tr_" + canonical_digest({"ns": namespace, "kind": kind, "subject": subject})[:20]
        try:
            blob = _json(spec)
        except ValueError as exc:
            raise SchedulerError(f"trigger is not finite JSON: {exc}") from None
        with open_sqlite(self.path, immediate=True) as conn:
            self.policy(namespace, conn)
            for dep in spec["depends_on"]:
                if dep == trigger_id or conn.execute("SELECT 1 FROM triggers WHERE trigger_id=? AND namespace=?", (dep, namespace)).fetchone() is None:
                    raise SchedulerError(f"dependency {dep!r} is not an earlier trigger of this namespace")
            row = conn.execute("SELECT spec_json, status FROM triggers WHERE trigger_id=?", (trigger_id,)).fetchone()
            if row is not None:
                if row[0] != blob:
                    raise SchedulerError(f"trigger {trigger_id} already exists with different content")
                return {"trigger_id": trigger_id, "status": row[1], "duplicate": True}
            conn.execute("INSERT INTO triggers (trigger_id, namespace, kind, subject, spec_json, created_at) VALUES (?,?,?,?,?,?)",
                         (trigger_id, namespace, kind, subject, blob, _iso(self._now())))
        return {"trigger_id": trigger_id, "status": "pending", "duplicate": False}

    # -- tick --------------------------------------------------------------

    def tick(self) -> dict:
        now = self._now()
        report: dict[str, Any] = {"at": _iso(now), "pods": {}}
        with open_sqlite(self.path, immediate=True) as conn:
            for (namespace,) in conn.execute("SELECT namespace FROM pods ORDER BY namespace").fetchall():
                report["pods"][namespace] = self._tick_pod(conn, self.policy(namespace, conn), now)
        return report

    def _open_cycle(self, conn, pol: PodPolicy, cycle_id: str, cadence: str, starts: dt.datetime, now: dt.datetime) -> dict:
        row = conn.execute("SELECT quota_json FROM cycles WHERE namespace=? AND cycle_id=?", (pol.namespace, cycle_id)).fetchone()
        if row is None:
            quota = dict(pol.weekly_quota if cadence == "weekly" else pol.monthly_quota)
            conn.execute("INSERT INTO cycles VALUES (?,?,?,?,?,?)", (pol.namespace, cycle_id, cadence, _iso(starts), _json(quota), _iso(now)))
            return quota
        return json.loads(row[0])

    def _tick_pod(self, conn, pol: PodPolicy, now: dt.datetime) -> dict:
        ns = pol.namespace
        out: dict[str, Any] = {k: [] for k in ("routed", "review_only", "rejected", "blocked", "deferred_quota", "deferred_cohort", "paused")}
        self._sync(conn, pol, now)
        wk_id, wk_start = weekly_cycle(now)
        self._open_cycle(conn, pol, wk_id, "weekly", wk_start, now)
        mo = latest_matured_cohort(now, pol.cohort_maturity_days)
        self._open_cycle(conn, pol, f"m:{mo}", "monthly", cohort_due_at(mo, pol.cohort_maturity_days), now)
        out["cycles"] = [wk_id, f"m:{mo}"]
        rows = conn.execute("SELECT trigger_id, kind, spec_json FROM triggers WHERE namespace=? AND status='pending'", (ns,)).fetchall()
        ranked = []
        for tid, kind, blob in rows:
            spec = json.loads(blob)
            ranked.append((-self._priority(pol, spec["scores"]), tid, kind, spec))
        for neg, tid, kind, spec in sorted(ranked, key=lambda r: (r[0], r[1])):
            outcome = self._schedule(conn, pol, tid, kind, spec, -neg, now, wk_id, wk_start)
            out[outcome[0]].append({"trigger_id": tid, "detail": outcome[1]})
        return out

    @staticmethod
    def _priority(pol: PodPolicy, scores: Mapping[str, float]) -> float:
        w = pol.weights
        return round(w["economic_importance"] * scores["economic_importance"] + w["evidence"] * scores["evidence"]
                     + w["feasibility"] * scores["feasibility"] - w["cost"] * scores["cost"]
                     - w["redundancy"] * scores["redundancy"], 9)

    def _sync(self, conn, pol: PodPolicy, now: dt.datetime) -> None:
        """Mirror terminal job states into work items, then apply the line stopping rule."""
        ns = pol.namespace
        for item_id, job_id in conn.execute("SELECT item_id, job_id FROM work_items WHERE namespace=? AND status='routed'", (ns,)).fetchall():
            job = self.queue.get(job_id)
            if job is not None and job.terminal:
                reason = "" if job.state == "succeeded" else str((job.error or {}).get("message", "job failed"))[:300]
                conn.execute("UPDATE work_items SET status=?, reason=? WHERE item_id=?", (job.state, reason, item_id))
        failures: dict[str, int] = {}
        for (blob,) in conn.execute("SELECT t.spec_json FROM work_items w JOIN triggers t ON t.trigger_id=w.trigger_id "
                                    "WHERE w.namespace=? AND w.status='failed'", (ns,)).fetchall():
            line = json.loads(blob)["line_id"]
            if line:
                failures[line] = failures.get(line, 0) + 1
        for line, count in failures.items():
            if count >= pol.max_failures_per_line:
                conn.execute("INSERT OR IGNORE INTO pauses VALUES (?,?,?,?)",
                             (ns, line, f"stopping rule: {count} failed jobs on this line", _iso(now)))

    def _schedule(self, conn, pol, tid, kind, spec, priority, now, wk_id, wk_start) -> tuple[str, str]:
        ns, lane = pol.namespace, KIND_LANE[kind]
        paused = {r[0] for r in conn.execute("SELECT scope FROM pauses WHERE namespace=?", (ns,))}
        if "*" in paused or (spec["line_id"] and spec["line_id"] in paused):
            return "paused", "pod or research line is paused"
        for dep in spec["depends_on"]:
            drow = conn.execute("SELECT status, reason FROM triggers WHERE trigger_id=? AND namespace=?", (dep, ns)).fetchone()
            item = conn.execute("SELECT status FROM work_items WHERE trigger_id=? AND namespace=?", (dep, ns)).fetchone()
            if (drow and drow[0] == "rejected") or (item and item[0] in _DEAD):
                return self._reject(conn, tid, f"dependency_failed:{dep}")
            if item is None or item[0] not in _DONE:
                return "blocked", f"waiting for {dep}"
        if kind in COHORT_KINDS:
            cohort = spec["cohort_month"]
            due = cohort_due_at(cohort, pol.cohort_maturity_days)
            if due > now:
                return "deferred_cohort", f"cohort {cohort} matures at {_iso(due)}"
            cycle_id, starts = f"m:{cohort}", due
            self._open_cycle(conn, pol, cycle_id, "monthly", starts, now)
        else:
            cycle_id, starts = wk_id, wk_start
        key = spec["redundancy_key"]
        if key:
            twin = conn.execute("SELECT w.item_id FROM work_items w JOIN triggers t ON t.trigger_id=w.trigger_id WHERE w.namespace=? "
                                "AND w.status NOT IN ('failed','review_abandoned') AND json_extract(t.spec_json,'$.redundancy_key')=?",
                                (ns, key)).fetchone()
            if twin:
                return self._reject(conn, tid, f"duplicate_of:{twin[0]}")
        try:
            # Freeze a stable research clock from the persisted trigger and
            # cycle boundary. Do not backdate new mid-week evidence to Monday.
            created_at = dt.datetime.fromisoformat(conn.execute('SELECT created_at FROM triggers WHERE trigger_id=?', (tid,)).fetchone()[0])
            work_asof = max(starts, created_at)
            job_kind, payload, reason = self._plan(pol, kind, spec, work_asof, now)
        except _Reject as exc:
            return self._reject(conn, tid, str(exc))
        quota_lane = lane if job_kind else "review"
        quota = json.loads(conn.execute("SELECT quota_json FROM cycles WHERE namespace=? AND cycle_id=?", (ns, cycle_id)).fetchone()[0])
        used = conn.execute("SELECT COUNT(*) FROM work_items WHERE namespace=? AND cycle_id=? AND quota_lane=?", (ns, cycle_id, quota_lane)).fetchone()[0]
        if used >= quota[quota_lane]:
            return "deferred_quota", f"{quota_lane} quota {quota[quota_lane]} of {cycle_id} is spent"
        item_id = "si_" + canonical_digest({"ns": ns, "trigger": tid})[:20]
        job_id = None
        if job_kind:  # submit under the lock; the key makes a crash-and-retry resubmit the identical job
            job_id = self.queue.submit(ns, job_kind, payload, idempotency_key=f"sched:{item_id}").job_id
        status = "routed" if job_kind else "review_only"
        conn.execute("INSERT INTO work_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (item_id, ns, cycle_id, tid, lane, quota_lane, used, status, priority, _json(spec["scores"]), key, job_id,
                      job_kind, reason, _iso(now)))
        conn.execute("UPDATE triggers SET status='scheduled' WHERE trigger_id=?", (tid,))
        return status, item_id

    @staticmethod
    def _reject(conn, tid: str, reason: str) -> tuple[str, str]:
        conn.execute("UPDATE triggers SET status='rejected', reason=? WHERE trigger_id=?", (reason[:300], tid))
        return "rejected", reason

    def _plan(self, pol: PodPolicy, kind: str, spec: dict, starts: dt.datetime, now: dt.datetime):
        """Return (job_kind, payload, review_reason). job_kind None means honest review-only."""
        payload = spec["payload"]
        if kind == "replication":
            draft_id = payload.get("draft_entry_id")
            if not draft_id or self.memory_root is None:
                return None, None, "review_only: replication needs a draft_entry_id in this pod's memory (none available)"
            draft = PodMemory(self.memory_root, pol.namespace).get(str(draft_id))
            if draft is None or draft.kind != "draft_proposal":
                return None, None, "review_only: draft not found in this pod's private memory"
            try:
                plan = ResearchProposal.model_validate(draft.content["proposal"]).experiment_plan
            except Exception:  # unreadable stored draft: review, never guess
                return None, None, "review_only: stored draft is not a valid proposal"
            if plan.executor_id not in pol.allowed_executors:
                return None, None, f"review_only: executor {plan.executor_id!r} is not allowed for this pod"
            names = [v.name for v in plan.variants]
            wanted = payload.get("variants") or names
            if not set(wanted) <= set(names) or len(wanted) > self.queue.max_variants:
                raise _Reject("variants are not part of the stored plan")
            return "experiment", {"draft_entry_id": str(draft_id), "variants": sorted(set(wanted))}, ""
        items = []
        try:
            for name, card_kind in CARD_FIELDS:
                items.extend(normalise_evidence(payload.get(name) or [], card_kind))
            if items:
                filter_eligible(items, mode=ResearchMode.CONTEMPORARY, research_asof=starts, now=now)
        except ValueError as exc:
            raise _Reject(f"invalid_evidence: {exc}"[:300]) from None
        if not items:
            return None, None, f"review_only: no executor performs a {kind} check without evidence cards"
        body = {"question": spec["question"], "research_asof": _iso(starts), "mode": "contemporary", "policy": "release_gated",
                "checkpoint_id": None, "executors": list(pol.allowed_executors), "prior_work": [],
                **{name: payload.get(name) or [] for name, _ in CARD_FIELDS}}
        return "propose", body, ""

    # -- reads -------------------------------------------------------------

    def resolve_review(self, namespace: str, item_id: str, outcome: str, note: str) -> None:
        """A human closes a review-only item (``completed`` satisfies dependents, ``abandoned`` fails them)."""
        if outcome not in ("completed", "abandoned") or not note.strip():
            raise SchedulerError("outcome must be completed|abandoned and a note is required")
        with open_sqlite(self.path, immediate=True) as conn:
            cur = conn.execute("UPDATE work_items SET status=?, reason=? WHERE item_id=? AND namespace=? AND status='review_only'",
                               (f"review_{'done' if outcome == 'completed' else 'abandoned'}", note[:300], item_id, namespace))
            if cur.rowcount != 1:
                raise SchedulerError("no review-only item with that id in this namespace")

    def status(self, namespace: str) -> dict:
        with open_sqlite(self.path) as conn:
            self.policy(namespace, conn)
            cycles = {}
            for cid, cadence, quota in conn.execute("SELECT cycle_id, cadence, quota_json FROM cycles WHERE namespace=? ORDER BY cycle_id", (namespace,)):
                used = dict(conn.execute("SELECT quota_lane, COUNT(*) FROM work_items WHERE namespace=? AND cycle_id=? GROUP BY quota_lane", (namespace, cid)))
                cycles[cid] = {"cadence": cadence, "quota": json.loads(quota), "used": {k: used.get(k, 0) for k in QUOTA_LANES}}
            items = [dict(zip(("item_id", "cycle_id", "trigger_id", "lane", "quota_lane", "status", "priority", "job_id", "job_kind", "reason"), r))
                     for r in conn.execute("SELECT item_id, cycle_id, trigger_id, lane, quota_lane, status, priority, job_id, job_kind, reason "
                                           "FROM work_items WHERE namespace=? ORDER BY created_at, item_id", (namespace,))]
            triggers = dict(conn.execute("SELECT status, COUNT(*) FROM triggers WHERE namespace=? GROUP BY status", (namespace,)))
            pauses = {s: r for s, r in conn.execute("SELECT scope, reason FROM pauses WHERE namespace=?", (namespace,))}
        return {"namespace": namespace, "cycles": cycles, "items": items, "triggers": triggers, "pauses": pauses}
