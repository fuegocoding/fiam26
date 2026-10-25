"""Persistent, concurrency-safe LLM call/token budget
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 12.4, 20.5, 20.9).

``PersistentCallBudget`` is a drop-in subclass of ``CallBudget`` whose state is
an append-only, hash-chained event log in SQLite:

* a *named* account (``budget_id`` x ``pod`` x ``cycle``) survives process
  restarts, so shared queue runners cannot evade the cap by restarting;
* ``reserve`` runs inside ``BEGIN IMMEDIATE`` so two connections (threads or
  processes) can never both pass the same remaining headroom;
* the reservation is charged *before* the provider call and stays charged if
  the call fails, is interrupted or never settles (conservative);
* ``settle`` is idempotent per reservation: a second settlement never refunds or
  charges again (a conflicting second report is audited, not applied);
* unknown usage (``actual is None``, e.g. the Claude CLI) is recorded as
  unknown - the reservation simply stays; no precision is invented;
* usage above the reservation is charged as overspend and every later
  ``reserve`` is refused until an operator opens a new account;
* limits can only be lowered by re-opening an account; raising them needs the
  explicit ``raise_limits=True`` and leaves an audit event. Counters are always
  derived from the events, so re-opening can never reset charges.

Tamper evidence: events are immutable (SQL triggers) and chained by hash. This
detects partial edits; a party that can rewrite the whole file *and* recompute
every hash is outside the threat model (no external anchor is configured).

Nothing here calls a model.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from ..contracts.base import require_aware
from .provider import MAX_REPORTED_TOKENS, BudgetExhausted, CallBudget, utc_now

MAX_CALLS_BOUND = 1_000_000
MAX_TOKENS_BOUND = 10**12
GENESIS_HASH = "0" * 64
DEFAULT_BUDGET_ID = "pod-research-default"
DEFAULT_CYCLE = "default"
DEFAULT_POD = "*"  # one account shared by every pod
_IDENT = re.compile(r"^[A-Za-z0-9*][A-Za-z0-9_.:\-]{0,127}$")

_LIMIT_KINDS = ("account_created", "limit_lowered", "limit_raised")


class BudgetError(ValueError):
    """Invalid budget configuration or usage report."""


class BudgetConfigError(BudgetError):
    """Re-opening an account with limits that would loosen the cap."""


class BudgetIntegrityError(RuntimeError):
    """The event log fails its hash chain or invariants."""


class Reservation(int):
    """The reserved token count (an ``int``, for ``CallBudget`` compatibility)
    that also carries the reservation id needed for idempotent settlement."""

    reservation_id: str
    account_key: str

    def __new__(
        cls, tokens: int, reservation_id: str, account_key: str
    ) -> "Reservation":
        obj = super().__new__(cls, tokens)
        obj.reservation_id = reservation_id
        obj.account_key = account_key
        return obj


_SCHEMA = """
CREATE TABLE IF NOT EXISTS budget_accounts (
    account_key TEXT PRIMARY KEY,
    budget_id TEXT NOT NULL,
    pod TEXT NOT NULL,
    cycle TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS budget_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    account_key TEXT NOT NULL REFERENCES budget_accounts(account_key),
    account_seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    reservation_id TEXT,
    calls_delta INTEGER NOT NULL,
    tokens_delta INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    UNIQUE (account_key, account_seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS budget_one_reserve
    ON budget_events (account_key, reservation_id) WHERE kind = 'reserve';
CREATE UNIQUE INDEX IF NOT EXISTS budget_one_final
    ON budget_events (account_key, reservation_id) WHERE kind IN ('settle', 'settle_unknown');
CREATE TRIGGER IF NOT EXISTS budget_events_no_update BEFORE UPDATE ON budget_events
BEGIN SELECT RAISE(ABORT, 'budget events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS budget_events_no_delete BEFORE DELETE ON budget_events
BEGIN SELECT RAISE(ABORT, 'budget events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS budget_accounts_no_update BEFORE UPDATE ON budget_accounts
BEGIN SELECT RAISE(ABORT, 'budget accounts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS budget_accounts_no_delete BEFORE DELETE ON budget_accounts
BEGIN SELECT RAISE(ABORT, 'budget accounts are immutable'); END;
"""


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strict_int(value: Any, name: str, *, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BudgetError(f"{name} must be an integer")
    if value < 0 or value > upper:
        raise BudgetError(f"{name} must lie in [0, {upper}]")
    return int(value)


def _ident(value: str, name: str) -> str:
    if not isinstance(value, str) or not _IDENT.match(value):
        raise BudgetError(f"{name} must match {_IDENT.pattern}")
    return value


def _event_hash(
    account_key: str,
    account_seq: int,
    kind: str,
    reservation_id: Optional[str],
    calls_delta: int,
    tokens_delta: int,
    payload_json: str,
    created_at: str,
    prev_hash: str,
) -> str:
    return _sha(
        _canonical(
            {
                "account_key": account_key,
                "account_seq": account_seq,
                "kind": kind,
                "reservation_id": reservation_id,
                "calls_delta": calls_delta,
                "tokens_delta": tokens_delta,
                "payload": payload_json,
                "created_at": created_at,
                "prev_hash": prev_hash,
            }
        )
    )


class PersistentCallBudget(CallBudget):
    """SQLite-backed ``CallBudget``. See the module docstring for semantics."""

    def __init__(
        self,
        path: "str | Path",
        *,
        budget_id: str,
        max_calls: int,
        max_tokens: int,
        pod: str = DEFAULT_POD,
        cycle: str = DEFAULT_CYCLE,
        raise_limits: bool = False,
        clock: Callable[[], dt.datetime] = utc_now,
        verify_on_open: bool = True,
    ):
        # CallBudget.__init__ is intentionally not called: no state lives in this object.
        self.path = Path(path)
        self.budget_id = _ident(budget_id, "budget_id")
        self.pod = _ident(pod, "pod")
        self.cycle = _ident(cycle, "cycle")
        self.account_key = f"{self.budget_id}|{self.pod}|{self.cycle}"
        self._clock = clock
        want_calls = _strict_int(max_calls, "max_calls", upper=MAX_CALLS_BOUND)
        want_tokens = _strict_int(
            max_tokens, "max_tokens", upper=MAX_TOKENS_BOUND
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()
        self._open_account(want_calls, want_tokens, raise_limits)
        if verify_on_open:
            self.verify()

    # ------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _now(self) -> str:
        return require_aware(self._clock(), "budget clock").isoformat()

    def _append(
        self,
        conn: sqlite3.Connection,
        kind: str,
        *,
        reservation_id: Optional[str] = None,
        calls_delta: int = 0,
        tokens_delta: int = 0,
        payload: Optional[dict[str, Any]] = None,
    ) -> int:
        """Append one event (caller holds the write transaction)."""
        last = conn.execute(
            "SELECT account_seq, event_hash FROM budget_events WHERE account_key = ? "
            "ORDER BY account_seq DESC LIMIT 1",
            (self.account_key,),
        ).fetchone()
        seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS_HASH)
        payload_json = _canonical(payload or {})
        created = self._now()
        digest = _event_hash(
            self.account_key,
            seq,
            kind,
            reservation_id,
            calls_delta,
            tokens_delta,
            payload_json,
            created,
            prev,
        )
        conn.execute(
            "INSERT INTO budget_events (account_key, account_seq, kind, reservation_id, calls_delta, "
            "tokens_delta, payload_json, created_at, prev_hash, event_hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                self.account_key,
                seq,
                kind,
                reservation_id,
                calls_delta,
                tokens_delta,
                payload_json,
                created,
                prev,
                digest,
            ),
        )
        return seq

    def _begin(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except BaseException:
            conn.close()
            raise
        return conn

    def _limits(self, conn: sqlite3.Connection) -> tuple[int, int]:
        row = conn.execute(
            f"SELECT payload_json FROM budget_events WHERE account_key = ? AND kind IN "
            f"({','.join('?' * len(_LIMIT_KINDS))}) ORDER BY account_seq DESC LIMIT 1",
            (self.account_key, *_LIMIT_KINDS),
        ).fetchone()
        if row is None:
            raise BudgetIntegrityError("budget account has no limit event")
        doc = json.loads(row[0])
        return int(doc["max_calls"]), int(doc["max_tokens"])

    def _used(self, conn: sqlite3.Connection) -> tuple[int, int]:
        calls, tokens = conn.execute(
            "SELECT COALESCE(SUM(calls_delta), 0), COALESCE(SUM(tokens_delta), 0) "
            "FROM budget_events WHERE account_key = ?",
            (self.account_key,),
        ).fetchone()
        return int(calls), int(tokens)

    def _open_account(
        self, want_calls: int, want_tokens: int, raise_limits: bool
    ) -> None:
        conn = self._begin()
        try:
            exists = conn.execute(
                "SELECT 1 FROM budget_accounts WHERE account_key = ?",
                (self.account_key,),
            ).fetchone()
            if not exists:
                conn.execute(
                    "INSERT INTO budget_accounts VALUES (?,?,?,?,?)",
                    (
                        self.account_key,
                        self.budget_id,
                        self.pod,
                        self.cycle,
                        self._now(),
                    ),
                )
                self._append(
                    conn,
                    "account_created",
                    payload={
                        "max_calls": want_calls,
                        "max_tokens": want_tokens,
                    },
                )
            else:
                cur_calls, cur_tokens = self._limits(conn)
                raising = want_calls > cur_calls or want_tokens > cur_tokens
                if raising and not raise_limits:
                    raise BudgetConfigError(
                        f"budget account {self.account_key!r} is capped at {cur_calls} calls/{cur_tokens} tokens; "
                        f"requested {want_calls}/{want_tokens} would loosen it. Use a new budget id or cycle, "
                        "or pass raise_limits=True (audited) to raise it deliberately."
                    )
                if (want_calls, want_tokens) != (cur_calls, cur_tokens):
                    # without an explicit raise the cap can only shrink; counters are never touched
                    new_calls = (
                        want_calls
                        if raise_limits
                        else min(want_calls, cur_calls)
                    )
                    new_tokens = (
                        want_tokens
                        if raise_limits
                        else min(want_tokens, cur_tokens)
                    )
                    self._append(
                        conn,
                        "limit_raised" if raising else "limit_lowered",
                        payload={
                            "max_calls": new_calls,
                            "max_tokens": new_tokens,
                            "previous_max_calls": cur_calls,
                            "previous_max_tokens": cur_tokens,
                        },
                    )
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    # --------------------------------------------------------- CallBudget API
    @property  # type: ignore[override]
    def max_calls(self) -> int:  # type: ignore[override]
        return self._read(lambda c: self._limits(c)[0])

    @property  # type: ignore[override]
    def max_tokens(self) -> int:  # type: ignore[override]
        return self._read(lambda c: self._limits(c)[1])

    @property  # type: ignore[override]
    def calls_used(self) -> int:  # type: ignore[override]
        return self._read(lambda c: self._used(c)[0])

    @property  # type: ignore[override]
    def tokens_used(self) -> int:  # type: ignore[override]
        return self._read(lambda c: self._used(c)[1])

    def _read(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        conn = self._connect()
        try:
            return fn(conn)
        finally:
            conn.close()

    def reserve(self, est_tokens: int) -> Reservation:  # type: ignore[override]
        est = _strict_int(est_tokens, "est_tokens", upper=MAX_TOKENS_BOUND)
        denial: Optional[str] = None
        reservation: Optional[Reservation] = None
        conn = self._begin()
        try:
            self._check_tail(conn)
            max_calls, max_tokens = self._limits(conn)
            calls, tokens = self._used(conn)
            if tokens > max_tokens:
                denial = f"LLM token budget overspent ({tokens}>{max_tokens}); no further calls are allowed"
            elif calls + 1 > max_calls:
                denial = f"LLM call budget exhausted ({calls}/{max_calls})"
            elif tokens + est > max_tokens:
                denial = (
                    f"LLM token budget exhausted ({tokens}+{est}>{max_tokens})"
                )
            if denial is not None:
                self._append(
                    conn,
                    "reserve_denied",
                    payload={"est_tokens": est, "reason": denial},
                )
            else:
                rid = uuid.uuid4().hex
                self._append(
                    conn,
                    "reserve",
                    reservation_id=rid,
                    calls_delta=1,
                    tokens_delta=est,
                    payload={"est_tokens": est},
                )
                reservation = Reservation(est, rid, self.account_key)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()
        if denial is not None:
            raise BudgetExhausted(denial)
        assert reservation is not None
        return reservation

    def settle(  # type: ignore[override]
        self,
        reserved: int,
        actual: Optional[int],
        *,
        reservation_id: Optional[str] = None,
    ) -> None:
        rid = reservation_id or getattr(reserved, "reservation_id", None)
        if actual is not None:
            actual = _strict_int(actual, "actual", upper=MAX_REPORTED_TOKENS)
        conn = self._begin()
        try:
            if rid is None:
                # A bare int cannot identify a reservation. Never refund on a guess;
                # only record usage above what that int could have covered.
                if actual is not None and actual > int(reserved):
                    self._append(
                        conn,
                        "unattributed_settlement",
                        tokens_delta=actual - int(reserved),
                        payload={"reserved": int(reserved), "actual": actual},
                    )
            else:
                self._settle_locked(conn, rid, actual)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def _settle_locked(
        self, conn: sqlite3.Connection, rid: str, actual: Optional[int]
    ) -> None:
        res = conn.execute(
            "SELECT tokens_delta FROM budget_events WHERE account_key = ? AND reservation_id = ? AND kind = 'reserve'",
            (self.account_key, rid),
        ).fetchone()
        if res is None:
            raise BudgetError("unknown reservation id for this budget account")
        reserved = int(res[0])
        final = conn.execute(
            "SELECT kind, payload_json FROM budget_events WHERE account_key = ? AND reservation_id = ? "
            "AND kind IN ('settle', 'settle_unknown')",
            (self.account_key, rid),
        ).fetchone()
        if final is not None:
            first = json.loads(final[1])
            if (
                first.get("actual") != actual
            ):  # same report again is a silent no-op
                self._append(
                    conn,
                    "settle_conflict",
                    reservation_id=rid,
                    payload={
                        "first_actual": first.get("actual"),
                        "attempted_actual": actual,
                        "applied": False,
                    },
                )
            return
        if actual is None:
            self._append(
                conn,
                "settle_unknown",
                reservation_id=rid,
                payload={
                    "reserved": reserved,
                    "actual": None,
                    "note": "usage unknown; reservation kept",
                },
            )
            return
        self._append(
            conn,
            "settle",
            reservation_id=rid,
            tokens_delta=actual - reserved,
            payload={"reserved": reserved, "actual": actual},
        )

    def snapshot(self) -> dict[str, Any]:  # type: ignore[override]
        def read(conn: sqlite3.Connection) -> dict[str, Any]:
            max_calls, max_tokens = self._limits(conn)
            calls, tokens = self._used(conn)
            open_n = conn.execute(
                "SELECT COUNT(*) FROM budget_events r WHERE r.account_key = ? AND r.kind = 'reserve' AND NOT EXISTS "
                "(SELECT 1 FROM budget_events s WHERE s.account_key = r.account_key AND s.reservation_id = r.reservation_id "
                "AND s.kind IN ('settle', 'settle_unknown'))",
                (self.account_key,),
            ).fetchone()[0]
            unknown = conn.execute(
                "SELECT COUNT(*) FROM budget_events WHERE account_key = ? AND kind = 'settle_unknown'",
                (self.account_key,),
            ).fetchone()[0]
            denied = conn.execute(
                "SELECT COUNT(*) FROM budget_events WHERE account_key = ? AND kind = 'reserve_denied'",
                (self.account_key,),
            ).fetchone()[0]
            return {
                "max_calls": max_calls,
                "max_tokens": max_tokens,
                "calls_used": calls,
                "tokens_used": tokens,
                "remaining_calls": max(0, max_calls - calls),
                "remaining_tokens": max(0, max_tokens - tokens),
                "overspent_tokens": max(0, tokens - max_tokens),
                "unsettled_reservations": int(open_n),
                "unknown_usage_calls": int(unknown),
                "denied_reservations": int(denied),
                "persistent": True,
                "account_key": self.account_key,
            }

        return self._read(read)

    # ------------------------------------------------------------- audit
    def events(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Immutable audit events for this account, oldest first."""

        def read(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            sql = (
                "SELECT account_seq, kind, reservation_id, calls_delta, tokens_delta, payload_json, created_at, "
                "prev_hash, event_hash FROM budget_events WHERE account_key = ? ORDER BY account_seq"
            )
            args: tuple = (self.account_key,)
            if limit is not None:
                sql += " LIMIT ?"
                args += (int(limit),)
            keys = (
                "seq",
                "kind",
                "reservation_id",
                "calls_delta",
                "tokens_delta",
                "payload",
                "created_at",
                "prev_hash",
                "event_hash",
            )
            out = []
            for row in conn.execute(sql, args):
                doc = dict(zip(keys, row))
                doc["payload"] = json.loads(doc["payload"])
                out.append(doc)
            return out

        return self._read(read)

    def _check_tail(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT account_seq, kind, reservation_id, calls_delta, tokens_delta, payload_json, created_at, "
            "prev_hash, event_hash FROM budget_events WHERE account_key = ? ORDER BY account_seq DESC LIMIT 1",
            (self.account_key,),
        ).fetchone()
        if row is None:
            raise BudgetIntegrityError("budget account has no events")
        seq, kind, rid, cd, td, payload, created, prev, digest = row
        if (
            _event_hash(
                self.account_key, seq, kind, rid, cd, td, payload, created, prev
            )
            != digest
        ):
            raise BudgetIntegrityError(f"budget event {seq} fails its hash")

    def verify(self) -> int:
        """Re-verify the whole chain and the settlement invariants; returns the event count."""

        def read(conn: sqlite3.Connection) -> int:
            prev_hash = GENESIS_HASH
            expected_seq = 1
            reserved: dict[str, int] = {}
            finals: set[str] = set()
            rows = conn.execute(
                "SELECT account_seq, kind, reservation_id, calls_delta, tokens_delta, payload_json, created_at, "
                "prev_hash, event_hash FROM budget_events WHERE account_key = ? ORDER BY account_seq",
                (self.account_key,),
            ).fetchall()
            for seq, kind, rid, cd, td, payload, created, prev, digest in rows:
                if seq != expected_seq or prev != prev_hash:
                    raise BudgetIntegrityError(
                        f"budget chain broken at event {expected_seq}"
                    )
                if (
                    _event_hash(
                        self.account_key,
                        seq,
                        kind,
                        rid,
                        cd,
                        td,
                        payload,
                        created,
                        prev,
                    )
                    != digest
                ):
                    raise BudgetIntegrityError(
                        f"budget event {seq} fails its hash"
                    )
                doc = json.loads(payload)
                if kind == "reserve":
                    if (
                        cd != 1
                        or td != doc.get("est_tokens")
                        or td < 0
                        or rid in reserved
                    ):
                        raise BudgetIntegrityError(
                            f"budget reserve event {seq} is inconsistent"
                        )
                    reserved[rid] = td
                elif kind == "settle":
                    if (
                        rid not in reserved
                        or rid in finals
                        or td != doc.get("actual", -1) - reserved[rid]
                        or cd != 0
                    ):
                        raise BudgetIntegrityError(
                            f"budget settle event {seq} is inconsistent"
                        )
                    finals.add(rid)
                elif kind == "settle_unknown":
                    if (
                        rid not in reserved
                        or rid in finals
                        or td != 0
                        or cd != 0
                    ):
                        raise BudgetIntegrityError(
                            f"budget settle_unknown event {seq} is inconsistent"
                        )
                    finals.add(rid)
                elif kind in (
                    "account_created",
                    "limit_lowered",
                    "limit_raised",
                ):
                    if (
                        cd != 0
                        or td != 0
                        or "max_calls" not in doc
                        or "max_tokens" not in doc
                    ):
                        raise BudgetIntegrityError(
                            f"budget limit event {seq} is inconsistent"
                        )
                elif kind in ("reserve_denied", "settle_conflict"):
                    if cd != 0 or td != 0:
                        raise BudgetIntegrityError(
                            f"budget audit event {seq} must not change counters"
                        )
                elif kind != "unattributed_settlement":
                    raise BudgetIntegrityError(
                        f"unknown budget event kind {kind!r}"
                    )
                prev_hash = digest
                expected_seq += 1
            if not rows:
                raise BudgetIntegrityError("budget account has no events")
            return len(rows)

        return self._read(read)
