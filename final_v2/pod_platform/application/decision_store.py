"""Persistent decision-event store and position/date dossier lookup
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 6.2 ``decisions.explain``, 21.10, 24.7, 26.8 item 9).

``reporting.dossier.DossierBuilder`` keeps events in memory and matches a security
by substring. This store persists events with *explicit* security tags and the
actual post-trade positions by holding period, so a judge can ask "why was this
security held on this date?" and get the formation decision chain that governed
that date - not only the earliest or latest event. If nothing was stored the
lookup fails with a typed error (the run is then incomplete, never narrated).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..contracts.base import require_aware
from ..contracts.decisions import ActorKind, DecisionEvent
from ..contracts.errors import ErrorCode
from ..contracts.forecasts import Forecast
from ..research.promotion import AdmissionCertificate
from .common import UTC, digest, fail, read_json, utc_close

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    effective_market_at TEXT NOT NULL,
    digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (run_id, event_id)
);
CREATE TABLE IF NOT EXISTS event_securities (
    run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    PRIMARY KEY (run_id, event_id, security_id)
);
CREATE INDEX IF NOT EXISTS event_securities_lookup ON event_securities (run_id, security_id);
CREATE TABLE IF NOT EXISTS positions (
    run_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    formation_date TEXT NOT NULL,
    period_end TEXT NOT NULL,
    weight TEXT NOT NULL,
    side TEXT NOT NULL,
    PRIMARY KEY (run_id, security_id, formation_date)
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'decision events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'decision events are never deleted'); END;
"""


def _as_date(value: "dt.date | dt.datetime") -> dt.date:
    if isinstance(value, dt.datetime):
        return require_aware(value, "as_of").astimezone(UTC).date()
    return value


def _event_digest(event: DecisionEvent) -> str:
    return event.content_hash or event.digest()


class DecisionStore:
    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=30)

    # --------------------------------------------------------------- writes
    def record(self, event: DecisionEvent, *, security_ids: Sequence[str], run_id: str) -> bool:
        """Append one event with explicit security tags. Identical re-record is a no-op."""
        return self.record_many([(event, tuple(security_ids))], run_id=run_id) == 1

    def record_many(self, items: Iterable[tuple[DecisionEvent, Sequence[str]]], *, run_id: str) -> int:
        inserted = 0
        conn = self._connect()
        try:
            with conn:
                for event, securities in items:
                    if not securities:
                        fail(ErrorCode.SCHEMA_MISMATCH, f"decision event {event.event_id} has no security tag",
                             "Events are stored with explicit security ids (no substring matching).")
                    sealed = event if event.content_hash else event.sealed()
                    row = conn.execute("SELECT digest FROM events WHERE run_id = ? AND event_id = ?", (run_id, event.event_id)).fetchone()
                    if row is not None:
                        if row[0] != _event_digest(sealed):
                            fail(ErrorCode.SCHEMA_MISMATCH, f"decision event {event.event_id} already stored with different content",
                                 "The decision ledger is append-only.")
                        continue
                    conn.execute(
                        "INSERT INTO events (run_id, event_id, effective_market_at, digest, payload_json) VALUES (?,?,?,?,?)",
                        (
                            run_id,
                            event.event_id,
                            event.effective_market_at.astimezone(UTC).isoformat(),
                            _event_digest(sealed),
                            json.dumps(sealed.model_dump(mode="json"), sort_keys=True),
                        ),
                    )
                    conn.executemany(
                        "INSERT OR IGNORE INTO event_securities (run_id, event_id, security_id) VALUES (?,?,?)",
                        [(run_id, event.event_id, sid) for sid in sorted(set(securities))],
                    )
                    inserted += 1
        finally:
            conn.close()
        return inserted

    def record_positions(self, run_id: str, rows: Iterable[Mapping[str, Any]]) -> int:
        conn = self._connect()
        count = 0
        try:
            with conn:
                for row in rows:
                    conn.execute(
                        "INSERT OR IGNORE INTO positions (run_id, security_id, formation_date, period_end, weight, side)"
                        " VALUES (?,?,?,?,?,?)",
                        (run_id, row["security_id"], row["formation_date"], row["period_end"], str(row["weight"]), row["side"]),
                    )
                    count += 1
        finally:
            conn.close()
        return count

    # ------------------------------------------------------------- ingest
    def ingest_replay_run(
        self,
        run_dir: "str | Path",
        *,
        run_id: str,
        forecasts: Sequence[Forecast],
        certificates: Mapping[str, AdmissionCertificate],
    ) -> dict[str, int]:
        """Rebuild the decision chain of a ControlledReplay output from its stored artifacts."""
        run_dir = Path(run_dir)
        periods = read_json(run_dir / "periods.json")
        monthly = {m["period"]: m for m in read_json(run_dir / "monthly.json")}
        by_id = {f.forecast_id: f for f in forecasts}
        held: dict[str, dict[str, dict[str, str]]] = {}
        with (run_dir / "holdings_by_period.csv").open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                held.setdefault(row["period"], {})[row["security_id"]] = row
        position_rows: list[dict[str, Any]] = []
        items: list[tuple[DecisionEvent, Sequence[str]]] = []
        for period in periods:
            label = period["period"]
            formation = dt.date.fromisoformat(label)
            at = utc_close(formation)
            period_end = monthly[label]["period_end"]
            holdings = held.get(label, {})
            for sid, row in sorted(holdings.items()):
                position_rows.append(
                    {"security_id": sid, "formation_date": label, "period_end": period_end,
                     "weight": row["post_trade_weight"], "side": row["side"]}
                )

                def event(event_id: str, action: str, result_id: str, inputs: dict[str, str], authority: str) -> DecisionEvent:
                    return DecisionEvent(
                        owner="orchestration",
                        created_at=at,
                        event_id=event_id,
                        actor_kind=ActorKind.CODE,
                        actor_id="controlled_replay",
                        effective_research_at=at,
                        effective_market_at=at,
                        input_artifact_ids=sorted(inputs),
                        input_hashes=dict(inputs),
                        chosen_action=action,
                        rationale="deterministic controlled replay step reconstructed from stored artifacts",
                        authorizing_rule_or_certificate_id=authority,
                        resulting_artifact_id=result_id,
                    )

                for fid in period.get("forecast_ids", []):
                    f = by_id.get(fid)
                    if f is None or f.security_id != sid:
                        continue
                    cert = certificates.get(f.admission_certificate_id)
                    inputs = {f.forecast_id: _event_digest(f)}
                    if cert is not None:
                        inputs[cert.certificate_id] = _event_digest(cert)
                    items.append((event(f"forecast_{label}_{sid}_{f.strategy_id}", f"admit forecast {fid} for {sid}", fid, inputs,
                                        f.admission_certificate_id), [sid]))
                items.append((event(f"target_{label}_{sid}", f"target position in {sid} from {period['target_book_id']}",
                                    period["target_book_id"], {period["target_book_id"]: period["target_book_digest"]},
                                    "rule_deterministic_construction_v1"), [sid]))
                items.append((event(f"review_{label}_{sid}", f"risk review {period['risk_review_id']} for {sid}",
                                    period["risk_review_id"], {period["risk_review_id"]: period["risk_review_digest"]},
                                    "risk_policy_in_run_config"), [sid]))
                for order in period.get("orders", []):
                    if order["security_id"] == sid:
                        items.append((event(f"fill_{label}_{order['order_id']}", f"fill order for {sid}", f"fill_{order['order_id']}",
                                            {order["order_id"]: digest(order)}, "execution_policy_in_run_config"), [sid]))
        events = self.record_many(items, run_id=run_id)
        positions = self.record_positions(run_id, position_rows)
        return {"events": events, "positions": positions, "periods": len(periods)}

    # -------------------------------------------------------------- reads
    def run_ids(self) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT DISTINCT run_id FROM positions UNION SELECT DISTINCT run_id FROM events ORDER BY 1").fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]

    def positions_on(self, on: "dt.date | dt.datetime", run_id: str) -> list[dict[str, str]]:
        day = _as_date(on).isoformat()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT security_id, weight, side, formation_date, period_end FROM positions"
                " WHERE run_id = ? AND formation_date <= ? AND ? < period_end ORDER BY security_id",
                (run_id, day, day),
            ).fetchall()
        finally:
            conn.close()
        return [
            {"security_id": r[0], "weight": r[1], "side": r[2], "formation_date": r[3], "period_end": r[4]} for r in rows
        ]

    def _events(self, conn: sqlite3.Connection, run_id: str, security_id: str, upto: str) -> list[tuple[str, DecisionEvent, str]]:
        rows = conn.execute(
            "SELECT e.event_id, e.effective_market_at, e.digest, e.payload_json FROM events e"
            " JOIN event_securities s ON s.run_id = e.run_id AND s.event_id = e.event_id"
            " WHERE e.run_id = ? AND s.security_id = ? AND e.effective_market_at <= ?"
            " ORDER BY e.effective_market_at, e.event_id",
            (run_id, security_id, upto),
        ).fetchall()
        out: list[tuple[str, DecisionEvent, str]] = []
        for event_id, at, stored_digest, payload in rows:
            event = DecisionEvent.model_validate(json.loads(payload))
            if not event.verify_seal() or event.content_hash != stored_digest:
                fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"stored decision event {event_id} failed its seal",
                     "A damaged decision record is never narrated.")
            out.append((at, event, stored_digest))
        return out

    def explain(self, security_id: str, on: "dt.date | dt.datetime", run_id: Optional[str] = None) -> dict[str, Any]:
        """The actual position on ``on`` and the formation decision chain that governed it."""
        day = _as_date(on)
        runs = [run_id] if run_id else self.run_ids()
        if not runs:
            fail(ErrorCode.SCHEMA_MISMATCH, "the decision store is empty", "An arbitrary position/date must be explainable from stored evidence.")
        last_error: Optional[str] = None
        for rid in runs:
            result = self._explain_run(security_id, day, rid)
            if result is not None:
                return result
            last_error = rid
        fail(
            ErrorCode.SCHEMA_MISMATCH,
            f"no stored decision evidence for {security_id} on {day.isoformat()} (searched {', '.join(runs)})",
            "If a position/date cannot be explained from stored records, the run is incomplete.",
        )

    def _explain_run(self, security_id: str, day: dt.date, run_id: str) -> Optional[dict[str, Any]]:
        conn = self._connect()
        try:
            end_of_day = dt.datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=UTC).isoformat()
            held = conn.execute(
                "SELECT formation_date, period_end, weight, side FROM positions WHERE run_id = ? AND security_id = ?"
                " AND formation_date <= ? AND ? < period_end ORDER BY formation_date DESC LIMIT 1",
                (run_id, security_id, day.isoformat(), day.isoformat()),
            ).fetchone()
            previous = None
            if held is None:
                previous = conn.execute(
                    "SELECT formation_date, period_end, weight, side FROM positions WHERE run_id = ? AND security_id = ?"
                    " AND period_end <= ? ORDER BY formation_date DESC LIMIT 1",
                    (run_id, security_id, day.isoformat()),
                ).fetchone()
            events = self._events(conn, run_id, security_id, end_of_day)
        finally:
            conn.close()
        if held is None and not events:
            return None
        governing = held or previous
        if governing is not None:
            governing_date = governing[0]
            chain = [(at, e, d) for at, e, d in events if at[:10] == governing_date]
        else:
            governing_date = events[-1][0][:10]
            chain = [(at, e, d) for at, e, d in events if at[:10] == governing_date]
        chain_events = [e for _at, e, _d in chain]
        earlier = len(events) - len(chain)
        return {
            "run_id": run_id,
            "security_id": security_id,
            "as_of": day.isoformat(),
            "held_on_date": held is not None,
            "position": (
                {"formation_date": held[0], "period_end": held[1], "weight": held[2], "side": held[3]} if held is not None else None
            ),
            "last_held_period": (
                {"formation_date": previous[0], "period_end": previous[1], "weight": previous[2], "side": previous[3]}
                if previous is not None else None
            ),
            "governing_formation_date": governing_date,
            "decision_events": [
                {
                    "event_id": e.event_id,
                    "actor_kind": e.actor_kind.value,
                    "effective_market_at": e.effective_market_at.isoformat(),
                    "chosen_action": e.chosen_action,
                    "rationale": e.rationale,
                    "authorizing_rule_or_certificate_id": e.authorizing_rule_or_certificate_id,
                    "resulting_artifact_id": e.resulting_artifact_id,
                    "input_hashes": dict(sorted(e.input_hashes.items())),
                    "event_digest": d,
                }
                for _at, e, d in chain
            ],
            "forecast_ids": sorted({e.resulting_artifact_id for e in chain_events if e.event_id.startswith("forecast_")}),
            "risk_review_ids": sorted({e.resulting_artifact_id for e in chain_events if e.event_id.startswith("review_")}),
            "order_ids": sorted({e.resulting_artifact_id.removeprefix("fill_") for e in chain_events if e.event_id.startswith("fill_")}),
            "target_book_ids": sorted({e.resulting_artifact_id for e in chain_events if e.event_id.startswith("target_")}),
            "earlier_decision_events": earlier,
            "provenance": {"events_digest": digest([d for _at, _e, d in chain]), "store": "decision_store.sqlite"},
        }
