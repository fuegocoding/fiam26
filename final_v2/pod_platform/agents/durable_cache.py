"""Persistent SQLite extraction cache and cached structured LLM calls
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 20.6).

Only responses that parsed *and* validated against the schema are stored.
Entries are append-only (first stored response stays authoritative). Every
read re-verifies three hashes; a damaged entry raises instead of silently
triggering a fresh model call:

* ``fields_hash``  - the validated fields;
* ``call_hash``    - the call-provenance record, bound to the entry key and
  ``fields_hash`` so a provenance record cannot be swapped between entries;
* ``entry_hash``   - key, creation time, namespace and the two hashes above.

Caches written before ``call_hash``/``entry_hash`` existed ("legacy" rows) are
never silently trusted: their call provenance is reported as
``legacy_unverified`` until ``migrate_legacy_provenance`` is run explicitly. The
migration only *attests what is observed at migration time* in a separate
append-only table (it cannot know the record was untampered before that, and
the append-only triggers forbid rewriting the original rows), so afterwards
changes are detected, but the pre-migration history stays unverified.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from ..contracts.base import require_aware
from .extractor import ExtractionRecord, cache_key
from .provider import (
    CallBudget,
    CallRecord,
    GenerationParams,
    LLMProvider,
    LLMRequest,
    ResearchMode,
    TemporalAuthorization,
    authorize_model_use,
    call_structured,
    sha256_text,
    utc_now,
)
from .registry import ModelRegistry, TemporalPolicy


class CacheCorruptionError(RuntimeError):
    pass


class LegacyProvenanceError(RuntimeError):
    """A pre-hash cache row carries call provenance that nobody has attested."""


PROVENANCE_VERIFIED = "verified"
PROVENANCE_NONE = "none"
PROVENANCE_LEGACY_MIGRATED = "legacy_migrated"
PROVENANCE_LEGACY_UNVERIFIED = "legacy_unverified"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extraction_cache (
    key TEXT PRIMARY KEY,
    fields_json TEXT NOT NULL,
    fields_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    call_json TEXT,
    namespace TEXT NOT NULL DEFAULT '',
    call_hash TEXT,
    entry_hash TEXT
);
CREATE TRIGGER IF NOT EXISTS extraction_cache_no_update BEFORE UPDATE ON extraction_cache
BEGIN SELECT RAISE(ABORT, 'extraction cache is append-only'); END;
CREATE TRIGGER IF NOT EXISTS extraction_cache_no_delete BEFORE DELETE ON extraction_cache
BEGIN SELECT RAISE(ABORT, 'extraction cache is append-only'); END;
CREATE TABLE IF NOT EXISTS provenance_migrations (
    key TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    observed_fields_hash TEXT NOT NULL,
    observed_call_sha256 TEXT NOT NULL,
    migrated_at TEXT NOT NULL,
    operator TEXT NOT NULL,
    note TEXT NOT NULL,
    migration_hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS provenance_migrations_no_update BEFORE UPDATE ON provenance_migrations
BEGIN SELECT RAISE(ABORT, 'provenance migrations are append-only'); END;
CREATE TRIGGER IF NOT EXISTS provenance_migrations_no_delete BEFORE DELETE ON provenance_migrations
BEGIN SELECT RAISE(ABORT, 'provenance migrations are append-only'); END;
"""

_COLUMNS = "key, fields_json, fields_hash, created_at, call_json, namespace, call_hash, entry_hash"
MIGRATION_STATUS = "legacy_unverified_attested_at_migration"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compute_call_hash(key: str, fields_hash: str, call_json: str) -> str:
    return sha256_text(_canonical({"key": key, "fields_hash": fields_hash, "call_json": call_json}))


def compute_entry_hash(
    key: str, fields_hash: str, created_at: str, namespace: str, call_hash: Optional[str]
) -> str:
    return sha256_text(
        _canonical(
            {
                "key": key,
                "fields_hash": fields_hash,
                "created_at": created_at,
                "namespace": namespace,
                "call_hash": call_hash,
            }
        )
    )


def _migration_hash(row: dict[str, str]) -> str:
    return sha256_text(_canonical(row))


@dataclass(frozen=True)
class _Row:
    key: str
    fields_json: str
    fields_hash: str
    created_at: str
    call_json: Optional[str]
    namespace: str
    call_hash: Optional[str]
    entry_hash: Optional[str]

    @property
    def legacy(self) -> bool:
        return self.entry_hash is None


class DurableExtractionCache:
    """Drop-in for ``ExtractorCache`` backed by a SQLite file."""

    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._upgrade_columns(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return _Closing(conn)  # type: ignore[return-value]

    @staticmethod
    def _upgrade_columns(conn: sqlite3.Connection) -> None:
        """Schema-only upgrade of an old file: new nullable columns, no row is rewritten."""
        present = {row[1] for row in conn.execute("PRAGMA table_info(extraction_cache)")}
        for column in ("call_hash", "entry_hash"):
            if column not in present:
                try:
                    conn.execute(f"ALTER TABLE extraction_cache ADD COLUMN {column} TEXT")
                except sqlite3.OperationalError as exc:  # another process won the race
                    if "duplicate column" not in str(exc).lower():
                        raise

    # ------------------------------------------------------------- reads
    def _row(self, key: str) -> Optional[_Row]:
        with self._connect() as conn:
            raw = conn.execute(f"SELECT {_COLUMNS} FROM extraction_cache WHERE key = ?", (key,)).fetchone()
        return _Row(*raw) if raw else None

    def _verify_row(self, row: _Row) -> None:
        short = row.key[:12]
        if sha256_text(row.fields_json) != row.fields_hash:
            raise CacheCorruptionError(f"cache entry {short} failed its content hash")
        if row.legacy:
            return
        if row.call_json is None:
            if row.call_hash is not None:
                raise CacheCorruptionError(f"cache entry {short} lost its call record")
        elif row.call_hash != compute_call_hash(row.key, row.fields_hash, row.call_json):
            raise CacheCorruptionError(f"cache entry {short} failed its call-provenance hash")
        if row.entry_hash != compute_entry_hash(
            row.key, row.fields_hash, row.created_at, row.namespace, row.call_hash
        ):
            raise CacheCorruptionError(f"cache entry {short} failed its entry hash")

    def get(self, key: str) -> Optional[ExtractionRecord]:
        row = self._row(key)
        if row is None:
            return None
        self._verify_row(row)
        return ExtractionRecord(
            key=key,
            fields=json.loads(row.fields_json),
            created_at=dt.datetime.fromisoformat(row.created_at),
            from_cache=True,
        )

    def provenance_status(self, key: str) -> str:
        """``verified`` | ``none`` | ``legacy_migrated`` | ``legacy_unverified`` (raises on tamper)."""
        row = self._row(key)
        if row is None:
            raise KeyError(key)
        self._verify_row(row)
        if row.call_json is None:
            return PROVENANCE_NONE
        if not row.legacy:
            return PROVENANCE_VERIFIED
        migration = self._migration(key)
        if migration is None:
            return PROVENANCE_LEGACY_UNVERIFIED
        check = {k: migration[k] for k in migration if k != "migration_hash"}
        if _migration_hash(check) != migration["migration_hash"]:
            raise CacheCorruptionError(f"provenance migration for {key[:12]} failed its hash")
        if (
            migration["observed_call_sha256"] != sha256_text(row.call_json)
            or migration["observed_fields_hash"] != row.fields_hash
        ):
            raise CacheCorruptionError(f"legacy cache entry {key[:12]} changed after its provenance migration")
        return PROVENANCE_LEGACY_MIGRATED

    def get_call_record(self, key: str, *, allow_unverified: bool = False) -> Optional[CallRecord]:
        """The stored call provenance, hash-verified.

        A legacy row that was never migrated raises ``LegacyProvenanceError``
        unless ``allow_unverified`` (the record is then returned *unverified*).
        """
        row = self._row(key)
        if row is None or not row.call_json:
            if row is not None:
                self._verify_row(row)
            return None
        status = self.provenance_status(key)
        if status == PROVENANCE_LEGACY_UNVERIFIED and not allow_unverified:
            raise LegacyProvenanceError(
                f"cache entry {key[:12]} predates provenance hashes; run migrate_legacy_provenance() explicitly"
            )
        try:
            return CallRecord.model_validate_json(row.call_json)
        except ValueError as exc:
            raise CacheCorruptionError(f"cache entry {key[:12]} call record no longer validates: {exc}") from None

    def _migration(self, key: str) -> Optional[dict[str, str]]:
        with self._connect() as conn:
            raw = conn.execute(
                "SELECT key, status, observed_fields_hash, observed_call_sha256, migrated_at, operator, note, "
                "migration_hash FROM provenance_migrations WHERE key = ?",
                (key,),
            ).fetchone()
        if raw is None:
            return None
        names = (
            "key", "status", "observed_fields_hash", "observed_call_sha256", "migrated_at", "operator", "note",
            "migration_hash",
        )
        return dict(zip(names, raw))

    # ------------------------------------------------------------ writes
    def put(
        self,
        key: str,
        record: ExtractionRecord,
        *,
        call: Optional[CallRecord] = None,
        namespace: str = "",
    ) -> bool:
        """Store once; returns False if the key already existed."""
        created = require_aware(record.created_at, "created_at").isoformat()
        fields_json = _canonical(record.fields)
        fields_hash = sha256_text(fields_json)
        call_json = call.model_dump_json() if call else None
        call_hash = compute_call_hash(key, fields_hash, call_json) if call_json is not None else None
        entry_hash = compute_entry_hash(key, fields_hash, created, namespace, call_hash)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO extraction_cache ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?)",
                (key, fields_json, fields_hash, created, call_json, namespace, call_hash, entry_hash),
            )
            return cur.rowcount == 1

    def migrate_legacy_provenance(
        self, *, operator: str, note: str, at: Optional[dt.datetime] = None
    ) -> dict[str, Any]:
        """Explicitly attest the legacy rows' call records *as currently stored*.

        Rows whose ``fields_json`` no longer matches ``fields_hash`` are reported
        as ``corrupt`` and are not migrated. Nothing is validated retroactively:
        the status stays ``legacy_*`` and the note records who decided to trust
        what was observed.
        """
        if not operator.strip() or not note.strip():
            raise ValueError("a migration needs an operator and a note")
        stamp = require_aware(at or utc_now(), "at").isoformat()
        migrated: list[str] = []
        corrupt: list[str] = []
        already = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM extraction_cache WHERE entry_hash IS NULL AND call_json IS NOT NULL "
                    "ORDER BY key"
                ).fetchall()
                for raw in rows:
                    row = _Row(*raw)
                    if conn.execute("SELECT 1 FROM provenance_migrations WHERE key = ?", (row.key,)).fetchone():
                        already += 1
                        continue
                    if sha256_text(row.fields_json) != row.fields_hash:
                        corrupt.append(row.key)
                        continue
                    body = {
                        "key": row.key,
                        "status": MIGRATION_STATUS,
                        "observed_fields_hash": row.fields_hash,
                        "observed_call_sha256": sha256_text(row.call_json or ""),
                        "migrated_at": stamp,
                        "operator": operator.strip(),
                        "note": note.strip(),
                    }
                    conn.execute(
                        "INSERT INTO provenance_migrations VALUES (?,?,?,?,?,?,?,?)",
                        (*body.values(), _migration_hash(body)),
                    )
                    migrated.append(row.key)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return {
            "migrated": len(migrated),
            "already_migrated": already,
            "corrupt": corrupt,
            "status": MIGRATION_STATUS,
            "validated_retroactively": False,
        }

    def verify_all(self) -> dict[str, Any]:
        """Check every row; tamper is reported (not raised) so the whole file can be audited."""
        with self._connect() as conn:
            keys = [r[0] for r in conn.execute("SELECT key FROM extraction_cache ORDER BY key")]
        counts: dict[str, int] = {}
        errors: list[dict[str, str]] = []
        for key in keys:
            try:
                status = self.provenance_status(key)
            except CacheCorruptionError as exc:
                errors.append({"key": key, "error": str(exc)})
                continue
            counts[status] = counts.get(status, 0) + 1
        return {"entries": len(keys), "provenance": counts, "errors": errors}

    def __len__(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM extraction_cache").fetchone()[0]


class _Closing:
    """sqlite3 connection that closes on ``with`` exit (stdlib's does not)."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self._conn

    def __exit__(self, *exc: Any) -> None:
        self._conn.close()


@dataclass(frozen=True)
class StructuredRun:
    value: BaseModel
    record: ExtractionRecord
    call: Optional[CallRecord]  # provenance of the call that produced the response
    from_cache: bool
    authorization: TemporalAuthorization
    cache_key: str
    provider: str
    model: str
    provider_is_mock: bool
    call_provenance: str = PROVENANCE_VERIFIED  # see DurableExtractionCache.provenance_status


class CachedStructuredExtractor:
    """provider + schema + durable cache + budget + temporal gate."""

    def __init__(
        self,
        provider: LLMProvider,
        cache: DurableExtractionCache,
        schema: type[BaseModel],
        system_prompt: str,
        *,
        budget: Optional[CallBudget] = None,
        registry: Optional[ModelRegistry] = None,
        clock: Callable[[], dt.datetime] = utc_now,
    ):
        self.provider = provider
        self.cache = cache
        self.schema = schema
        self.system_prompt = system_prompt
        self.budget = budget
        self.registry = registry
        self.clock = clock
        self.prompt_hash = sha256_text(system_prompt + "\n" + _canonical(schema.model_json_schema()))

    def key_for(
        self,
        user_text: str,
        *,
        role: str,
        namespace: str,
        research_asof: dt.datetime,
        mode: ResearchMode,
        params: GenerationParams,
        checkpoint_id: Optional[str],
    ) -> str:
        checkpoint_revision = "unregistered"
        if checkpoint_id and self.registry is not None:
            checkpoint_revision = self.registry.checkpoint(checkpoint_id).checkpoint_revision
        return cache_key(
            model_id=f"{self.provider.name}:{self.provider.model}",
            checkpoint_revision=checkpoint_revision,
            prompt_hash=self.prompt_hash,
            input_hashes={"user": sha256_text(user_text)},
            research_asof=research_asof,
            generation_params=params.identity(),
            role=f"{role}|{namespace}|{ResearchMode(mode).value}",
        )

    def run(
        self,
        user_payload: dict[str, Any],
        *,
        role: str,
        namespace: str,
        research_asof: dt.datetime,
        mode: ResearchMode,
        checkpoint_id: Optional[str] = None,
        policy: TemporalPolicy = TemporalPolicy.RELEASE_GATED,
        params: Optional[GenerationParams] = None,
        validator: Optional[Callable[[BaseModel], None]] = None,
    ) -> StructuredRun:
        """``validator`` runs on fresh *and* cached values; a fresh value that
        fails it is not cached."""
        params = params or GenerationParams()
        auth = authorize_model_use(
            mode=mode,
            research_asof=research_asof,
            provider=self.provider,
            registry=self.registry,
            checkpoint_id=checkpoint_id,
            policy=policy,
            now=self.clock(),
        )
        user_text = _canonical(user_payload)
        key = self.key_for(
            user_text,
            role=role,
            namespace=namespace,
            research_asof=auth.research_asof,
            mode=mode,
            params=params,
            checkpoint_id=checkpoint_id,
        )
        hit = self.cache.get(key)
        if hit is not None:
            try:
                value = self.schema.model_validate(hit.fields)
            except ValueError as exc:
                raise CacheCorruptionError(f"cached entry {key[:12]} no longer validates: {exc}") from None
            if validator is not None:
                validator(value)
            provenance = self.cache.provenance_status(key)
            call: Optional[CallRecord] = None
            if provenance != PROVENANCE_LEGACY_UNVERIFIED:  # unattested provenance is not attributed
                call = self.cache.get_call_record(key)
            return StructuredRun(
                value=value,
                record=hit,
                call=call,
                from_cache=True,
                authorization=auth,
                cache_key=key,
                provider=self.provider.name,
                model=self.provider.model,
                provider_is_mock=bool(getattr(self.provider, "is_mock", False)),
                call_provenance=provenance,
            )
        request = LLMRequest(system=self.system_prompt, user=user_text, params=params, role=role)
        result = call_structured(
            self.provider, request, self.schema, budget=self.budget, clock=self.clock, validator=validator
        )
        record = ExtractionRecord(
            key=key,
            fields=json.loads(_canonical(result.value.model_dump(mode="json"))),
            created_at=result.record.received_at,
            from_cache=False,
        )
        self.cache.put(key, record, call=result.record, namespace=namespace)
        return StructuredRun(
            value=result.value,
            record=record,
            call=result.record,
            from_cache=False,
            authorization=auth,
            cache_key=key,
            provider=self.provider.name,
            model=self.provider.model,
            provider_is_mock=result.record.provider_is_mock,
            call_provenance=PROVENANCE_VERIFIED,
        )
