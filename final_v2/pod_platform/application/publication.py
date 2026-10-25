"""Forecast publication boundary (POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 6.2 ``forecasts.publish``,
14.3, 15.2, 21.6, 21.12).

Private strategy output becomes a *published*, immutable forecast batch only when
every validation in spec 21.6 holds: the certificate covers this model, target
and horizon; the forecast carries no future evidence; units/horizon match the
integration policy; formation follows eligibility; evidence and uncertainty
artifacts exist; the forecast is not expired. Allocation reads published
batches only. Stale (expired), future-dated and uncertified forecasts are never
returned as active. Fixture-scope certificates cannot be published into a
formation-scope store.
"""

from __future__ import annotations

import datetime as dt
import enum
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from pydantic import Field

from ..contracts.base import DurableModel, require_aware
from ..contracts.errors import ErrorCode
from ..contracts.forecasts import RETURN_UNIT, Forecast
from ..research.certificates import CertificatePolicy, validate_certificate
from ..research.promotion import AdmissionCertificate
from ..research.registry import Registry
from .common import (
    FIXTURE_LIMITATION,
    FIXTURE_MODE_PREFIX,
    FORMATION_MODE_PREFIX,
    UTC,
    digest,
    fail,
)
from .namespaces import RoleContext, RoleGateway

SCOPE_FIXTURE = "fixture"
SCOPE_RESEARCH = "research"
SCOPE_FORMATION = "formation"
SCOPES = (SCOPE_FIXTURE, SCOPE_RESEARCH, SCOPE_FORMATION)


def certificate_scope(certificate: AdmissionCertificate) -> str:
    """Machine-readable certificate scope.

    Prefers an explicit ``scope`` attribute (added by the controls patch when
    present); otherwise parses ``approved_mode`` and the fixture limitation marker.
    """
    explicit = getattr(certificate, "scope", None)
    if explicit is not None and hasattr(explicit, "environment"):
        environment = getattr(
            explicit.environment, "value", explicit.environment
        )
        return {
            "test": SCOPE_FIXTURE,
            "shadow": SCOPE_RESEARCH,
            "formation": SCOPE_FORMATION,
        }.get(environment, SCOPE_RESEARCH)
    if isinstance(explicit, enum.Enum):
        explicit = explicit.value
    if isinstance(explicit, str) and explicit:
        lowered = explicit.lower()
        if lowered.startswith(FIXTURE_MODE_PREFIX):
            return SCOPE_FIXTURE
        if lowered.startswith(FORMATION_MODE_PREFIX):
            return SCOPE_FORMATION
        return SCOPE_RESEARCH
    mode = str(certificate.approved_mode).lower()
    if (
        mode.startswith(FIXTURE_MODE_PREFIX)
        or FIXTURE_LIMITATION in certificate.known_limitations
    ):
        return SCOPE_FIXTURE
    if mode.startswith(FORMATION_MODE_PREFIX):
        return SCOPE_FORMATION
    return SCOPE_RESEARCH


class ModelVersionRecord(DurableModel):
    """Registry ``model_versions`` row: the exact conventional model behind a forecast."""

    ID_FIELD = "model_id"

    model_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    charter: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    model_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bridge_id: str = Field(min_length=1)
    bridge_slope_bp: str = Field(min_length=1)
    training_cutoff: dt.datetime
    uses_pretrained_model: bool = False
    feature_names: list[str] = Field(default_factory=list)


class ForecastBatch(DurableModel):
    """Registry ``forecast_publications`` row; the forecasts themselves live in the store."""

    ID_FIELD = "batch_id"

    batch_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    admission_certificate_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    information_cutoff: dt.datetime
    published_at: dt.datetime
    forecast_count: int = Field(ge=1)
    forecast_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    scope: str = Field(pattern=r"^(fixture|research|formation)$")
    earliest_expiry: dt.datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    model_id TEXT NOT NULL,
    certificate_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    published_at TEXT NOT NULL,
    digest TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    namespace TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    security_id TEXT NOT NULL,
    information_cutoff TEXT NOT NULL,
    earliest_formation_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    digest TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS forecasts_ns ON forecasts (namespace, security_id);
CREATE TRIGGER IF NOT EXISTS batches_no_update BEFORE UPDATE ON batches
BEGIN SELECT RAISE(ABORT, 'published batches are immutable'); END;
CREATE TRIGGER IF NOT EXISTS batches_no_delete BEFORE DELETE ON batches
BEGIN SELECT RAISE(ABORT, 'published batches are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts
BEGIN SELECT RAISE(ABORT, 'published forecasts are immutable'); END;
CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts
BEGIN SELECT RAISE(ABORT, 'published forecasts are never deleted'); END;
"""


def _ts(value: dt.datetime) -> str:
    return require_aware(value, "timestamp").astimezone(UTC).isoformat()


class PublicationStore:
    def __init__(self, path: "str | Path"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def has_batch(self, batch_id: str) -> Optional[str]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT digest FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def insert_batch(
        self, batch: ForecastBatch, forecasts: Sequence[Forecast]
    ) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO batches (batch_id, namespace, model_id, certificate_id, scope, published_at, digest,"
                    " payload_json) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        batch.batch_id,
                        batch.namespace,
                        batch.model_id,
                        batch.admission_certificate_id,
                        batch.scope,
                        _ts(batch.published_at),
                        batch.content_hash or batch.digest(),
                        json.dumps(
                            batch.model_dump(mode="json"), sort_keys=True
                        ),
                    ),
                )
                conn.executemany(
                    "INSERT INTO forecasts (forecast_id, batch_id, namespace, strategy_id, security_id,"
                    " information_cutoff, earliest_formation_at, created_at, expires_at, digest, payload_json)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            f.forecast_id,
                            batch.batch_id,
                            batch.namespace,
                            f.strategy_id,
                            f.security_id,
                            _ts(f.information_cutoff),
                            _ts(f.earliest_formation_at),
                            _ts(f.created_at),
                            _ts(f.expires_at),
                            f.content_hash or f.digest(),
                            json.dumps(
                                f.model_dump(mode="json"), sort_keys=True
                            ),
                        )
                        for f in forecasts
                    ],
                )
        finally:
            conn.close()

    def fetch(
        self, namespaces: Optional[Sequence[str]] = None
    ) -> list[tuple[str, str, Forecast]]:
        """Return (namespace, batch_id, Forecast) rows, verified against their stored digests."""
        sql = "SELECT namespace, batch_id, digest, payload_json FROM forecasts"
        args: list[Any] = []
        if namespaces is not None:
            if not namespaces:
                return []
            sql += " WHERE namespace IN (%s)" % ",".join("?" * len(namespaces))
            args.extend(namespaces)
        sql += " ORDER BY forecast_id"
        conn = self._connect()
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
        out: list[tuple[str, str, Forecast]] = []
        for namespace, batch_id, stored_digest, payload in rows:
            forecast = Forecast.model_validate(json.loads(payload))
            if (
                not forecast.verify_seal()
                or forecast.content_hash != stored_digest
            ):
                fail(
                    ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                    f"published forecast {forecast.forecast_id} failed its seal",
                    "A damaged published forecast is never returned.",
                )
            out.append((namespace, batch_id, forecast))
        return out

    def batches(
        self, namespaces: Optional[Sequence[str]] = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT payload_json FROM batches"
        args: list[Any] = []
        if namespaces is not None:
            if not namespaces:
                return []
            sql += " WHERE namespace IN (%s)" % ",".join("?" * len(namespaces))
            args.extend(namespaces)
        sql += " ORDER BY published_at, batch_id"
        conn = self._connect()
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
        return [json.loads(r[0]) for r in rows]


@dataclass
class ActiveForecasts:
    decision_at: dt.datetime
    active: list[Forecast] = field(default_factory=list)
    stale_counts: dict[str, int] = field(default_factory=dict)
    stale_examples: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_at": self.decision_at.isoformat(),
            "active": len(self.active),
            "stale_counts": dict(sorted(self.stale_counts.items())),
            "stale_examples": list(self.stale_examples),
        }


class PublicationService:
    def __init__(
        self,
        path: "str | Path",
        *,
        registry: Optional[Registry] = None,
        gateway: Optional[RoleGateway] = None,
        store_scope: str = SCOPE_FIXTURE,
    ):
        if store_scope not in SCOPES:
            raise ValueError(f"unknown store scope {store_scope!r}")
        self.store = PublicationStore(path)
        self.registry = registry
        self.gateway = gateway or RoleGateway()
        self.store_scope = store_scope

    # ------------------------------------------------------------- publish
    def publish(
        self,
        ctx: RoleContext,
        *,
        forecasts: Sequence[Forecast],
        certificate: AdmissionCertificate,
        model: ModelVersionRecord,
        published_at: dt.datetime,
        artifact_exists: Callable[[str], bool],
        integration_horizon_days: int = 20,
        source_snapshots: Optional[dict[str, str]] = None,
    ) -> ForecastBatch:
        published_at = require_aware(published_at, "published_at")
        self.gateway.authorize(ctx, "publication.publish", model.namespace)
        if not forecasts:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                "an empty forecast batch cannot be published",
                "Publish at least one forecast.",
            )
        if not certificate.verify_seal():
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"certificate {certificate.certificate_id} is unsealed or its seal fails",
                "Allocation refuses forecasts without a valid certificate (spec 14.3).",
            )
        scope = certificate_scope(certificate)
        if scope == SCOPE_FIXTURE and self.store_scope == SCOPE_FORMATION:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"certificate {certificate.certificate_id} is fixture-scoped and cannot enter a formation store",
                "Fixture certificates never authorize a competition book (spec 27.4).",
            )
        if scope != SCOPE_FORMATION and self.store_scope == SCOPE_FORMATION:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"certificate {certificate.certificate_id} has scope {scope!r}, not formation",
                "A formation store accepts only formation-scoped certificates.",
            )
        if (
            certificate.model_id != model.model_id
            or certificate.model_hash != model.model_hash
        ):
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"certificate {certificate.certificate_id} does not cover model {model.model_id}",
                "Certificate model id and hash must match the forecasting model.",
            )
        if certificate.horizon_trading_days != integration_horizon_days:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"certificate horizon {certificate.horizon_trading_days}d differs from the integration horizon {integration_horizon_days}d",
                "Forecasts with unresolved horizon conversion are excluded from allocation (spec 15.4).",
            )
        seen: set[str] = set()
        for f in forecasts:
            if f.forecast_id in seen:
                fail(
                    ErrorCode.SCHEMA_MISMATCH,
                    f"duplicate forecast id {f.forecast_id} in the batch",
                    "Forecast ids are unique.",
                )
            seen.add(f.forecast_id)
            self._validate_forecast(
                f,
                certificate,
                model,
                published_at,
                artifact_exists,
                integration_horizon_days,
            )
            validate_certificate(
                f,
                certificate,
                decision_at=f.earliest_formation_at,
                source_snapshots=source_snapshots,
                universe=certificate.approved_universe,
                namespace=model.namespace,
                policy=CertificatePolicy.production()
                if self.store_scope == SCOPE_FORMATION
                else CertificatePolicy.test(),
            )
        sealed = [f if f.content_hash else f.sealed() for f in forecasts]
        for f in sealed:
            if not f.verify_seal():
                fail(
                    ErrorCode.SCHEMA_MISMATCH,
                    f"forecast {f.forecast_id} seal does not verify",
                    "Forecasts are sealed before publication.",
                )
        info_cutoffs = {f.information_cutoff for f in sealed}
        newest = max(info_cutoffs)
        forecast_digest = digest(sorted(f.content_hash for f in sealed))
        batch_id = f"batch_{model.model_id}_{newest.astimezone(UTC).strftime('%Y%m%d')}_{forecast_digest[:10]}"
        batch = ForecastBatch(
            owner=model.namespace,
            created_at=published_at,
            batch_id=batch_id,
            model_id=model.model_id,
            admission_certificate_id=certificate.certificate_id,
            namespace=model.namespace,
            information_cutoff=newest,
            published_at=published_at,
            forecast_count=len(sealed),
            forecast_digest=forecast_digest,
            scope=scope,
            earliest_expiry=min(f.expires_at for f in sealed),
        ).sealed()
        existing = self.store.has_batch(batch_id)
        if existing is not None:
            if existing == batch.content_hash:
                return batch  # idempotent re-publication of the identical batch
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"batch {batch_id} already published with different content",
                "Published batches are immutable.",
            )
        self._register(model, certificate, batch)
        self.store.insert_batch(batch, sealed)
        return batch

    def _validate_forecast(
        self,
        f: Forecast,
        certificate: AdmissionCertificate,
        model: ModelVersionRecord,
        published_at: dt.datetime,
        artifact_exists: Callable[[str], bool],
        horizon: int,
    ) -> None:
        if f.model_id != model.model_id or f.strategy_id != model.candidate_id:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"forecast {f.forecast_id} does not belong to model {model.model_id}",
                "A strategy publishes only forecasts of its own admitted model.",
            )
        if f.admission_certificate_id != certificate.certificate_id:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"forecast {f.forecast_id} cites certificate {f.admission_certificate_id}, not {certificate.certificate_id}",
                "Every forecast must cite the certificate that authorizes it.",
            )
        if certificate.target_id != f.target.target_id:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"forecast {f.forecast_id} target {f.target.target_id!r} is outside certificate target {certificate.target_id!r}",
                "Certificate target must match the published forecast.",
            )
        if (
            f.return_forecast.horizon_trading_days != horizon
            or f.return_forecast.unit != RETURN_UNIT
        ):
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"forecast {f.forecast_id} horizon/unit does not match the integration policy ({horizon}d {RETURN_UNIT})",
                "Units and horizon must match the registered integration policy.",
            )
        if not certificate.is_valid_at(f.earliest_formation_at):
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"certificate {certificate.certificate_id} is not yet effective at formation {f.earliest_formation_at.isoformat()}",
                "A forecast cannot be formed before its certificate's effective start.",
            )
        if certificate.created_at > f.created_at:
            fail(
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                f"certificate {certificate.certificate_id} was issued {certificate.created_at.isoformat()}, after forecast {f.forecast_id}",
                "A certificate issued after the forecast cannot authorize it.",
            )
        if (
            f.created_at < f.information_cutoff
            or f.earliest_formation_at < f.information_cutoff
        ):
            fail(
                ErrorCode.ASOF_VIOLATION,
                f"forecast {f.forecast_id} predates its own information cutoff",
                "Information cutoff precedes publication and formation.",
            )
        if f.created_at > published_at:
            fail(
                ErrorCode.ASOF_VIOLATION,
                f"forecast {f.forecast_id} is dated {f.created_at.isoformat()}, after publication time {published_at.isoformat()}",
                "No future-dated evidence can be published.",
            )
        if f.is_expired(published_at):
            fail(
                ErrorCode.FORECAST_EXPIRED,
                f"forecast {f.forecast_id} expired at {f.expires_at.isoformat()} before publication at {published_at.isoformat()}",
                "Stale forecasts are never published.",
            )
        if not f.evidence_artifact_ids:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"forecast {f.forecast_id} has no evidence artifact",
                "Evidence artifacts are required.",
            )
        for artifact in (
            *f.evidence_artifact_ids,
            f.return_forecast.uncertainty_artifact_id,
        ):
            if not artifact_exists(artifact):
                fail(
                    ErrorCode.SCHEMA_MISMATCH,
                    f"forecast {f.forecast_id} cites artifact {artifact!r} that does not exist",
                    "Evidence and uncertainty artifacts must exist (spec 21.6).",
                )

    def _register(
        self,
        model: ModelVersionRecord,
        certificate: AdmissionCertificate,
        batch: ForecastBatch,
    ) -> None:
        if self.registry is None:
            return
        if self.registry.get("model_versions", model.model_id) is None:
            self.registry.register("model_versions", model)
        if (
            self.registry.get(
                "admission_certificates", certificate.certificate_id
            )
            is None
        ):
            self.registry.register("admission_certificates", certificate)
        if self.registry.get("forecast_publications", batch.batch_id) is None:
            self.registry.register("forecast_publications", batch)

    # ---------------------------------------------------------------- read
    def _readable_namespaces(self, ctx: RoleContext) -> Optional[list[str]]:
        if self.gateway.allowed(ctx, "publication.read_all"):
            return None
        self.gateway.authorize(ctx, "publication.read_own", ctx.namespace)
        return [ctx.namespace]

    def export_forecasts(self, ctx: RoleContext) -> list[Forecast]:
        """All published forecasts the caller may see (strategy: own namespace only)."""
        return [
            f
            for _ns, _batch, f in self.store.fetch(
                self._readable_namespaces(ctx)
            )
        ]

    def list_batches(self, ctx: RoleContext) -> list[dict[str, Any]]:
        return self.store.batches(self._readable_namespaces(ctx))

    def active_for_decision(
        self, ctx: RoleContext, decision_at: dt.datetime
    ) -> ActiveForecasts:
        """Forecasts a decision may use: published, formed, unexpired; newest per (security, strategy)."""
        decision_at = require_aware(decision_at, "decision_at")
        result = ActiveForecasts(decision_at=decision_at)

        def note(reason: str, f: Forecast) -> None:
            result.stale_counts[reason] = result.stale_counts.get(reason, 0) + 1
            if len(result.stale_examples) < 20:
                result.stale_examples.append(
                    {"forecast_id": f.forecast_id, "reason": reason}
                )

        best: dict[tuple[str, str], Forecast] = {}
        for _ns, _batch, f in self.store.fetch(self._readable_namespaces(ctx)):
            if (
                f.created_at > decision_at
                or f.information_cutoff > decision_at
                or f.earliest_formation_at > decision_at
            ):
                note("not_yet_published_or_formed", f)
                continue
            if f.is_expired(decision_at):
                note("expired", f)
                continue
            key = (f.security_id, f.strategy_id)
            rank = (f.information_cutoff, f.created_at, f.forecast_id)
            current = best.get(key)
            if current is None or rank > (
                current.information_cutoff,
                current.created_at,
                current.forecast_id,
            ):
                if current is not None:
                    note("superseded", current)
                best[key] = f
            else:
                note("superseded", f)
        result.active = [best[k] for k in sorted(best)]
        return result
