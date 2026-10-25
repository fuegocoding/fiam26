"""Machine-readable admission-certificate scope and validation
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 14.3, 21; fail-closed controls).

An ``AdmissionCertificate`` historically carried only free-text restrictions
(``risk_restrictions``, ``capacity_restrictions``, ``expiry_policy``) which no
code can verify. ``CertificateScope`` adds the verifiable part: environment
(test / shadow / formation), validity window, allowed source ids and snapshots,
universe and namespace, and numeric exposure limits. ``validate_certificate``
is the single gate used by controlled replay and by anything else that deploys
a forecast.

Principles
* Boundaries fail closed: a restriction that is declared but cannot be
  evaluated with the supplied context is a rejection, not a pass.
* A certificate never *widens* code authority. Source eligibility stays with
  the source cards (``SourceEligibilityGate``); the scope can only narrow it.
* Legacy certificates (no scope) are accepted only by an explicit
  ``CertificatePolicy`` (``fixture_compat``) and are reported as
  ``legacy_unverified``; ``CertificatePolicy.production()`` rejects them.
* A test- or shadow-environment certificate can never authorize a formation
  run, regardless of policy.
* All timestamps are timezone-aware; naive values are rejected.

This module does not decide who may issue a certificate or declare an
environment: that is an authority decision outside the code.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

from pydantic import Field, field_validator, model_validator

from ..contracts.base import StrictModel, finite_decimal, require_aware
from ..contracts.errors import ErrorCode, PodPlatformError, ToolResponse

if TYPE_CHECKING:  # pragma: no cover - avoids an import cycle with promotion.py
    from ..contracts.forecasts import Forecast
    from .promotion import AdmissionCertificate


class CertificateEnvironment(str, enum.Enum):
    TEST = "test"
    SHADOW = "shadow"
    FORMATION = "formation"


_RANK = {
    CertificateEnvironment.TEST: 0,
    CertificateEnvironment.SHADOW: 1,
    CertificateEnvironment.FORMATION: 2,
}


class ScopeLimits(StrictModel):
    """Numeric limits the certificate holder agreed to; every field is optional."""

    max_gross_exposure: Optional[Decimal] = None  # sum |w|, fraction of NAV
    max_abs_net_exposure: Optional[Decimal] = None  # |sum w|
    max_name_weight: Optional[Decimal] = None  # max |w|
    max_nav_usd: Optional[Decimal] = None  # capacity: NAV the model is certified for
    max_position_count: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "ScopeLimits":
        for name in ("max_gross_exposure", "max_abs_net_exposure", "max_name_weight", "max_nav_usd"):
            value = getattr(self, name)
            if value is not None:
                finite_decimal(value, name)
                if value < 0 or (name != "max_abs_net_exposure" and value == 0):
                    raise ValueError(f"{name} must be positive")
        return self

    def declared(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


def _nonempty_unique(values: Optional[list[str]], name: str) -> Optional[list[str]]:
    if values is None:
        return None
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must be a non-empty list of non-empty strings when given")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not repeat entries")
    return values


class CertificateScope(StrictModel):
    """Verifiable restrictions of one certificate. Part of the sealed digest."""

    scope_version: str = "1"
    environment: CertificateEnvironment
    issued_at: Optional[dt.datetime] = None
    valid_from: Optional[dt.datetime] = None
    valid_until: Optional[dt.datetime] = None
    allowed_source_ids: Optional[list[str]] = None
    allowed_source_snapshots: Optional[dict[str, list[str]]] = None
    universe_ids: Optional[list[str]] = None
    namespaces: Optional[list[str]] = None
    limits: ScopeLimits = Field(default_factory=ScopeLimits)
    issuer: Optional[str] = None

    @field_validator("issued_at", "valid_from", "valid_until")
    @classmethod
    def _aware(cls, value: Optional[dt.datetime]) -> Optional[dt.datetime]:
        return None if value is None else require_aware(value, "scope timestamp")

    @field_validator("allowed_source_ids", "universe_ids", "namespaces")
    @classmethod
    def _lists(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        return _nonempty_unique(value, "scope list")

    @model_validator(mode="after")
    def _check(self) -> "CertificateScope":
        if self.valid_from and self.valid_until and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must lie after valid_from")
        if self.allowed_source_snapshots is not None:
            if not self.allowed_source_snapshots:
                raise ValueError("allowed_source_snapshots must not be empty when given")
            for source_id, snapshots in self.allowed_source_snapshots.items():
                _nonempty_unique(snapshots, f"snapshots of {source_id}")
                if not source_id.strip():
                    raise ValueError("snapshot pins need a source id")
                if self.allowed_source_ids is not None and source_id not in self.allowed_source_ids:
                    raise ValueError(f"snapshot pin for {source_id} is outside allowed_source_ids")
        return self


class CertificateRevocation(StrictModel):
    """Post-issuance fact; certificates are sealed, so revocation is a separate record."""

    certificate_id: str = Field(min_length=1)
    revoked_at: dt.datetime  # effective time: decisions at/after it are not authorized
    recorded_at: dt.datetime
    reason: str = Field(min_length=1)
    authority: str = Field(min_length=1)

    @field_validator("revoked_at", "recorded_at")
    @classmethod
    def _aware(cls, value: dt.datetime) -> dt.datetime:
        return require_aware(value, "revocation timestamp")


class RevocationLedger:
    """Append-only set of revocations (earliest effective time wins)."""

    def __init__(self, revocations: Iterable["CertificateRevocation | Mapping[str, Any]"] = ()):
        self._by_id: dict[str, list[CertificateRevocation]] = {}
        for item in revocations:
            self.add(item)

    def add(self, item: "CertificateRevocation | Mapping[str, Any]") -> CertificateRevocation:
        revocation = item if isinstance(item, CertificateRevocation) else CertificateRevocation.model_validate(item)
        self._by_id.setdefault(revocation.certificate_id, []).append(revocation)
        return revocation

    def effective(self, certificate_id: str, when: dt.datetime) -> Optional[CertificateRevocation]:
        when = require_aware(when, "when")
        hits = [r for r in self._by_id.get(certificate_id, []) if r.revoked_at <= when]
        return min(hits, key=lambda r: r.revoked_at) if hits else None

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_id.values())


@dataclass(frozen=True)
class CertificatePolicy:
    """What a caller is willing to accept. Use the named constructors."""

    name: str
    environment: CertificateEnvironment = CertificateEnvironment.FORMATION
    require_scope: bool = True
    require_expiry: bool = True
    require_source_scope: bool = True
    require_snapshot_pins: bool = True
    forbid_backdated: bool = True
    require_seal: bool = True
    allow_legacy_unscoped: bool = False
    legacy_approved_modes: frozenset = frozenset()

    @classmethod
    def production(cls) -> "CertificatePolicy":
        """Formation-grade: sealed, scoped, expiring, source- and snapshot-pinned, not backdated."""
        return cls(name="production")

    @classmethod
    def shadow(cls) -> "CertificatePolicy":
        return cls(name="shadow", environment=CertificateEnvironment.SHADOW)

    @classmethod
    def test(cls) -> "CertificatePolicy":
        """Any environment is acceptable; results are never formation evidence."""
        return cls(
            name="test",
            environment=CertificateEnvironment.TEST,
            require_scope=False,
            require_expiry=False,
            require_source_scope=False,
            require_snapshot_pins=False,
            forbid_backdated=False,
            allow_legacy_unscoped=True,
            legacy_approved_modes=frozenset({"controlled_replay"}),
        )

    @classmethod
    def fixture_compat(cls) -> "CertificatePolicy":
        """Controlled-replay fixtures: legacy free-text certificates whose approved
        mode is exactly ``controlled_replay`` are accepted, flagged ``legacy_unverified``.
        Scoped certificates are still held to their declared environment and limits.
        This policy never makes a fixture certificate authorize a real book."""
        return cls(
            name="fixture_compat",
            require_scope=False,
            require_expiry=False,
            require_source_scope=False,
            require_snapshot_pins=False,
            forbid_backdated=False,
            allow_legacy_unscoped=True,
            legacy_approved_modes=frozenset({"controlled_replay"}),
        )

    def replace(self, **changes: Any) -> "CertificatePolicy":
        return dataclasses.replace(self, **changes)


def resolve_policy(value: "None | str | CertificatePolicy", default: str = "fixture_compat") -> CertificatePolicy:
    if isinstance(value, CertificatePolicy):
        return value
    name = value or default
    factory = {
        "production": CertificatePolicy.production,
        "shadow": CertificatePolicy.shadow,
        "test": CertificatePolicy.test,
        "fixture_compat": CertificatePolicy.fixture_compat,
    }.get(name)
    if factory is None:
        raise PodPlatformError(
            ToolResponse.failure(
                summary=f"unknown certificate policy {name!r}",
                code=ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                stop_condition="Choose production, shadow, test or fixture_compat explicitly.",
            )
        )
    return factory()


@dataclass(frozen=True)
class CertificateCheck:
    certificate_id: str
    policy: str
    environment: Optional[str]  # None for a legacy (unscoped) certificate
    scoped: bool
    legacy_unverified: bool
    warnings: tuple = ()

    @property
    def authoritative(self) -> bool:
        """True only for a scoped formation-environment certificate that passed a policy
        requiring scope; never true for legacy or test/shadow certificates."""
        return self.scoped and not self.legacy_unverified and self.environment == CertificateEnvironment.FORMATION.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "policy": self.policy,
            "environment": self.environment,
            "scoped": self.scoped,
            "legacy_unverified": self.legacy_unverified,
            "authoritative": self.authoritative,
            "warnings": list(self.warnings),
        }


def _fail(summary: str, stop: str) -> None:
    raise PodPlatformError(
        ToolResponse.failure(summary=summary, code=ErrorCode.ADMISSION_CERTIFICATE_INVALID, stop_condition=stop)
    )


def _to_decimal(value: Any, name: str) -> Decimal:
    try:
        out = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception:
        _fail(f"exposure {name} is not a number", "Exposure checks need measured numeric values.")
    if not out.is_finite():
        _fail(f"exposure {name} is not finite", "Exposure checks need measured numeric values.")
    return out


EXPOSURE_KEYS = ("gross", "net", "max_name_weight", "nav_usd", "position_count")


def check_scope_limits(certificate: "AdmissionCertificate", exposures: Mapping[str, Any]) -> None:
    """Reject when measured exposures break a declared limit.

    ``exposures`` keys: gross, net (signed), max_name_weight, nav_usd,
    position_count. A declared limit whose measurement is absent is a
    rejection (fail closed)."""
    scope = certificate.scope
    if scope is None:
        return
    cid = certificate.certificate_id
    pairs = (
        ("max_gross_exposure", "gross", lambda v: abs(v)),
        ("max_abs_net_exposure", "net", lambda v: abs(v)),
        ("max_name_weight", "max_name_weight", lambda v: abs(v)),
        ("max_nav_usd", "nav_usd", lambda v: v),
        ("max_position_count", "position_count", lambda v: v),
    )
    for limit_name, key, norm in pairs:
        limit = getattr(scope.limits, limit_name)
        if limit is None:
            continue
        if key not in exposures or exposures[key] is None:
            _fail(
                f"certificate {cid} declares {limit_name} but {key} was not measured",
                "A declared certificate limit must be checked against a measured exposure.",
            )
        measured = norm(_to_decimal(exposures[key], key))
        if measured > Decimal(str(limit)):
            _fail(
                f"certificate {cid} limit {limit_name}={limit} exceeded by measured {key}={measured}",
                "Positions beyond the certified limits are not authorized.",
            )


def validate_certificate(
    forecast: "Forecast",
    certificate: "Optional[AdmissionCertificate]",
    *,
    decision_at: dt.datetime,
    source_ids: Optional[Iterable[str]] = None,
    source_snapshots: Optional[Mapping[str, str]] = None,
    universe: Optional[str] = None,
    namespace: Optional[str] = None,
    policy: "None | str | CertificatePolicy" = None,
    environment: Optional[CertificateEnvironment] = None,
    revocations: Optional[RevocationLedger] = None,
    exposures: Optional[Mapping[str, Any]] = None,
) -> CertificateCheck:
    """Raise ``PodPlatformError(ADMISSION_CERTIFICATE_INVALID)`` unless ``certificate``
    authorizes ``forecast`` at ``decision_at`` under ``policy`` (default: production).

    ``source_ids`` are extra sources consumed beyond ``forecast.source_family_ids``;
    ``source_snapshots`` maps source id -> the snapshot actually used.
    ``environment`` overrides the environment level the policy requires (a
    research-only run passes TEST). Exposure limits are evaluated only when
    ``exposures`` is given (book level: see ``check_scope_limits``); otherwise a
    warning records that they were not evaluated.
    """
    pol = resolve_policy(policy, default="production")
    try:
        decision_at = require_aware(decision_at, "decision_at")
    except (ValueError, AttributeError):
        _fail("decision_at must be timezone-aware", "All clocks are timezone-aware.")
    fid = forecast.forecast_id
    if certificate is None:
        _fail(
            f"forecast {fid} cites unknown certificate {forecast.admission_certificate_id}",
            "Every deployed forecast needs a valid certificate; none is fabricated.",
        )
    cert = certificate
    cid = cert.certificate_id
    if cid != forecast.admission_certificate_id:
        _fail(f"certificate {cid} is not the one cited by {fid}", "The cited certificate must be the one validated.")
    if cert.model_id != forecast.model_id or decision_at < cert.effective_start:
        _fail(
            f"certificate {cid} does not cover {fid} at {decision_at.isoformat()}",
            "Certificate model and effective start must match the forecast.",
        )
    if cert.created_at > decision_at:
        _fail(
            f"certificate {cid} was created {cert.created_at.isoformat()}, after the decision {decision_at.isoformat()}",
            "A certificate issued after the decision (even if back-dated via effective_start) cannot authorize it.",
        )
    if cert.target_id != forecast.target.target_id or cert.horizon_trading_days != forecast.return_forecast.horizon_trading_days:
        _fail(
            f"certificate {cid} covers target {cert.target_id!r}/{cert.horizon_trading_days}d, not "
            f"{forecast.target.target_id!r}/{forecast.return_forecast.horizon_trading_days}d of {fid}",
            "Certificate target and horizon must match the deployed forecast.",
        )
    if pol.require_seal and not cert.verify_seal():
        _fail(f"certificate {cid} is unsealed or its seal fails", "Only sealed certificates are accepted here.")
    if pol.forbid_backdated:
        if cert.effective_start < cert.created_at:
            _fail(
                f"certificate {cid} is back-dated: effective_start {cert.effective_start.isoformat()} precedes issuance "
                f"{cert.created_at.isoformat()}",
                "A certificate cannot authorize decisions made before it existed.",
            )
        if forecast.created_at < cert.created_at:
            _fail(
                f"forecast {fid} was published before certificate {cid} was issued",
                "A forecast cannot cite a certificate that did not yet exist.",
            )
    ledger = revocations
    if ledger is not None:
        hit = ledger.effective(cid, decision_at)
        if hit is not None:
            _fail(
                f"certificate {cid} was revoked effective {hit.revoked_at.isoformat()} ({hit.reason})",
                "A revoked certificate authorizes no decision at or after its revocation time.",
            )
    scope = getattr(cert, "scope", None)
    warnings: list[str] = []
    if scope is None:
        if pol.require_scope or not pol.allow_legacy_unscoped:
            _fail(
                f"certificate {cid} carries only free-text restrictions (risk_restrictions, capacity_restrictions, "
                "expiry_policy) that cannot be verified; a machine-readable scope is required",
                "Unverified restrictions fail closed outside an explicit fixture/test policy.",
            )
        if cert.approved_mode not in pol.legacy_approved_modes:
            _fail(
                f"legacy certificate {cid} approved_mode {cert.approved_mode!r} is not accepted by policy {pol.name}",
                "Legacy certificates are only accepted for explicitly allowed modes.",
            )
        warnings.append("legacy_unverified: free-text restrictions are not machine-checked")
        if exposures is not None or source_ids is not None or universe is not None or namespace is not None:
            warnings.append("legacy certificate has no scope: source/universe/namespace/limits not evaluated")
        return CertificateCheck(cid, pol.name, None, False, True, tuple(warnings))

    required = environment or pol.environment
    if _RANK[scope.environment] < _RANK[required]:
        _fail(
            f"certificate {cid} is a {scope.environment.value} certificate and cannot authorize a "
            f"{required.value} run",
            "Test and shadow certificates never authorize formation; issue a formation-environment certificate.",
        )
    start = scope.valid_from or cert.effective_start
    if decision_at < start:
        _fail(f"certificate {cid} is not yet valid at {decision_at.isoformat()}", "Decision precedes certificate validity.")
    if scope.valid_until is None:
        if pol.require_expiry:
            _fail(f"certificate {cid} has no valid_until", "Production certificates must carry a machine-readable expiry.")
        warnings.append("no valid_until: certificate does not expire")
    elif decision_at >= scope.valid_until:
        _fail(f"certificate {cid} expired at {scope.valid_until.isoformat()}", "An expired certificate authorizes nothing.")
    if scope.issued_at is not None and scope.issued_at > decision_at:
        _fail(f"certificate {cid} was issued after the decision", "A certificate issued after the decision cannot authorize it.")

    used = set(forecast.source_family_ids) | set(source_ids or ())
    if scope.allowed_source_ids is None:
        if pol.require_source_scope:
            _fail(f"certificate {cid} does not restrict source ids", "Production certificates must pin allowed_source_ids.")
        warnings.append("no allowed_source_ids: certificate does not restrict sources")
    else:
        extra = sorted(used - set(scope.allowed_source_ids))
        if extra:
            _fail(
                f"certificate {cid} does not authorize source(s) {', '.join(extra)}",
                "A certificate can only narrow source authority, never widen it.",
            )
    pins = scope.allowed_source_snapshots
    if pins is None:
        if pol.require_snapshot_pins:
            _fail(f"certificate {cid} does not pin source snapshots", "Production certificates must pin allowed_source_snapshots.")
    else:
        supplied = dict(source_snapshots or {})
        for sid in sorted(used):
            if sid not in pins:
                if pol.require_snapshot_pins:
                    _fail(f"certificate {cid} has no snapshot pin for source {sid}", "Every consumed source needs a pinned snapshot.")
                continue
            got = supplied.get(sid)
            if got is None:
                _fail(f"snapshot of source {sid} was not supplied for certificate {cid}", "A pinned snapshot must be checked against the one used.")
            if got not in pins[sid]:
                _fail(f"source {sid} snapshot {got!r} is not pinned by certificate {cid}", "Only the certified snapshots may be used.")
    if scope.universe_ids is not None:
        if universe is None or universe not in scope.universe_ids:
            _fail(f"universe {universe!r} is outside certificate {cid} scope", "Certificates are universe-specific.")
    elif universe is not None and universe != cert.approved_universe and pol.require_scope:
        _fail(f"universe {universe!r} differs from approved_universe {cert.approved_universe!r} of {cid}", "Certificates are universe-specific.")
    if scope.namespaces is not None:
        if namespace is None or namespace not in scope.namespaces:
            _fail(f"namespace {namespace!r} is outside certificate {cid} scope", "Certificates are namespace-specific.")
    if exposures is not None:
        check_scope_limits(cert, exposures)
    elif scope.limits.declared():
        warnings.append("limits_not_evaluated: supply exposures or call check_scope_limits at book level")
    return CertificateCheck(cid, pol.name, scope.environment.value, True, False, tuple(warnings))
