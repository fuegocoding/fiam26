"""Role context, service-side authorization and private namespace workspaces
(POD_FRAMEWORK_IMPLEMENTATION_SPEC.md 3.2-3.3, 6.3, 20.4).

The caller identity is a trusted ``RoleContext`` injected by the host, never an
argument an agent can choose. A strategy namespace can read and write only its
own private workspace; Research Integrity may read experiment records across
namespaces; Allocation/Risk/Reporting see *published* forecasts only. Denials
are typed (``PARTITION_ACCESS_DENIED``) and fail closed for unknown actions.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..contracts.base import require_aware
from ..contracts.errors import ErrorCode
from .common import atomic_write_json, digest, fail, read_json

ROLE_STRATEGY = "strategy"
ROLE_RESEARCH = "research"
ROLE_INTEGRITY = "integrity"
ROLE_ALLOCATION = "allocation"
ROLE_RISK = "risk"
ROLE_EXECUTION = "execution"
ROLE_REPORTING = "reporting"

ROLES = (
    ROLE_STRATEGY,
    ROLE_RESEARCH,
    ROLE_INTEGRITY,
    ROLE_ALLOCATION,
    ROLE_RISK,
    ROLE_EXECUTION,
    ROLE_REPORTING,
)

KNOWN_PARTITIONS = ("design", "check", "replay", "prospective")

_NAMESPACE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,100}\.json$")

# role -> actions it may invoke (anything else is denied)
PERMISSIONS: dict[str, frozenset[str]] = {
    ROLE_STRATEGY: frozenset(
        {
            "workspace.write_own",
            "workspace.read_own",
            "publication.publish",
            "publication.read_own",
            "research.submit",
            "runs.read",
            "notes.append_role",
        }
    ),
    ROLE_RESEARCH: frozenset({"runs.read", "research.read_all_jobs", "notes.append_role", "publication.read_all"}),
    ROLE_INTEGRITY: frozenset(
        {
            "workspace.read_any",
            "publication.read_all",
            "integrity.evaluate_protected",
            "certificate.issue",
            "research.read_all_jobs",
            "runs.read",
            "notes.append_role",
        }
    ),
    ROLE_ALLOCATION: frozenset({"publication.read_all", "allocation.build", "runs.read", "notes.append_role"}),
    ROLE_RISK: frozenset({"publication.read_all", "risk.review", "runs.read", "notes.append_role"}),
    ROLE_EXECUTION: frozenset({"execution.plan", "accounting.post", "runs.read", "notes.append_role"}),
    ROLE_REPORTING: frozenset({"publication.read_all", "reporting.build", "runs.read", "notes.append_role"}),
}

# actions where the target namespace must equal the caller's own namespace
OWN_NAMESPACE_ACTIONS = frozenset(
    {"workspace.write_own", "workspace.read_own", "publication.publish", "publication.read_own", "research.submit"}
)


@dataclass(frozen=True)
class RoleContext:
    """Trusted request context (spec 6.3). Built by the host, never by an agent."""

    caller_role: str
    namespace: str
    run_id: str
    research_asof: dt.datetime
    allowed_partitions: tuple[str, ...] = ("design",)
    data_policy_version: str = "policy_unversioned"
    request_id: str = "request_unset"

    def __post_init__(self) -> None:
        if self.caller_role not in ROLES:
            raise ValueError(f"unknown caller_role {self.caller_role!r}")
        if not _NAMESPACE.match(self.namespace or ""):
            raise ValueError(f"invalid namespace {self.namespace!r}")
        if self.caller_role != ROLE_STRATEGY and self.namespace != self.caller_role:
            raise ValueError("a non-strategy role runs in the namespace named after the role")
        require_aware(self.research_asof, "research_asof")
        unknown = [p for p in self.allowed_partitions if p not in KNOWN_PARTITIONS]
        if unknown:
            raise ValueError(f"unknown partitions: {unknown}")
        if not self.run_id:
            raise ValueError("run_id is required")

    def as_dict(self) -> dict[str, Any]:
        return {
            "caller_role": self.caller_role,
            "namespace": self.namespace,
            "run_id": self.run_id,
            "research_asof": self.research_asof.isoformat(),
            "allowed_partitions": list(self.allowed_partitions),
            "data_policy_version": self.data_policy_version,
            "request_id": self.request_id,
        }


class RoleGateway:
    """Service-side authorization. Unknown actions are denied, never defaulted open."""

    def __init__(self, permissions: Optional[dict[str, frozenset[str]]] = None):
        self._permissions = dict(permissions or PERMISSIONS)

    def allowed(self, ctx: RoleContext, action: str, namespace: Optional[str] = None) -> bool:
        actions = self._permissions.get(ctx.caller_role, frozenset())
        if action not in actions:
            return False
        if action in OWN_NAMESPACE_ACTIONS and namespace is not None and namespace != ctx.namespace:
            return False
        return True

    def authorize(self, ctx: RoleContext, action: str, namespace: Optional[str] = None) -> None:
        if not isinstance(ctx, RoleContext):
            fail(ErrorCode.PARTITION_ACCESS_DENIED, "a trusted RoleContext is required", "Identity is injected by the host.")
        known = set().union(*self._permissions.values()) if self._permissions else set()
        if action not in known:
            fail(ErrorCode.PARTITION_ACCESS_DENIED, f"unknown action {action!r}", "Unknown actions are denied.")
        if not self.allowed(ctx, action, namespace):
            fail(
                ErrorCode.PARTITION_ACCESS_DENIED,
                f"role {ctx.caller_role!r} in namespace {ctx.namespace!r} may not {action}"
                + (f" on namespace {namespace!r}" if namespace else ""),
                "Access is enforced service-side; supplying another pod id or namespace does not change the caller.",
            )

    def actions_for(self, role: str) -> frozenset[str]:
        return self._permissions.get(role, frozenset())


class WorkspaceStore:
    """Private per-namespace JSON staging (spec 3.3 private items, 20.4 private paths).

    Files are immutable once written (same content is an idempotent no-op; changed
    content fails) and carry a payload hash that is re-verified on every read.
    """

    def __init__(self, root: "str | Path", gateway: Optional[RoleGateway] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.gateway = gateway or RoleGateway()

    def _path(self, namespace: str, name: str) -> Path:
        if not _NAMESPACE.match(namespace or ""):
            fail(ErrorCode.SCHEMA_MISMATCH, f"invalid namespace {namespace!r}", "Namespaces are plain lower-case tokens.")
        if not _FILE_NAME.match(name or ""):
            fail(ErrorCode.SCHEMA_MISMATCH, f"invalid workspace file name {name!r}", "Names are plain *.json file names.")
        return self.root / namespace / name

    def write_json(self, ctx: RoleContext, namespace: str, name: str, obj: Any) -> dict[str, str]:
        self.gateway.authorize(ctx, "workspace.write_own", namespace)
        target = self._path(namespace, name)
        payload_hash = digest(obj)
        if target.exists():
            stored = read_json(target)
            if stored.get("payload_sha256") == payload_hash:
                return {"namespace": namespace, "name": name, "sha256": payload_hash, "path": str(target)}
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"workspace file {namespace}/{name} already exists with different content",
                "Staged artifacts are immutable; write a new name for a new version.",
            )
        atomic_write_json(target, {"namespace": namespace, "name": name, "payload": obj, "payload_sha256": payload_hash})
        return {"namespace": namespace, "name": name, "sha256": payload_hash, "path": str(target)}

    def read_json(self, ctx: RoleContext, namespace: str, name: str) -> Any:
        if ctx.caller_role == ROLE_STRATEGY:
            self.gateway.authorize(ctx, "workspace.read_own", namespace)
        else:
            self.gateway.authorize(ctx, "workspace.read_any", namespace)
        target = self._path(namespace, name)
        if not target.is_file():
            fail(ErrorCode.SCHEMA_MISMATCH, f"no workspace file {namespace}/{name}", "Nothing was staged under that name.")
        stored = read_json(target)
        if stored.get("namespace") != namespace or digest(stored.get("payload")) != stored.get("payload_sha256"):
            fail(ErrorCode.ACCOUNTING_RECONCILIATION_FAILED, f"workspace file {namespace}/{name} failed its hash",
                 "A damaged private artifact is not trusted.")
        return stored["payload"]

    def exists(self, namespace: str, name: str) -> bool:
        """Trusted internal existence check used by publication validation."""
        return self._path(namespace, name).is_file()

    def list_names(self, ctx: RoleContext, namespace: str) -> list[str]:
        if ctx.caller_role == ROLE_STRATEGY:
            self.gateway.authorize(ctx, "workspace.read_own", namespace)
        else:
            self.gateway.authorize(ctx, "workspace.read_any", namespace)
        folder = self.root / namespace
        if not folder.is_dir():
            return []
        return sorted(p.name for p in folder.iterdir() if p.is_file() and _FILE_NAME.match(p.name))
