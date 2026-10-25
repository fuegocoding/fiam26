"""Registered design candidate -> independent protected check -> certificate / publication bridge.

Pipeline (all authority is deterministic code; no model output and no caller boolean can admit anything):

1. ``ChallengeBridge.freeze_candidate`` re-derives the candidate from the *registered ledger report* of a built-in
   executor run (report hash, code hash, recipe digest, dataset/source/data hashes, persisted frozen design-trained
   model and exact design predictions) and requires an accepted measured design result. The candidate, its scoring
   policy and multiplicity are sealed in an append-only store BEFORE any check row is read.
2. A trusted host holds a ``CheckAuthority`` (integrity-owned HMAC key; never given to pods) and issues a
   ``CheckAuthorization`` bound to the candidate digest, family, dataset, check window and a finite family budget.
3. ``ChallengeBridge.run_challenge`` verifies provenance again, reserves a family check access, opens the check window
   (rows after the design window), scores the frozen model WITHOUT refitting, and hands design + check observations to
   the existing ``IntegrityService.evaluate`` (baseline, timing placebo, stability, multiplicity, certificate).
4. ``ChallengeBridge.publish_forecasts`` publishes frozen-model forecasts only for a stored ADMITTED decision, through
   ``PublicationService`` and its certificate checks.

Real (non-fixture) candidates whose recipe declares proxy / unsupported targets stop at research SHADOW: their check
rows are never opened and no certificate exists. Fixture data is labelled and only receives fixture certificates.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import threading
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import Field

from ..application.common import fail
from ..application.integrity import (
    CandidateRecord,
    EvidenceSeal,
    IntegrityDecision,
    IntegrityPolicy,
    IntegrityService,
    Outcome,
    classify_research_candidate,
)
from ..application.namespaces import RoleContext, RoleGateway
from ..application.publication import (
    SCOPE_FIXTURE,
    SCOPE_RESEARCH,
    ModelVersionRecord,
    PublicationService,
)
from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.errors import ErrorCode
from ..contracts.forecasts import Forecast, ReturnForecast, TargetPrediction
from ..contracts.sources import SourceCard
from ..research.evaluation import ForecastOutcome
from .bridge_economics import (
    CalibrationError,
    calibrate_return_bridge,
    tail_membership,
)
from .builtins.catalog import RECIPES, VariantConfig
from .builtins.engine import required_columns, score_frozen, target_resolve_at
from .builtins.executors import (
    REPORT_SCHEMA,
    BuiltinContext,
    _implementation_hash,
)
from .builtins.loader import file_sha256, load_panel
from .builtins.manifest import DESIGN, ApprovedDataset, PartitionSpec
from .builtins.timeutil import (
    end_of_day,
    month_end_date,
    month_end_known_at,
    month_index,
)

BRIDGE_VERSION = "challenge_bridge.v1"
SYNTHETIC_DATASET_ID = (
    "SYNTHETIC_FIXTURE"  # IntegrityService's fixture-scope source gate
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS frozen (candidate_id TEXT PRIMARY KEY, digest TEXT NOT NULL, payload_json TEXT NOT NULL, frozen_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS family_access (family TEXT NOT NULL, candidate_id TEXT NOT NULL, authorization_id TEXT NOT NULL,
    PRIMARY KEY (family, candidate_id));
CREATE TABLE IF NOT EXISTS runs (candidate_id TEXT PRIMARY KEY, state TEXT NOT NULL CHECK (state IN ('started','completed','failed')),
    authorization_id TEXT NOT NULL, decision_json TEXT, hidden_json TEXT, failure TEXT);
CREATE TRIGGER IF NOT EXISTS frozen_no_update BEFORE UPDATE ON frozen BEGIN SELECT RAISE(ABORT, 'frozen candidates are immutable'); END;
CREATE TRIGGER IF NOT EXISTS frozen_no_delete BEFORE DELETE ON frozen BEGIN SELECT RAISE(ABORT, 'frozen candidates are immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS access_no_update BEFORE UPDATE ON family_access BEGIN SELECT RAISE(ABORT, 'check accesses are append-only'); END;
CREATE TRIGGER IF NOT EXISTS access_no_delete BEFORE DELETE ON family_access BEGIN SELECT RAISE(ABORT, 'check accesses are append-only'); END;
CREATE TRIGGER IF NOT EXISTS runs_final BEFORE UPDATE ON runs WHEN OLD.state != 'started' BEGIN SELECT RAISE(ABORT, 'finished runs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT, 'runs are never deleted'); END;
"""


class FrozenCandidate(StrictModel):
    """Everything about a design candidate that may not change between design and check."""

    candidate_id: str
    charter: str
    namespace: str
    mechanism_family: str
    registration_id: (
        str  # built-in executor registration (namespace:draft_entry_id)
    )
    variant_name: str
    spec_hash: str
    report_hash: str
    artifact_sha256: str
    frozen_model_hash: str
    predictions_hash: str
    implementation_hash: str
    recipe_id: str
    recipe_digest: str
    variant_params: dict[str, Any]
    dataset_id: str
    descriptor_hash: str
    data_file_sha256: str
    source_id: str
    source_snapshot_id: str
    source_card_hash: str
    fixture: bool
    proxy_labels: list[str]
    bridge_sign: int
    target_id: str
    horizon_trading_days: int = Field(ge=1)
    design_window: list[str]
    scope: str
    multiplicity: int = Field(ge=1)
    policy_id: str
    policy_digest: str
    registered_at: dt.datetime
    frozen_at: dt.datetime
    design_metrics: dict[str, float]

    @property
    def model_id(self) -> str:
        return f"model_{self.candidate_id[5:]}"

    @property
    def model_hash(self) -> str:
        """Binds frozen model + recipe + code + data + spec: any amendment yields a different hash."""
        return canonical_digest(
            {
                "model": self.frozen_model_hash,
                "recipe": self.recipe_digest,
                "code": self.implementation_hash,
                "data": self.data_file_sha256,
                "spec": self.spec_hash,
            }
        )

    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


@dataclass(frozen=True)
class CheckAuthorization:
    """Integrity-issued permission to open ONE candidate's check window. Not a pod/model argument; sealed by the authority."""

    authorization_id: str
    candidate_digest: str
    family: str
    dataset_id: str
    scope: str
    check_start: dt.date
    check_end: dt.date
    issued_at: dt.datetime
    expires_at: dt.datetime
    max_family_accesses: int
    seal: str

    def body(self) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        raw.pop("seal")
        return {
            k: (v.isoformat() if isinstance(v, (dt.date, dt.datetime)) else v)
            for k, v in raw.items()
        }


class CheckAuthority:
    """Integrity-owned issuer. The HMAC key stays inside this object; hold it only in the trusted host process."""

    def __init__(
        self, key: Optional[bytes] = None, gateway: Optional[RoleGateway] = None
    ):
        self._key = key if key is not None else secrets.token_bytes(32)
        self._gateway = gateway or RoleGateway()

    def __repr__(self) -> str:
        return "CheckAuthority(<key hidden>)"

    def __reduce__(self):  # the key never travels through pickles or logs
        raise TypeError("CheckAuthority cannot be serialized")

    @classmethod
    def load_or_init(
        cls, key_path: "str | Path", gateway: Optional[RoleGateway] = None
    ) -> "CheckAuthority":
        """Restart-safe host key: create a private 0600 state file once (atomic), reload it on every later start.

        The file lives in a directory with a ``.gitignore`` of ``*``; group/other-accessible, foreign-owned, symlinked or
        malformed key files are refused. The key is never returned or logged, so a resumed process can verify earlier
        authorizations (quotas live in the bridge store, not in the key) but nobody can read the key through this API.
        """
        path = Path(key_path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        ignore = path.parent / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")
        if not os.path.lexists(path):
            tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="ascii") as handle:
                    handle.write(secrets.token_hex(32) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(
                        tmp, path
                    )  # atomic create-if-absent; a concurrent initializer wins cleanly
                except FileExistsError:
                    pass
            finally:
                os.unlink(tmp)
        info = os.lstat(path)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_mode & 0o077
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())
        ):
            fail(
                ErrorCode.PARTITION_ACCESS_DENIED,
                "authority key file must be a regular file owned by this user with mode 0600",
                "Refusing an unsafe key file; fix its permissions or provision a new host key.",
            )
        text = path.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", text):
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                "authority key file is malformed",
                "Refusing to guess a key; provision a new host key.",
            )
        return cls(bytes.fromhex(text), gateway)

    def evidence_seal(self) -> EvidenceSeal:
        """Domain-separated seal for evidence bundles; give it to the producer (bridge) and to the IntegrityService."""
        return EvidenceSeal(
            hmac.new(self._key, b"evidence-seal-v1", hashlib.sha256).digest(),
            self._gateway,
        )

    def _seal(self, body: dict[str, Any]) -> str:
        return hmac.new(
            self._key, canonical_digest(body).encode(), hashlib.sha256
        ).hexdigest()

    def issue(
        self,
        ctx: RoleContext,
        frozen: FrozenCandidate,
        *,
        check_start: dt.date,
        check_end: dt.date,
        issued_at: dt.datetime,
        expires_at: dt.datetime,
        max_family_accesses: int,
    ) -> CheckAuthorization:
        self._gateway.authorize(ctx, "integrity.evaluate_protected")
        if max_family_accesses < 1:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                "max_family_accesses must be at least 1",
                "Check access is finite but not empty.",
            )
        draft = CheckAuthorization(
            "",
            frozen.digest(),
            frozen.mechanism_family,
            frozen.dataset_id,
            frozen.scope,
            check_start,
            check_end,
            require_aware(issued_at, "issued_at"),
            require_aware(expires_at, "expires_at"),
            int(max_family_accesses),
            "",
        )
        body = draft.body()
        body["authorization_id"] = "chk_" + canonical_digest(body)[:20]
        return dataclasses.replace(
            draft,
            authorization_id=body["authorization_id"],
            seal=self._seal(body),
        )

    def verify(self, auth: Any) -> bool:
        if not isinstance(auth, CheckAuthorization):
            return False
        return hmac.compare_digest(auth.seal, self._seal(auth.body()))


@dataclass(frozen=True)
class ChallengeResult:
    decision: IntegrityDecision
    hidden: dict[str, Any]
    check_opened: bool


class ChallengeBridge:
    def __init__(
        self,
        state_dir: "str | Path",
        *,
        authority: CheckAuthority,
        gateway: Optional[RoleGateway] = None,
    ):
        root = Path(state_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.authority = authority
        self.gateway = gateway or RoleGateway()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(
            str(root / "challenge_bridge.sqlite"),
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    # ------------------------------------------------------------------ store
    def _tx(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                out = fn(self._db)
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")
            return out

    def event(self, candidate_id: str, kind: str, detail: str) -> None:
        self._tx(
            lambda c: c.execute(
                "INSERT INTO events (candidate_id, kind, detail) VALUES (?,?,?)",
                (candidate_id, kind, detail[:800]),
            )
        )

    def events(
        self, candidate_id: Optional[str] = None
    ) -> list[tuple[str, str, str]]:
        sql, args = "SELECT candidate_id, kind, detail FROM events", ()
        if candidate_id:
            sql, args = sql + " WHERE candidate_id=?", (candidate_id,)
        return [
            tuple(r)
            for r in self._db.execute(sql + " ORDER BY seq", args).fetchall()
        ]

    def family_accesses(self, family: str) -> int:
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM family_access WHERE family=?", (family,)
            ).fetchone()[0]
        )

    def frozen(self, candidate_id: str) -> FrozenCandidate:
        row = self._db.execute(
            "SELECT digest, payload_json FROM frozen WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"candidate {candidate_id} is not frozen",
                "Freeze the candidate before any check access.",
            )
        frozen = FrozenCandidate.model_validate_json(row[1])
        if frozen.digest() != row[0]:
            fail(
                ErrorCode.ACCOUNTING_RECONCILIATION_FAILED,
                f"frozen candidate {candidate_id} failed its digest",
                "Damaged state is not trusted.",
            )
        return frozen

    def run_state(self, candidate_id: str) -> Optional[dict[str, Any]]:
        row = self._db.execute(
            "SELECT state, authorization_id, decision_json, hidden_json, failure FROM runs WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "state": row[0],
            "authorization_id": row[1],
            "decision": json.loads(row[2]) if row[2] else None,
            "hidden": json.loads(row[3]) if row[3] else None,
            "failure": row[4],
        }

    def _deny(
        self,
        candidate_id: str,
        code: ErrorCode,
        summary: str,
        stop: str = "Challenge denied; nothing was opened or admitted.",
    ) -> None:
        self.event(candidate_id, "denied", summary)
        fail(code, summary, stop)

    # ----------------------------------------------------------- provenance
    def _provenance(
        self, context: BuiltinContext, report: dict, key: str
    ) -> tuple[ApprovedDataset, dict]:
        """Re-derive every hash from the live code, recipe, manifest and data file; any drift refuses."""

        def bad(msg: str) -> None:
            self._deny(
                key, ErrorCode.SCHEMA_MISMATCH, f"provenance mismatch: {msg}"
            )

        if report.get("schema_version") != REPORT_SCHEMA:
            bad("unknown report schema")
        body = {k: v for k, v in report.items() if k != "report_hash"}
        if canonical_digest(body) != report.get("report_hash"):
            bad("registered report fails its report hash")
        spec = report["spec"]
        if canonical_digest(spec) != report["spec_hash"]:
            bad("report spec does not match spec hash")
        if spec["implementation_hash"] != _implementation_hash():
            bad("reviewed implementation changed since the design run")
        recipe = RECIPES.get(report["recipe_id"])
        if recipe is None or recipe.digest() != spec["recipe_digest"]:
            bad("recipe changed since the design run")
        dataset = context.manifest.dataset(report["dataset_id"])
        if (
            dataset.descriptor_hash != spec["dataset_descriptor_hash"]
            or dataset.source_card_hash != spec["source_card_hash"]
            or file_sha256(dataset.path) != spec["data_file_sha256"]
        ):
            bad(
                "dataset descriptor, source card or data file changed since the design run"
            )
        art = report.get("candidate_artifact")
        path = context.candidate_path(report["spec_hash"])
        if not art or path is None or not path.is_file():
            bad(
                "no persisted candidate artifact (run the executor with a persistent state_dir)"
            )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != art["artifact_sha256"]:
            bad("candidate artifact bytes do not match the registered report")
        artifact = json.loads(raw)
        if (
            artifact["spec_hash"] != report["spec_hash"]
            or canonical_digest(artifact["frozen_model"]) != art["model_hash"]
            or canonical_digest(artifact["predictions"])
            != art["predictions_hash"]
        ):
            bad("candidate artifact contents are inconsistent")
        return dataset, artifact

    # ---------------------------------------------------------------- freeze
    def freeze_candidate(
        self,
        ctx: RoleContext,
        context: BuiltinContext,
        registration_id: str,
        variant_name: str,
        *,
        policy: IntegrityPolicy,
        registered_at: dt.datetime,
        multiplicity: int,
    ) -> FrozenCandidate:
        """Seal one measured design candidate and the scoring policy. Opens no check row."""
        self.gateway.authorize(ctx, "integrity.evaluate_protected")
        registered_at = require_aware(registered_at, "registered_at")
        key = f"reg:{registration_id}/{variant_name}"
        slot = context.ledger.slot(registration_id, variant_name)
        report = context.report(registration_id, variant_name)
        if (
            slot is None
            or slot["state"] != "succeeded"
            or report is None
            or slot["spec_hash"] != report.get("spec_hash")
        ):
            self._deny(
                key,
                ErrorCode.SCHEMA_MISMATCH,
                "no succeeded registered design result for this registration and variant",
            )
        if registered_at > policy.frozen_at:
            self._deny(
                key,
                ErrorCode.ASOF_VIOLATION,
                "scoring policy must be frozen at or after registration and before check access",
            )
        if multiplicity < context.ledger.variants_used(registration_id):
            self._deny(
                key,
                ErrorCode.SCHEMA_MISMATCH,
                "multiplicity must count every scored variant of the registration",
            )
        dataset, artifact = self._provenance(context, report, key)
        recipe = RECIPES[report["recipe_id"]]
        metrics = {k: float(v) for k, v in report["metrics"].items()}
        verdict = classify_research_candidate(
            {
                "candidate_id": key,
                "charter": report["charter"],
                "metrics": metrics,
                "status": "evaluated",
                "source_ids": [report["source_id"]],
            },
            policy,
            {report["source_id"]: dataset.source_card},
        )
        if (
            verdict.outcome != Outcome.SHADOW
            or metrics.get("belief_ic_candidate", 0.0) < policy.min_mean_ic
            or metrics.get("coverage_eval_months", 0.0)
            < policy.min_dates_design
        ):
            self._deny(
                key,
                ErrorCode.SCHEMA_MISMATCH,
                f"design result not accepted ({verdict.outcome.value}): "
                + "; ".join(verdict.reasons[2:])
                or "thin coverage or weak belief IC",
            )
        spec_hash = report["spec_hash"]
        frozen = FrozenCandidate(
            candidate_id=f"cand_{spec_hash[:16]}",
            charter=report["charter"],
            namespace=registration_id.split(":")[0],
            mechanism_family=recipe.mechanism_family,
            registration_id=registration_id,
            variant_name=variant_name,
            spec_hash=spec_hash,
            report_hash=report["report_hash"],
            artifact_sha256=report["candidate_artifact"]["artifact_sha256"],
            frozen_model_hash=report["candidate_artifact"]["model_hash"],
            predictions_hash=report["candidate_artifact"]["predictions_hash"],
            implementation_hash=report["spec"]["implementation_hash"],
            recipe_id=recipe.recipe_id,
            recipe_digest=recipe.digest(),
            variant_params=dict(report["params"]),
            dataset_id=dataset.dataset_id,
            descriptor_hash=dataset.descriptor_hash,
            data_file_sha256=report["spec"]["data_file_sha256"],
            source_id=dataset.spec.source_id,
            source_snapshot_id=dataset.spec.source_snapshot_id,
            source_card_hash=dataset.source_card_hash,
            fixture=dataset.fixture,
            proxy_labels=list(recipe.proxy_labels),
            bridge_sign=recipe.bridge_sign,
            target_id=recipe.target.target_id,
            horizon_trading_days=21 * dataset.spec.label.resolve_months,
            design_window=[d.isoformat() for d in dataset.design_window],
            scope=SCOPE_FIXTURE if dataset.fixture else SCOPE_RESEARCH,
            multiplicity=multiplicity,
            policy_id=policy.policy_id,
            policy_digest=canonical_digest(policy.model_dump(mode="json")),
            registered_at=registered_at,
            frozen_at=policy.frozen_at,
            design_metrics=metrics,
        )
        digest = frozen.digest()

        def put(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                "SELECT digest FROM frozen WHERE candidate_id=?",
                (frozen.candidate_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO frozen VALUES (?,?,?,?)",
                    (
                        frozen.candidate_id,
                        digest,
                        frozen.model_dump_json(),
                        frozen.frozen_at.isoformat(),
                    ),
                )
            elif row[0] != digest:
                raise ValueError(
                    "candidate already frozen with different content (amendments need a new registration/variant)"
                )

        try:
            self._tx(put)
        except ValueError as exc:
            self._deny(frozen.candidate_id, ErrorCode.SCHEMA_MISMATCH, str(exc))
        return frozen

    # ------------------------------------------------------------- challenge
    def integrity_source_card(
        self, context: BuiltinContext, frozen: FrozenCandidate
    ) -> SourceCard:
        """Return the original card unchanged; fixture status travels in sealed provenance."""
        return context.manifest.dataset(frozen.dataset_id).source_card

    def candidate_record(
        self, context: BuiltinContext, frozen: FrozenCandidate
    ) -> CandidateRecord:
        dataset = context.manifest.dataset(frozen.dataset_id)
        recipe = RECIPES[frozen.recipe_id]
        variant = VariantConfig(**frozen.variant_params)
        h = frozen.candidate_id[5:]
        return CandidateRecord(
            candidate_id=frozen.candidate_id,
            charter=frozen.charter,
            namespace=frozen.namespace,
            hypothesis_id=f"hyp_{h}",
            experiment_spec_id=f"exp_{h}",
            mechanism_family=frozen.mechanism_family,
            model_id=frozen.model_id,
            model_hash=frozen.model_hash,
            target_id=frozen.target_id,
            horizon_trading_days=frozen.horizon_trading_days,
            source_ids=[frozen.source_id],
            question=f"Does frozen recipe {recipe.recipe_id} improve on its declared baseline in an unseen check window?",
            mechanism=recipe.summary,
            declared_sign=1,
            approved_universe=dataset.spec.universe_id
            or f"{frozen.dataset_id}_universe",
            feature_hashes={
                f.name: canonical_digest({"spec": repr(f)})
                for f in recipe.features(variant.feature_set)
            },
            data_hashes={frozen.source_id: frozen.data_file_sha256},
            evidence_mode="synthetic_fixture"
            if frozen.fixture
            else "design_partition_research",
        )

    def _complete(
        self,
        candidate_id: str,
        state: str,
        decision: Optional[IntegrityDecision],
        hidden: Optional[dict],
        failure: Optional[str],
    ) -> None:
        self._tx(
            lambda c: c.execute(
                "UPDATE runs SET state=?, decision_json=?, hidden_json=?, failure=? WHERE candidate_id=?",
                (
                    state,
                    json.dumps(decision.as_dict(), sort_keys=True)
                    if decision
                    else None,
                    json.dumps(hidden, sort_keys=True, allow_nan=False)
                    if hidden is not None
                    else None,
                    failure,
                    candidate_id,
                ),
            )
        )

    def _check_window(
        self,
        candidate_id: str,
        frozen: FrozenCandidate,
        auth: Any,
        asof: dt.datetime,
        ctx: RoleContext,
    ) -> None:
        d = lambda code, msg: self._deny(candidate_id, code, msg)  # noqa: E731
        if "check" not in ctx.allowed_partitions:
            d(
                ErrorCode.PARTITION_ACCESS_DENIED,
                "the trusted context does not include the check partition",
            )
        if not self.authority.verify(auth):
            d(
                ErrorCode.PARTITION_ACCESS_DENIED,
                "check authorization is missing, forged or tampered; only the integrity authority can approve a check",
            )
        if (
            auth.candidate_digest != frozen.digest()
            or auth.family != frozen.mechanism_family
            or auth.dataset_id != frozen.dataset_id
            or auth.scope != frozen.scope
        ):
            d(
                ErrorCode.PARTITION_ACCESS_DENIED,
                "check authorization is not bound to this exact frozen candidate (amended candidates cannot reuse checks)",
            )
        if (
            auth.issued_at < frozen.frozen_at
            or not auth.issued_at <= asof <= auth.expires_at
        ):
            d(
                ErrorCode.ASOF_VIOLATION,
                "check authorization must be issued after the freeze and be unexpired at the evaluation clock",
            )
        if asof < frozen.frozen_at:
            d(
                ErrorCode.ASOF_VIOLATION,
                "evaluation clock precedes the frozen scoring policy",
            )
        design_end = dt.date.fromisoformat(frozen.design_window[1])
        if auth.check_start <= design_end or auth.check_end <= auth.check_start:
            d(
                ErrorCode.ASOF_VIOLATION,
                "check window must start after the design window ends (chronology / no overlap)",
            )
        if end_of_day(auth.check_end) > asof:
            d(
                ErrorCode.ASOF_VIOLATION,
                "check labels are not mature at the evaluation clock",
            )

    def run_challenge(
        self,
        ctx: RoleContext,
        context: BuiltinContext,
        service: IntegrityService,
        candidate_id: str,
        authorization: CheckAuthorization,
        *,
        evaluation_asof: dt.datetime,
    ) -> ChallengeResult:
        """Open the check window once, score the frozen model, delegate every criterion to ``IntegrityService``.

        Idempotent: a finished candidate returns its stored decision (no second look); a stored failure re-raises.
        """
        self.gateway.authorize(ctx, "integrity.evaluate_protected")
        self.gateway.authorize(ctx, "certificate.issue")
        asof = require_aware(evaluation_asof, "evaluation_asof")
        frozen = self.frozen(candidate_id)
        auth = authorization
        if (
            not self.authority.verify(auth)
            or getattr(auth, "candidate_digest", None) != frozen.digest()
        ):
            self._deny(
                candidate_id,
                ErrorCode.PARTITION_ACCESS_DENIED,
                "check authorization is missing, forged, or bound to another candidate",
            )
        prior = self.run_state(candidate_id)
        if prior and prior["state"] == "completed":
            if frozen.policy_digest != canonical_digest(
                service.policy.model_dump(mode="json")
            ):
                self._deny(
                    candidate_id,
                    ErrorCode.ASOF_VIOLATION,
                    "resumed integrity policy differs from frozen policy",
                )
            decision = IntegrityDecision.model_validate(prior["decision"])
            if (
                decision.admitted
                and service.registry.get(
                    "admission_certificates", decision.certificate_id
                )
                is not None
            ):
                service.restore_admission(
                    decision, model_hash=frozen.model_hash
                )
            return ChallengeResult(
                decision,
                prior["hidden"],
                bool(prior["hidden"].get("check_opened")),
            )
        if prior and prior["state"] == "failed":
            fail(
                ErrorCode.SCHEMA_MISMATCH,
                f"challenge previously failed: {prior['failure']}",
                "Failed checks are not retried; the access is spent.",
            )
        if frozen.policy_digest != canonical_digest(
            service.policy.model_dump(mode="json")
        ):
            self._deny(
                candidate_id,
                ErrorCode.ASOF_VIOLATION,
                "integrity policy differs from the policy frozen with the candidate",
            )
        seal = self.authority.evidence_seal()
        if (
            getattr(
                getattr(service, "evidence_seal", None), "fingerprint", None
            )
            != seal.fingerprint
            or service.allow_unbound_fixture
        ):
            self._deny(
                candidate_id,
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                "integrity service must hold this authority's evidence seal and refuse unbound observations (allow_unbound_fixture=False)",
            )
        card = self.integrity_source_card(context, frozen)
        if service.cards.get(frozen.source_id) != card:
            self._deny(
                candidate_id,
                ErrorCode.SCHEMA_MISMATCH,
                "integrity service source card differs from the frozen dataset source card",
            )
        report = (
            context.report(frozen.registration_id, frozen.variant_name) or {}
        )
        if report.get("report_hash") != frozen.report_hash:
            self._deny(
                candidate_id,
                ErrorCode.SCHEMA_MISMATCH,
                "registered ledger report no longer matches the frozen candidate",
            )
        dataset, artifact = self._provenance(context, report, candidate_id)
        if artifact["predictions_hash"] != frozen.predictions_hash or artifact[
            "model_hash"
        ] != canonical_digest(artifact["frozen_model"]):
            self._deny(
                candidate_id,
                ErrorCode.SCHEMA_MISMATCH,
                "design predictions or model differ from the frozen candidate",
            )
        self._check_window(candidate_id, frozen, auth, asof, ctx)
        recipe, cfg = (
            RECIPES[frozen.recipe_id],
            VariantConfig(**frozen.variant_params),
        )
        record = self.candidate_record(context, frozen)

        if (
            frozen.proxy_labels and not frozen.fixture
        ):  # real proxy/unsupported target: research shadow, check rows stay closed
            decision = IntegrityDecision(
                candidate_id=candidate_id,
                charter=frozen.charter,
                outcome=Outcome.SHADOW,
                scope=SCOPE_RESEARCH,
                reasons=[
                    "recipe relies on proxy / unsupported targets; authentic evidence required by the charter is absent",
                    "research shadow only: no check window opened, no certificate",
                ],
                labels={
                    "evidence_mode": "design_partition_research",
                    "admitted": "false",
                },
                evidence={"report_hash": frozen.report_hash},
                limitations=list(frozen.proxy_labels),
            )
            hidden = {
                "check_opened": False,
                "reason": "proxy_or_unsupported_target",
            }
            self._tx(
                lambda c: c.execute(
                    "INSERT OR IGNORE INTO runs (candidate_id, state, authorization_id) VALUES (?,?,?)",
                    (candidate_id, "started", auth.authorization_id),
                )
            )
            self._complete(candidate_id, "completed", decision, hidden, None)
            self.event(
                candidate_id,
                "research_shadow_no_check",
                "proxy/unsupported target",
            )
            return ChallengeResult(decision, hidden, False)

        service.register_candidate(
            record, registered_at=frozen.registered_at
        )  # registration precedes any outcome access

        def reserve(conn: sqlite3.Connection) -> bool:
            if conn.execute(
                "SELECT 1 FROM family_access WHERE family=? AND candidate_id=?",
                (frozen.mechanism_family, candidate_id),
            ).fetchone():
                return True  # resume of an interrupted run: already counted
            used = conn.execute(
                "SELECT COUNT(*) FROM family_access WHERE family=?",
                (frozen.mechanism_family,),
            ).fetchone()[0]
            if used >= auth.max_family_accesses:
                return False
            conn.execute(
                "INSERT INTO family_access VALUES (?,?,?)",
                (frozen.mechanism_family, candidate_id, auth.authorization_id),
            )
            conn.execute(
                "INSERT OR IGNORE INTO runs (candidate_id, state, authorization_id) VALUES (?,?,?)",
                (candidate_id, "started", auth.authorization_id),
            )
            return True

        if not self._tx(reserve):
            self._deny(
                candidate_id,
                ErrorCode.PROVIDER_BUDGET_EXCEEDED,
                f"finite check budget for family {frozen.mechanism_family!r} exhausted ({auth.max_family_accesses})",
            )
        self.event(candidate_id, "check_access_reserved", auth.authorization_id)
        try:
            design_start, _ = dataset.design_window
            view = dataclasses.replace(
                dataset,
                spec=dataset.spec.model_copy(
                    update={
                        "partitions": {
                            DESIGN: PartitionSpec(
                                start=design_start, end=auth.check_end
                            )
                        }
                    }
                ),
            )
            panel = load_panel(
                view,
                required_columns(recipe, dataset),
                partition=DESIGN,
                research_asof=asof,
            )
            months = [
                m
                for m in range(
                    month_index(auth.check_start),
                    month_index(auth.check_end) + 1,
                )
                if auth.check_start <= month_end_date(m) <= auth.check_end
            ]
            rows = score_frozen(
                panel,
                dataset,
                recipe,
                cfg,
                artifact["frozen_model"],
                months,
                resolved_by=min(end_of_day(auth.check_end), asof),
            )
            design_obs, check_obs = (
                _outcomes(artifact["predictions"], frozen.bridge_sign),
                _outcomes(rows, frozen.bridge_sign),
            )
            ev = dataset.spec.evaluation
            bundle = seal.issue(  # sealed only now: provenance was re-verified above and the check window is authorized
                ctx,
                record=record,
                scope=frozen.scope,
                design_observations=design_obs,
                check_observations=check_obs,
                design_portfolio=tail_membership(
                    artifact["predictions"], panel, dataset, frozen.bridge_sign
                ),
                check_portfolio=tail_membership(
                    rows, panel, dataset, frozen.bridge_sign
                ),
                bridge_sign=frozen.bridge_sign,
                tail_fraction=ev.tail_fraction,
                cost_bps_one_way=ev.cost_bps_one_way,
                cost_source="dataset manifest evaluation.cost_bps_one_way (descriptor-bound); integrity applies its own floor",
                issued_at=asof,
                provenance={
                    "frozen_digest": frozen.digest(),
                    "report_hash": frozen.report_hash,
                    "predictions_hash": frozen.predictions_hash,
                    "implementation_hash": frozen.implementation_hash,
                    "data_file_sha256": frozen.data_file_sha256,
                    "authorization_id": auth.authorization_id,
                    "dataset_class": "fixture"
                    if frozen.fixture
                    else "operator_authorized",
                },
            )
            decision = service.evaluate(
                record,
                ctx=ctx,
                evaluation_asof=asof,
                certificate_scope=frozen.scope,
                multiplicity=frozen.multiplicity,
                evidence=bundle,
            )
        except Exception as exc:
            self._complete(
                candidate_id,
                "failed",
                None,
                None,
                f"{type(exc).__name__}: {exc}"[:600],
            )
            self.event(candidate_id, "failed", f"{type(exc).__name__}: {exc}")
            raise
        hidden = {
            "check_opened": True,
            "check_rows": len(check_obs),
            "check_rows_hash": canonical_digest(rows),
            "check_dates": sorted(
                {o.formation_at.isoformat() for o in check_obs}
            ),
            "design_rows": len(design_obs),
            "check_window": [
                auth.check_start.isoformat(),
                auth.check_end.isoformat(),
            ],
            "metrics": decision.metrics,
            "criteria": decision.criteria,
            "outcome": decision.outcome.value,
        }
        self._complete(candidate_id, "completed", decision, hidden, None)
        self.event(candidate_id, "completed", decision.outcome.value)
        return ChallengeResult(decision, hidden, True)

    # ---------------------------------------------------------- publication
    def publish_forecasts(
        self,
        ctx: RoleContext,
        context: BuiltinContext,
        service: IntegrityService,
        publication: PublicationService,
        artifacts: Any,
        candidate_id: str,
        authorization: CheckAuthorization,
        *,
        published_at: dt.datetime,
        information_month: Optional[int] = None,
        feature_window: Optional[tuple[dt.date, dt.date]] = None,
    ) -> Any:
        """Publish frozen-model forecasts for a STORED ADMITTED decision. ``ctx`` is the strategy namespace's trusted context.

        ``artifacts`` is an ``ArtifactIndex``-like object (register_payload / exists). The return mapping is an in-sample
        out-of-training slope (``calibrate_return_bridge``) of the bridge-signed return on the walk-forward design predictions,
        with measured residual dispersion. ``TargetPrediction`` stays in the native ``centered_rank`` prediction; only
        ``ReturnForecast`` is converted to ``decimal_return``.
        """
        published_at = require_aware(published_at, "published_at")
        frozen = self.frozen(candidate_id)
        run = self.run_state(candidate_id)
        if (
            not run
            or run["state"] != "completed"
            or run["decision"]["outcome"] != Outcome.ADMITTED.value
            or not run["hidden"].get("check_opened")
        ):
            self._deny(
                candidate_id,
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                "no stored ADMITTED challenge decision for this candidate",
            )
        decision = IntegrityDecision.model_validate(run["decision"])
        service.restore_admission(decision, model_hash=frozen.model_hash)
        certificate = service.certificate(candidate_id)
        if (
            certificate.certificate_id != decision.certificate_id
            or certificate.model_hash != frozen.model_hash
        ):
            self._deny(
                candidate_id,
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                "certificate does not match the stored decision / frozen model",
            )
        if not str(decision.labels.get("evidence_binding", "")).startswith(
            "producer_sealed:"
        ):
            self._deny(
                candidate_id,
                ErrorCode.ADMISSION_CERTIFICATE_INVALID,
                "stored decision was not produced from sealed evidence",
            )
        if frozen.proxy_labels and not frozen.fixture:
            self._deny(
                candidate_id,
                ErrorCode.SCHEMA_MISMATCH,
                "proxy / unsupported target has no honest return bridge; it remains shadow",
            )
        if (
            not self.authority.verify(authorization)
            or authorization.candidate_digest != frozen.digest()
        ):
            self._deny(
                candidate_id,
                ErrorCode.PARTITION_ACCESS_DENIED,
                "publication row access needs the candidate's integrity authorization",
            )
        report = (
            context.report(frozen.registration_id, frozen.variant_name) or {}
        )
        dataset, artifact = self._provenance(context, report, candidate_id)
        recipe, cfg = (
            RECIPES[frozen.recipe_id],
            VariantConfig(**frozen.variant_params),
        )
        month = (
            information_month
            if information_month is not None
            else month_index(authorization.check_end)
        )
        start, end = feature_window or (
            authorization.check_start,
            authorization.check_end,
        )
        # This is a trusted-host feature grant, never a strategy-tool argument.
        # Read only declared observable columns; publication cannot open labels.
        if (
            start > end
            or end > published_at.date()
            or not start <= month_end_date(month) <= end
        ):
            self._deny(
                candidate_id,
                ErrorCode.ASOF_VIOLATION,
                "forecast information month lies outside the authorized window",
            )
        design_start, _ = dataset.design_window
        view = dataclasses.replace(
            dataset,
            spec=dataset.spec.model_copy(
                update={
                    "partitions": {
                        DESIGN: PartitionSpec(start=design_start, end=end)
                    }
                }
            ),
        )
        columns = {
            c
            for c in required_columns(recipe, dataset)
            if dataset.spec.columns[c].role != "label"
        }
        panel = load_panel(
            view, columns, partition=DESIGN, research_asof=published_at
        )
        rows = score_frozen(
            panel, dataset, recipe, cfg, artifact["frozen_model"], [month]
        )
        # Calibration uses ONLY the persisted walk-forward (out-of-training) design predictions, never the final model on its own training months.
        try:
            cal = calibrate_return_bridge(
                artifact["predictions"],
                bridge_sign=frozen.bridge_sign,
                design_end=dt.date.fromisoformat(frozen.design_window[1]),
                min_months=dataset.spec.evaluation.min_eval_months,
            )
        except CalibrationError as exc:
            self._deny(
                candidate_id,
                ErrorCode.SCHEMA_MISMATCH,
                f"no honest return bridge: {exc}",
            )
        slope = cal["slope"]
        if not rows or slope <= 0:
            self._deny(
                candidate_id,
                ErrorCode.SCHEMA_MISMATCH,
                "no forecastable rows or non-positive out-of-training return slope; nothing published",
            )
        month_mean = math.fsum(r["p"] for r in rows) / len(rows)
        cutoff = month_end_known_at(month)
        record = self.candidate_record(context, frozen)
        evidence_id, unc_id = (
            f"evidence_{candidate_id}",
            f"uncertainty_{candidate_id}",
        )
        artifacts.register_payload(
            evidence_id,
            "challenge_bridge_evidence",
            frozen.namespace,
            {
                "frozen_digest": frozen.digest(),
                "decision_evidence": decision.evidence,
                "scope": frozen.scope,
            },
        )
        artifacts.register_payload(
            unc_id,
            "design_oos_residual_sd",
            frozen.namespace,
            {
                **cal,
                "units": "decimal_return per unit of the native centered_rank prediction",
                "predictions_hash": frozen.predictions_hash,
            },
        )
        model = ModelVersionRecord(
            owner=frozen.namespace,
            created_at=published_at,
            model_id=frozen.model_id,
            candidate_id=candidate_id,
            charter=frozen.charter,
            namespace=frozen.namespace,
            model_hash=frozen.model_hash,
            bridge_id=f"design_oos_slope_sign{frozen.bridge_sign:+d}",
            bridge_slope_bp=f"{slope * 1e4:.6g}",
            training_cutoff=end_of_day(
                dt.date.fromisoformat(frozen.design_window[1])
            ),
            feature_names=list(artifact["frozen_model"]["feature_names"]),
        )
        forecasts = []
        for r in rows:
            alpha = Decimal(
                repr(frozen.bridge_sign * slope * (r["p"] - month_mean))
            )  # decimal return: only this field uses the slope
            forecasts.append(
                Forecast(
                    owner=frozen.namespace,
                    created_at=published_at,
                    forecast_id=f"fc_{candidate_id}_{month}_{r['s']}"[:100],
                    strategy_id=candidate_id,
                    model_id=frozen.model_id,
                    admission_certificate_id=certificate.certificate_id,
                    security_id=r["s"],
                    information_cutoff=cutoff,
                    earliest_formation_at=published_at,
                    target=TargetPrediction(
                        target_id=frozen.target_id,
                        prediction=Decimal(repr(r["p"])),
                        unit="centered_rank",
                        reference_prediction=Decimal("0"),
                        reference_type="cross_sectional_neutral_centered_rank",
                        expected_resolution_at=target_resolve_at(
                            recipe, month, dataset
                        ),
                    ),
                    return_forecast=ReturnForecast(
                        expected_excess_return=alpha,
                        unit="decimal_return",
                        horizon_trading_days=frozen.horizon_trading_days,
                        bridge_id=model.bridge_id,
                        uncertainty_artifact_id=unc_id,
                    ),
                    mechanism_ids=[frozen.mechanism_family],
                    source_family_ids=record.source_ids,
                    evidence_artifact_ids=[evidence_id],
                    expires_at=published_at
                    + dt.timedelta(days=frozen.horizon_trading_days),
                ).sealed()
            )
        return publication.publish(
            ctx,
            forecasts=forecasts,
            certificate=certificate,
            model=model,
            published_at=published_at,
            artifact_exists=artifacts.exists,
            integration_horizon_days=frozen.horizon_trading_days,
            source_snapshots={frozen.source_id: frozen.source_snapshot_id},
        )


def _outcomes(rows: list[dict], bridge_sign: int) -> list[ForecastOutcome]:
    """Rows -> ForecastOutcome. ``net_forward_return`` carries ``bridge_sign x`` GROSS next-period excess return (the field name is
    historical); trading costs are priced separately by integrity from the sealed tail-portfolio membership and turnover."""
    out = []
    for r in rows:
        formation = month_end_known_at(r["m"])
        out.append(
            ForecastOutcome(
                security_id=r["s"],
                formation_at=formation,
                feature_available_at=formation,
                label_resolves_at=dt.datetime.fromisoformat(r["x"]),
                prediction=r["p"],
                baseline_prediction=r["b"],
                target=r["t"],
                net_forward_return=bridge_sign * r["r"],
            )
        )
    return out
