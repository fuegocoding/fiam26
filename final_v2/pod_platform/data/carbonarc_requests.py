"""Host-bounded Carbon Arc extraction requests, receipts and research-only tables.

A pod selects registered dataset/entity/metric IDs; it cannot supply a tool,
question, URL, credential or path. Budgets count attempted calls (not guessed
vendor dollar prices). No import/construction operation makes a network call.
"""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import fcntl
import importlib.util
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # Research Parquet support is optional; CSV/core tools remain usable.

from ..application.common import atomic_write_json, now_utc
from ..application.common import file_sha256 as _sha
from ..contracts.base import StrictModel, canonical_digest, require_aware
from ..contracts.sources import SourceCard
from ..pods.charters import pod_spec
from ..research.jobs import open_sqlite

_IDENT = r"^[a-z][a-z0-9_]{0,63}$"
_TEXT = re.compile(r"^[A-Za-z0-9 ,.\-_&%()\'/]{1,200}$")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS carbon_request_events(
 key TEXT PRIMARY KEY, namespace TEXT NOT NULL, kind TEXT NOT NULL,
 payload TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS carbon_request_no_update BEFORE UPDATE ON carbon_request_events
 BEGIN SELECT RAISE(ABORT,'Carbon requests are immutable'); END;
CREATE TRIGGER IF NOT EXISTS carbon_request_no_delete BEFORE DELETE ON carbon_request_events
 BEGIN SELECT RAISE(ABORT,'Carbon requests are append-only'); END;
"""


class CarbonArcPolicy(StrictModel):
    """Immutable host limits; analytical attempts default to zero."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    allow_remote: bool = False
    max_analytical_calls: int = Field(default=0, ge=0, le=100)
    max_metadata_calls: int = Field(default=20, ge=0, le=1000)
    max_calls_per_pod: int = Field(default=3, ge=0, le=100)
    max_requests: int = Field(default=100, ge=1, le=1000)
    max_rows: int = Field(default=5000, ge=1, le=20000)
    max_days: int = Field(default=366, ge=1, le=3660)
    max_response_bytes: int = Field(default=2_000_000, ge=1000, le=10_000_000)


class CarbonArcCapability(StrictModel):
    """A host-reviewed dataset, entity, metric and exact provider schema."""

    capability_id: str = Field(pattern=_IDENT)
    namespace: str
    source_id: str = Field(pattern=_IDENT)
    dataset_id: str = Field(pattern=_IDENT)
    tearsheet_name: str = Field(min_length=1, max_length=100)
    entity_id: int = Field(gt=0)
    entity_rep: Literal[
        "company",
        "ticker",
        "product",
        "website",
        "brand",
        "merchant",
        "retailer",
    ]
    metric_id: int = Field(gt=0)
    metric_label: str = Field(min_length=1, max_length=200)
    topic_name: str = Field(min_length=1, max_length=200)
    unit: str = Field(min_length=1, max_length=100)
    date_column: str = "date"
    value_column: str = "value"
    entity_column: str | None = None
    available_column: str | None = None
    dimension_columns: list[str] = Field(default_factory=list, max_length=8)
    local_path: str | None = None
    local_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    local_format: Literal["csv", "parquet"] = "csv"
    access: Literal["request", "confirmed", "blocked"] = "request"
    reviewed_by: str = Field(min_length=1, max_length=100)

    @field_validator("entity_rep", mode="before")
    @classmethod
    def representation(cls, value):
        """Normalize vendor representation casing, without inventing entity types."""
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _valid(self):
        _namespace(self.namespace)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.tearsheet_name):
            raise ValueError(
                "tearsheet_name must be a documented dataset identifier"
            )
        if any(
            not _TEXT.fullmatch(t)
            for t in (self.metric_label, self.topic_name, self.unit)
        ):
            raise ValueError(
                "host labels must be plain bounded data descriptions"
            )
        columns = [self.date_column, self.value_column, *self.dimension_columns]
        columns += [c for c in (self.entity_column, self.available_column) if c]
        if len(set(columns)) != len(columns) or any(
            not re.fullmatch(r"[A-Za-z0-9_ .%-]{1,100}", c) for c in columns
        ):
            raise ValueError("distinct explicit provider column names required")
        if (self.local_path is None) != (self.local_sha256 is None):
            raise ValueError(
                "local snapshots require both a trusted host path and SHA256"
            )
        if self.local_path and not Path(self.local_path).is_absolute():
            raise ValueError("local snapshots require an absolute host path")
        return self


class CarbonArcRequest(StrictModel):
    """The bounded date selection available to a strategy pod."""

    capability_id: str = Field(pattern=_IDENT)
    start: dt.date
    end: dt.date

    @model_validator(mode="after")
    def _bounds(self):
        if self.start > self.end:
            raise ValueError("start must not follow end")
        return self


class CarbonArcSearch(StrictModel):
    """An economic entity/metric search, rather than a fixed stock list."""

    entity_query: str = Field(min_length=2, max_length=100)
    metric_query: str = Field(min_length=2, max_length=100)


class CarbonArcSelection(StrictModel):
    """Select actual results of this pod's recorded provider discovery."""

    discovery_id: str = Field(pattern=r"^discovery_[0-9a-f]{24}$")
    entity_id: int = Field(gt=0)
    metric_id: int = Field(gt=0)
    dataset_id: str = Field(min_length=1, max_length=100)
    start: dt.date
    end: dt.date


def _namespace(value):
    if pod_spec(value).namespace != value:
        raise ValueError("canonical strategy namespace required")
    return value


def _json(response):
    if hasattr(response, "model_dump"):
        response = response.model_dump(mode="json", by_alias=True)
    if not isinstance(response, dict):
        raise ValueError("provider response must be a mapping")
    raw = response
    if isinstance(response.get("structuredContent"), dict):
        response = response["structuredContent"]
    elif "content" in response:
        text = "\n".join(
            c.get("text", "")
            for c in response["content"]
            if c.get("type") == "text"
        )
        try:
            parsed = json.loads(text)
            response = parsed if isinstance(parsed, dict) else {"data": text}
        except ValueError:
            response = {"data": text}
    if raw.get("isError"):
        response = {**response, "isError": True}
    return raw, response


def _tables(text):
    tables, rows = [], []
    for line in str(text).splitlines() + [""]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = [
                c.strip().replace(r"\|", "|")
                for c in re.split(r"(?<!\\)\|", line.strip()[1:-1])
            ]
            if not all(
                re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells
            ):
                rows.append(cells)
        elif rows:
            if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows):
                tables.append([dict(zip(rows[0], row)) for row in rows[1:]])
            rows = []
    return tables


def _day(value):
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    # Year/month-only values do not establish a daily timestamp.
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}(?:T[^ ]+)?", value
    ):
        raise ValueError("observations need explicit ISO dates")
    return dt.date.fromisoformat(value[:10])


def _number(value):
    if (
        value is None
        or isinstance(value, str)
        and value.strip().lower()
        in ("", "-", "—", "na", "n/a", "null", "none", "nan")
    ):
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a metric value")
    if isinstance(value, str):
        value = value.replace(",", "")
    number = float(value)
    if math.isnan(number):
        return None
    if not math.isfinite(number):
        raise ValueError("metric values must be finite or explicitly missing")
    return number


def _flag(value):
    if (
        value is None
        or value is False
        or isinstance(value, str)
        and value.lower() == "false"
    ):
        return False
    if value is True or isinstance(value, str) and value.lower() == "true":
        return True
    raise ValueError("ambiguous provider completeness flag")


class CarbonArcOAuthTransport:
    """Use the project's existing OAuth helper; only the host configures this object."""

    def __init__(self, project_root, *, online=False, config_path=None):
        self.root, self.online = Path(project_root).resolve(), online
        self.config_path = str(
            config_path or Path.home() / ".codex/config.toml"
        )

    def __call__(self, tool_name, arguments):
        if not self.online:
            raise PermissionError(
                "Carbon Arc remote transport requires host online opt-in"
            )
        if tool_name not in (
            "data_library",
            "text_to_insight",
            "search_entities",
            "search_insights",
        ):
            raise ValueError("unreviewed Carbon Arc tool")
        spec = importlib.util.spec_from_file_location(
            "carbonarc_request_host_transport",
            self.root / "final_v2/scripts/carbonarc/download.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        connection = SimpleNamespace(
            config=self.config_path,
            server="carbonarc",
            url=None,
            auth_dir=self.root / "local/carbonarc-auth",
            callback_port=8765,
            wait_for_login=False,
        )

        async def call():
            async with module.connect(connection, interactive=False) as session:
                return (
                    await session.call_tool(tool_name, arguments)
                ).model_dump(mode="json", by_alias=True)

        return asyncio.run(call())


class CarbonArcRequests:
    """Persist requests and receipts before exposing research-only data artefacts."""

    def __init__(
        self, state_dir, *, transport=None, policy=None, clock=now_utc
    ):
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path, self.transport, self.clock = (
            self.state_dir / "carbon_requests.sqlite",
            transport,
            clock,
        )
        self.policy = CarbonArcPolicy.model_validate(
            policy or CarbonArcPolicy()
        )
        with open_sqlite(self.path) as conn:
            conn.executescript(_SCHEMA)
            self._verify(conn)
            self._put(
                conn,
                "policy",
                "host",
                "policy",
                self.policy.model_dump(mode="json"),
            )

    @staticmethod
    def _verify(conn):
        for key, ns, kind, body, digest in conn.execute(
            "SELECT * FROM carbon_request_events"
        ):
            if canonical_digest([key, ns, kind, json.loads(body)]) != digest:
                raise ValueError("Carbon request ledger integrity failure")

    @staticmethod
    def _put(conn, key, namespace, kind, value):
        sha = canonical_digest([key, namespace, kind, value])
        previous = conn.execute(
            "SELECT digest FROM carbon_request_events WHERE key=?", (key,)
        ).fetchone()
        if previous:
            if previous[0] != sha:
                raise ValueError("immutable Carbon request or policy changed")
            return
        conn.execute(
            "INSERT INTO carbon_request_events VALUES(?,?,?,?,?)",
            (
                key,
                namespace,
                kind,
                json.dumps(value, sort_keys=True, allow_nan=False),
                sha,
            ),
        )

    def _now(self):
        return require_aware(self.clock(), "clock")

    def _events(self, kind, namespace=None):
        with open_sqlite(self.path) as conn:
            self._verify(conn)
            rows = conn.execute(
                "SELECT namespace,payload FROM carbon_request_events WHERE kind=? ORDER BY rowid",
                (kind,),
            ).fetchall()
        return [
            json.loads(body)
            for ns, body in rows
            if namespace is None or ns == namespace
        ]

    def register_capability(self, capability):
        """Host-only review operation. Never expose as a model or MCP grant tool."""
        value = CarbonArcCapability.model_validate(capability)
        raw = value.model_dump(mode="json")
        with open_sqlite(self.path, immediate=True) as conn:
            self._verify(conn)
            self._put(
                conn,
                "cap:" + value.namespace + ":" + value.capability_id,
                value.namespace,
                "capability",
                raw,
            )
        return raw

    def discover_stock(self, namespace, request):
        """Discover entities and metrics chosen by a pod, charging free-call caps."""
        _namespace(namespace)
        request = CarbonArcSearch.model_validate(request)
        if any(
            not _TEXT.fullmatch(v)
            for v in (request.entity_query, request.metric_query)
        ):
            raise ValueError(
                "Searches must contain plain entity and economic metric names."
            )
        key = (
            "discovery_"
            + canonical_digest([namespace, request.model_dump()])[:24]
        )
        with open_sqlite(self.path, immediate=True) as conn:
            self._verify(conn)
            old = conn.execute(
                "SELECT payload FROM carbon_request_events WHERE key=?",
                ("result:" + key,),
            ).fetchone()
            if old:
                return json.loads(old[0])
            charged = conn.execute(
                "SELECT 1 FROM carbon_request_events WHERE key=?",
                ("reserve:" + key,),
            ).fetchone()
            used = [
                json.loads(r[0])
                for r in conn.execute(
                    "SELECT payload FROM carbon_request_events WHERE kind='reservation'"
                )
            ]
            count = sum(
                r.get("calls", 1) for r in used if r["call_kind"] == "metadata"
            )
            if charged:
                return {
                    "status": "uncertain",
                    "discovery_id": key,
                    "reason": "Interrupted discovery retained.",
                }
            if not self.policy.allow_remote or self.transport is None:
                return {"status": "remote_not_configured", "discovery_id": key}
            if (
                count + 3 > self.policy.max_metadata_calls
                or len(used) + 3 > self.policy.max_requests
            ):
                return {"status": "budget_exhausted", "discovery_id": key}
            self._put(
                conn,
                "reserve:" + key,
                namespace,
                "reservation",
                {
                    "request_id": key,
                    "namespace": namespace,
                    "call_kind": "metadata",
                    "calls": 3,
                },
            )
        try:
            answers = {}
            for kind, tool, query in (
                ("entities", "search_entities", request.entity_query),
                ("metrics", "search_insights", request.metric_query),
                ("datasets", "data_library", request.metric_query),
            ):
                arguments = {"query": query}
                if tool != "data_library":
                    arguments["limit"] = 10
                raw, response = _json(self.transport(tool, arguments))
                if (
                    response.get("isError")
                    or len(json.dumps(raw)) > self.policy.max_response_bytes
                ):
                    raise ValueError(
                        "Provider discovery failed or exceeded the response budget."
                    )
                answers[kind] = response.get("data", [])[:10]
            result = {
                "status": "discovered",
                "discovery_id": key,
                "namespace": namespace,
                **answers,
                "formation_eligible": False,
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "discovery_id": key,
                "namespace": namespace,
                "error_type": type(exc).__name__,
            }
        with open_sqlite(self.path, immediate=True) as conn:
            self._put(conn, "result:" + key, namespace, "discovery", result)
        return result

    def request_stock(self, namespace, selection):
        """Request a pod-selected stock and metric using recorded vendor IDs."""
        selection = CarbonArcSelection.model_validate(selection)
        discovery = next(
            (
                r
                for r in self._events("discovery", namespace)
                if r["discovery_id"] == selection.discovery_id
            ),
            None,
        )
        if not discovery or discovery["status"] != "discovered":
            raise ValueError("Select a successful discovery from this pod.")
        entity = next(
            (
                e
                for e in discovery["entities"]
                if e["entity_id"] == selection.entity_id
            ),
            None,
        )
        metric = next(
            (
                m
                for m in discovery["metrics"]
                if m["insight_id"] == selection.metric_id
            ),
            None,
        )
        dataset = next(
            (
                d
                for d in discovery["datasets"]
                if d["tearsheet_name"] == selection.dataset_id
            ),
            None,
        )
        if not all((entity, metric, dataset)):
            raise ValueError(
                "Entity, metric and dataset must come from the recorded discovery."
            )
        identifier = (
            "ca_"
            + canonical_digest(
                [selection.entity_id, selection.metric_id, selection.dataset_id]
            )[:24]
        )
        self.register_capability(
            dict(
                namespace=namespace,
                capability_id=identifier,
                source_id=identifier,
                dataset_id=identifier,
                tearsheet_name=selection.dataset_id,
                entity_id=selection.entity_id,
                entity_rep=entity["entity_representation"],
                metric_id=selection.metric_id,
                metric_label=metric["insight_label"],
                topic_name=metric["topic_label"],
                unit="provider_native_unverified",
                date_column="DATE",
                value_column=metric["insight_label"].upper(),
                entity_column="ENTITY ID",
                reviewed_by="configured_provider_discovery_policy",
            )
        )
        return self.request(
            namespace,
            {
                "capability_id": identifier,
                "start": selection.start,
                "end": selection.end,
            },
        )

    def _capability(self, namespace, capability_id):
        _namespace(namespace)
        return next(
            (
                CarbonArcCapability.model_validate(c)
                for c in self._events("capability", namespace)
                if c["capability_id"] == capability_id
            ),
            None,
        )

    def describe(self, namespace, capability_id):
        """Retrieve documented metadata within the frozen free-call allowance."""
        capability = self._capability(namespace, capability_id)
        if capability is None:
            raise ValueError("unknown capability in caller namespace")
        return self._execute(namespace, capability, None, "describe")

    def request(self, namespace, request):
        """Extract a registered native series, reusing a counted identical result."""
        request = CarbonArcRequest.model_validate(request)
        capability = self._capability(namespace, request.capability_id)
        if capability is None:
            raise ValueError("unknown capability in caller namespace")
        if (
            request.end > self._now().date()
            or (request.end - request.start).days + 1 > self.policy.max_days
        ):
            raise ValueError(
                "request is future-dated or exceeds the frozen window budget"
            )
        return self._execute(namespace, capability, request, "timeseries")

    def _execute(self, namespace, capability, request, operation):
        value = {
            "namespace": namespace,
            "operation": operation,
            "capability": capability.model_dump(mode="json"),
            "request": request.model_dump(mode="json") if request else None,
        }
        request_id = "carbon_" + canonical_digest(value)[:24]
        with (self.state_dir / (request_id + ".lock")).open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._locked(
                namespace, capability, request, operation, request_id, value
            )

    def _locked(self, namespace, cap, request, operation, request_id, value):
        local = bool(operation == "timeseries" and cap.local_path)
        call_kind = (
            "local"
            if local
            else "metadata"
            if operation == "describe"
            else "analytical"
        )
        with open_sqlite(self.path, immediate=True) as conn:
            self._verify(conn)
            previous = conn.execute(
                "SELECT payload FROM carbon_request_events WHERE key=?",
                ("result:" + request_id,),
            ).fetchone()
            if previous:
                result = json.loads(previous[0])
                self._check_artifacts(result)
                return result
            self._put(
                conn, "request:" + request_id, namespace, "request", value
            )
            interrupted = (
                conn.execute(
                    "SELECT 1 FROM carbon_request_events WHERE key=?",
                    ("reserve:" + request_id,),
                ).fetchone()
                is not None
            )
            reservations = [
                json.loads(r[0])
                for r in conn.execute(
                    "SELECT payload FROM carbon_request_events WHERE kind='reservation'"
                )
            ]
            refused = None
            if cap.access == "blocked":
                refused = "entitlement_blocked"
            elif not local and (
                not self.policy.allow_remote or self.transport is None
            ):
                refused = "remote_not_configured"
            elif not interrupted:
                used = sum(
                    r.get("calls", 1)
                    for r in reservations
                    if r["call_kind"] == call_kind
                )
                pod_used = sum(
                    r["call_kind"] == "analytical"
                    and r["namespace"] == namespace
                    for r in reservations
                )
                limit = (
                    self.policy.max_metadata_calls
                    if call_kind == "metadata"
                    else self.policy.max_analytical_calls
                )
                if (
                    len(reservations) >= self.policy.max_requests
                    or (call_kind != "local" and used >= limit)
                    or (
                        call_kind == "analytical"
                        and pod_used >= self.policy.max_calls_per_pod
                    )
                ):
                    refused = "budget_exhausted"
            if not refused and not interrupted:
                self._put(
                    conn,
                    "reserve:" + request_id,
                    namespace,
                    "reservation",
                    {
                        "request_id": request_id,
                        "namespace": namespace,
                        "call_kind": call_kind,
                        "reserved_at": self._now().isoformat(),
                    },
                )
        receipt = {
            "request_id": request_id,
            "namespace": namespace,
            "operation": operation,
            "call_kind": call_kind,
            "capability_id": cap.capability_id,
            "dataset_id": cap.dataset_id,
            "source_id": cap.source_id,
            "research_only": True,
            "formation_eligible": False,
            "historical_coverage_established": False,
            "vintage_established": False,
            "request_hash": canonical_digest(value),
        }
        try:
            if refused:
                receipt["status"] = refused
            elif interrupted:
                receipt.update(
                    status="uncertain",
                    reason="Interrupted reserved request retained; no automatic paid retry.",
                )
            else:
                raw_response = (
                    self._local(cap, request)
                    if local
                    else self.transport(
                        "data_library"
                        if operation == "describe"
                        else "text_to_insight",
                        {"dataset_id": cap.tearsheet_name}
                        if operation == "describe"
                        else self._question(cap, request),
                    )
                )
                raw, response = _json(raw_response)
                body = json.dumps(raw, sort_keys=True, allow_nan=False)
                if len(body.encode()) > self.policy.max_response_bytes:
                    receipt.update(
                        status="oversized_response",
                        reason="Response exceeds immutable host byte budget.",
                    )
                else:
                    directory = self.state_dir / "artifacts" / request_id
                    raw_path = directory / "raw.json"
                    atomic_write_json(raw_path, raw)
                    receipt.update(
                        raw_path=str(raw_path),
                        raw_sha256=_sha(raw_path),
                        retrieved_at=self._now().isoformat(),
                    )
                    status = self._response_status(response)
                    if status:
                        receipt["status"] = status
                    elif operation == "describe":
                        receipt.update(
                            status="documented", documentation=response
                        )
                    else:
                        receipt.update(
                            self._extract(cap, request, response, directory)
                        )
        except Exception as exc:
            receipt.update(
                status="failed",
                error_type=type(exc).__name__,
                reason="Request failed; the reserved attempt remains charged. Inspect the private raw receipt if present.",
            )
        receipt["receipt_hash"] = canonical_digest(receipt)
        with open_sqlite(self.path, immediate=True) as conn:
            self._put(
                conn, "result:" + request_id, namespace, "result", receipt
            )
        return receipt

    @staticmethod
    def _question(cap, request):
        return {
            "question": (
                f"Return a time series for Carbon Arc entity {cap.entity_id} ({cap.entity_rep}), "
                f"metric {cap.metric_label}, insight_id {cap.metric_id}, topic {cap.topic_name}, dataset {cap.tearsheet_name}. "
                f"Explicit inclusive dates {request.start.isoformat()} through {request.end.isoformat()}. "
                f"Return every native observation without aggregating dates. Use columns {cap.date_column} and {cap.value_column}; "
                "include entity IDs, dimensions, units, publication timestamps and any truncation. "
                "Do not apply a default trailing window or fabricate missing observations. "
                "Clearly report unavailable data or entitlement errors."
            )
        }

    @staticmethod
    def _response_status(response):
        meta = response.get("metadata") or {}
        text = str(response.get("data", ""))
        diagnostic = text + " " + str(meta.get("error", ""))
        if re.search(
            r"not entitled|entitlement|access denied|unauthorized|permission denied",
            diagnostic,
            re.I,
        ):
            return "entitlement_blocked"
        if (
            _flag(meta.get("rate_limited"))
            or response.get("http_status") == 429
        ):
            return "rate_limited"
        if re.search(
            r"insufficient.{0,30}(credit|balance)|quota|payment required",
            diagnostic,
            re.I,
        ):
            return "vendor_budget_blocked"
        if response.get("isError") or meta.get("error"):
            return "provider_error"
        frameworks = meta.get("frameworks") or []
        if any(
            _flag(f.get("default_date_range_applied"))
            or _flag(f.get("truncated"))
            for f in frameworks
        ):
            return "incomplete_response"
        if (
            meta.get("next_cursor")
            or any(
                _flag(meta.get(k))
                for k in ("truncated", "has_more", "default_date_range_applied")
            )
            or re.search(
                r"too large to display|truncated|only (?:the )?first \d+ rows",
                text,
                re.I,
            )
        ):
            return "incomplete_response"
        return None

    def _extract(self, cap, request, response, directory):
        metadata = response.get("metadata") or {}
        metrics = {
            metadata[k]
            for k in ("insight_id", "metric_id")
            if metadata.get(k) is not None
        }
        metrics.update(
            i["insight_id"]
            for f in metadata.get("frameworks", [])
            for i in f.get("insights", [])
            if "insight_id" in i
        )
        if metrics and {str(i) for i in metrics} != {str(cap.metric_id)}:
            return {
                "status": "scope_mismatch",
                "reason": "Provider metric differs from the host mapping.",
            }
        if metadata.get("entity_id") is not None and str(
            metadata["entity_id"]
        ) != str(cap.entity_id):
            return {
                "status": "scope_mismatch",
                "reason": "Provider entity differs from the host mapping.",
            }
        rows = response.get("rows")
        if rows is None:
            tables = [
                t
                for t in _tables(response.get("data", ""))
                if cap.date_column in t[0] and cap.value_column in t[0]
            ]
            if len(tables) != 1:
                return {
                    "status": "unstructured_result",
                    "reason": "Exactly one table with the host-mapped date/value columns is required.",
                }
            rows = tables[0]
        if not isinstance(rows, list) or not rows:
            return {"status": "empty_response"}
        if len(rows) > self.policy.max_rows:
            return {"status": "row_budget_exceeded"}
        selected, keys, missing = [], set(), 0
        for raw in rows:
            if (
                not isinstance(raw, dict)
                or cap.date_column not in raw
                or cap.value_column not in raw
            ):
                return {"status": "schema_mismatch"}
            day = _day(raw[cap.date_column])
            if not request.start <= day <= request.end:
                return {
                    "status": "scope_mismatch",
                    "reason": "Provider returned dates outside the explicit requested window.",
                }
            if cap.entity_column and str(raw.get(cap.entity_column)) != str(
                cap.entity_id
            ):
                return {
                    "status": "scope_mismatch",
                    "reason": "Provider entity differs from the host mapping.",
                }
            value = _number(raw[cap.value_column])
            missing += value is None
            dims = {c: raw.get(c) for c in cap.dimension_columns}
            if any(
                v is not None and not isinstance(v, (str, int, float, bool))
                for v in dims.values()
            ):
                return {"status": "schema_mismatch"}
            key = canonical_digest([day.isoformat(), dims])
            if key in keys:
                return {
                    "status": "duplicate_observations",
                    "reason": "No implicit deduplication or aggregation is allowed.",
                }
            keys.add(key)
            available = (
                raw.get(cap.available_column) if cap.available_column else None
            )
            if available is not None:
                available = _day(available).isoformat()
                if available > self._now().date().isoformat():
                    return {"status": "future_availability"}
            selected.append(
                {
                    "date": day.isoformat(),
                    "value": value,
                    "unit": cap.unit,
                    "requested_entity_id": cap.entity_id,
                    "observed_entity_id": raw.get(cap.entity_column)
                    if cap.entity_column
                    else None,
                    "metric_id": cap.metric_id,
                    "available_at": available,
                    **dims,
                }
            )
        selected.sort(key=lambda r: (r["date"], canonical_digest(r)))
        data_path = directory / "observations.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with data_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        data_sha = _sha(data_path)
        card = {
            "source_id": cap.source_id,
            "dataset_id": cap.dataset_id,
            "topic": cap.topic_name,
            "snapshot_id": data_sha,
            "status": "SAMPLED",
            "access_confirmed": True,
            "formation_eligible": False,
            "owner": cap.namespace,
            "entity_grain": [cap.entity_rep],
            "created_at": self._now().isoformat(),
            "screen": {
                n: {
                    "status": "sample_only" if n == "coverage" else "unresolved"
                }
                for n in ("history", "metric_lag", "vintage", "coverage")
            },
            "unresolved_questions": [
                "Vendor history, historical vintages, release timing, universe coverage and security joins require independent assessment."
            ],
        }
        source_path = directory / "source_card.json"
        atomic_write_json(
            source_path,
            SourceCard.model_validate(card).sealed().model_dump(mode="json"),
        )
        return {
            "status": "data_ready",
            "data_path": str(data_path),
            "data_sha256": data_sha,
            "source_card_path": str(source_path),
            "source_card_sha256": _sha(source_path),
            "rows": len(selected),
            "missing_values": missing,
            "observed_start": selected[0]["date"],
            "observed_end": selected[-1]["date"],
            "availability_column_observed": bool(cap.available_column),
            "entity_column_observed": bool(cap.entity_column),
            "metric_scope_observed": bool(metrics),
            "unit_attestation": "host_declared",
            "limitations": [
                "Research-only extraction; not a historical-vintage or complete-coverage attestation.",
                "Requested entity is recorded separately when provider rows do not expose entity identity.",
                "Native values and missing observations are retained without inferred interpolation or release dates.",
            ],
        }

    def _local(self, cap, request):
        path = Path(cap.local_path)
        if _sha(path) != cap.local_sha256:
            raise ValueError("trusted local snapshot hash mismatch")
        columns = [cap.date_column, cap.value_column, *cap.dimension_columns]
        columns += [c for c in (cap.entity_column, cap.available_column) if c]
        result = []
        if cap.local_format == "csv":
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    day = _day(row[cap.date_column])
                    if request.start <= day <= request.end:
                        result.append({c: row[c] for c in columns})
                        if len(result) > self.policy.max_rows:
                            break
        else:
            if pq is None:
                raise ValueError("Parquet snapshots require the research extra")
            for batch in pq.ParquetFile(path).iter_batches(
                columns=columns, batch_size=2048
            ):
                for row in batch.to_pylist():
                    if (
                        request.start
                        <= _day(row[cap.date_column])
                        <= request.end
                    ):
                        result.append(row)
                        if len(result) > self.policy.max_rows:
                            break
                if len(result) > self.policy.max_rows:
                    break
        if _sha(path) != cap.local_sha256:
            raise ValueError("trusted local snapshot changed during extraction")
        # Dates/values serialize without substituting receipt time for observation time.
        return {
            "rows": [
                {
                    c: v.isoformat()
                    if isinstance(v, (dt.date, dt.datetime))
                    else v
                    for c, v in r.items()
                }
                for r in result
            ],
            "metadata": {
                "origin": "host_authorized_local_snapshot",
                "source_sha256": cap.local_sha256,
            },
        }

    @staticmethod
    def _check_artifacts(result):
        for name in ("raw", "data", "source_card"):
            if (
                result.get(name + "_path")
                and _sha(result[name + "_path"]) != result[name + "_sha256"]
            ):
                raise ValueError("Carbon request artifact integrity failure")

    def read(self, namespace, request_id):
        """Read and verify an existing receipt from the caller's private namespace."""
        _namespace(namespace)
        result = next(
            (
                r
                for r in self._events("result", namespace)
                if r["request_id"] == request_id
            ),
            None,
        )
        if result is None:
            raise ValueError("request is unavailable in caller namespace")
        self._check_artifacts(result)
        return result

    def snapshots(self, namespace):
        """Return own capabilities, attempt counts and hash-verified receipts."""
        _namespace(namespace)
        results = self._events("result", namespace)
        for result in results:
            self._check_artifacts(result)
        reservations = self._events("reservation", namespace)
        return {
            "namespace": namespace,
            "policy": self.policy.model_dump(mode="json"),
            "capabilities": [
                {
                    k: v
                    for k, v in c.items()
                    if k not in ("local_path", "local_sha256")
                }
                for c in self._events("capability", namespace)
            ],
            "usage": {
                kind: sum(
                    r.get("calls", 1)
                    for r in reservations
                    if r["call_kind"] == kind
                )
                for kind in ("metadata", "analytical", "local")
            },
            "requests": results,
            "formation_eligible": False,
        }
