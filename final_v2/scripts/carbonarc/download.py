#!/usr/bin/env python3
"""Resumable Carbon Arc MCP download transport with persisted request limits."""

from __future__ import annotations

import asyncio
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import tomllib
import webbrowser
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
from jsonschema import validate
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings

ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT = ROOT / "data/carbonarc/nvidia"
CATALOG_OUTPUT = ROOT / "data/carbonarc/catalog"
ANALYTICS = {"text_to_insight", "framework_to_insight"}
# Analytical calls share a conservative local bucket; never alternate tools to evade a limit.
LIMITS = {
    "analytics": (8, 90),
    "search_insights": (45, 500),
    "get_insights_from_entity": (30, 350),
    "get_entities_from_insight": (30, 350),
}
NON_NUMERIC = re.compile(
    r"event|filing|earnings|connection|launch|calendar|ipo|merger", re.I
)
LOG = logging.getLogger("carbonarc")


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value, private=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600 if private else 0o644,
    )
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if private:
        path.chmod(0o600)


def unpack(response):
    if response.get("structuredContent") is not None:
        return response["structuredContent"]
    for item in response.get("content", []):
        if item.get("type") == "text":
            try:
                value = json.loads(item["text"])
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError):
                pass
    return {
        "data": "\n".join(
            x.get("text", "") for x in response.get("content", [])
        )
    }


def retry_seconds(value, now=None):
    now = time.time() if now is None else now
    try:
        seconds = float(value)
        if seconds >= 0 and seconds < float("inf"):
            return seconds
    except (ValueError, TypeError):
        pass
    try:
        return max(0, parsedate_to_datetime(value).timestamp() - now)
    except (ValueError, TypeError, OverflowError):
        return None


def classify(response):
    """Do not equate HTTP/MCP success with a successful data download."""
    data = unpack(response)
    meta = data.get("metadata") or {}
    text = str(data.get("data", ""))
    if (
        meta.get("rate_limited")
        or meta.get("admission_rejected")
        or response.get("http_status") == 429
    ):
        return "rate_limited", retry_seconds(
            meta.get("retry_after", response.get("retry_after"))
        )
    if re.search(
        r"insufficient.{0,30}(balance|credit)|payment required|daily.{0,30}limit.{0,20}(exceed|reach)",
        str(meta.get("error", "")) + " " + text,
        re.I,
    ) or re.search(r"quota", str(meta.get("error", "")), re.I):
        return "quota_blocked", None
    if response.get("isError") or meta.get("error"):
        return "error", None
    if re.search(
        r"retrieval error|KeyError:|Traceback|internal (processing |server )?error|technical error",
        text,
        re.I,
    ):
        return "error", None
    if re.search(
        r"result set is too large|too large to display|truncat(ed|ion)|only (the )?(first|last) \d+ rows",
        text,
        re.I,
    ):
        return "oversized", None
    if str(meta.get("default_date_range_applied", "")).lower() == "true":
        return "default_range", None
    for framework in meta.get("frameworks", []):
        if (
            str(framework.get("default_date_range_applied", "")).lower()
            == "true"
        ):
            return "default_range", None
    if meta.get("truncated") or meta.get("has_more") or meta.get("next_cursor"):
        return "oversized", None
    if not text.strip():
        return "empty_unverified", None
    return "downloaded_unverified", None


def validate_scope(job, response, status):
    details = json.loads(job["details"])
    if details["kind"] != "history" or status != "downloaded_unverified":
        return status
    frameworks = (unpack(response).get("metadata") or {}).get("frameworks", [])
    for framework in frameworks:
        insight_ids = {
            item.get("insight_id") for item in framework.get("insights", [])
        }
        if insight_ids and insight_ids != {details["metric"]["insight_id"]}:
            return "scope_mismatch"
        date_range = (
            framework.get("framework_request", {})
            .get("filters", {})
            .get("date_range", {})
        )
        if (
            date_range.get("start_date")
            and date_range["start_date"] > details["start"]
        ):
            return "scope_mismatch"
    return status


def markdown_tables(text):
    """Extract tables verbatim; no guessed numeric conversion or fabricated rows."""
    tables, rows = [], []
    for line in str(text).splitlines() + [""]:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            cells = re.split(r"(?<!\\)\|", line.strip()[1:-1])
            cells = [c.strip().replace(r"\|", "|") for c in cells]
            if all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):
                continue
            rows.append(cells)
        elif rows:
            if len(rows) > 1:
                tables.append(rows)
            rows = []
    return tables


class Queue:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "downloads.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, tool TEXT, args TEXT, details TEXT,
            priority TEXT, status TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0,
            not_before REAL DEFAULT 0, result_path TEXT, note TEXT);
          CREATE TABLE IF NOT EXISTS calls(bucket TEXT, at REAL);
          CREATE TABLE IF NOT EXISTS cooldown(bucket TEXT PRIMARY KEY, until REAL);
        """)
        self.db.commit()

    def add(self, tool, args, details, priority):
        encoded = json.dumps({"tool": tool, "args": args}, sort_keys=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()[:24]
        self.db.execute(
            "INSERT OR IGNORE INTO jobs(id,tool,args,details,priority) VALUES(?,?,?,?,?)",
            (key, tool, json.dumps(args), json.dumps(details), priority),
        )
        self.db.commit()
        return key

    @staticmethod
    def bucket(tool):
        return "analytics" if tool in ANALYTICS else tool

    def delay(self, tool, now=None):
        now = time.time() if now is None else now
        bucket = self.bucket(tool)
        limits = LIMITS.get(
            bucket, (30, 350)
        )  # conservative default, not a vendor-documented quota
        row = self.db.execute(
            "SELECT until FROM cooldown WHERE bucket=?", (bucket,)
        ).fetchone()
        waits = [max(0, row[0] - now) if row else 0]
        for window, limit in zip((60, 3600), limits):
            hits = [
                r[0]
                for r in self.db.execute(
                    "SELECT at FROM calls WHERE bucket=? AND at>? ORDER BY at",
                    (bucket, now - window),
                )
            ]
            if len(hits) >= limit:
                waits.append(hits[-limit] + window + 1 - now)
        return max(waits)

    def reserve(self, job, now=None):
        now = time.time() if now is None else now
        with self.db:
            self.db.execute(
                "INSERT INTO calls VALUES(?,?)", (self.bucket(job["tool"]), now)
            )
            self.db.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1 WHERE id=?",
                (job["id"],),
            )

    def defer(self, job, seconds, now=None):
        now = time.time() if now is None else now
        bucket = self.bucket(job["tool"])
        until = now + seconds
        with self.db:
            self.db.execute(
                "INSERT INTO cooldown VALUES(?,?) ON CONFLICT(bucket) DO UPDATE SET until=max(until,excluded.until)",
                (bucket, until),
            )
            self.db.execute(
                "UPDATE jobs SET status='pending',not_before=?,note='rate limited' WHERE id=?",
                (until, job["id"]),
            )

    def finish(self, job, status, path=None, note=None):
        self.db.execute(
            "UPDATE jobs SET status=?,result_path=?,note=? WHERE id=?",
            (status, str(path) if path else None, note, job["id"]),
        )
        self.db.commit()

    def status(self):
        return dict(
            self.db.execute(
                "SELECT status,count(*) FROM jobs GROUP BY status"
            ).fetchall()
        )

    def paid_attempts(self):
        return self.db.execute(
            "SELECT count(*) FROM calls WHERE bucket='analytics'"
        ).fetchone()[0]

    def recover_interrupted(self):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status='pending',note='Resuming interrupted free discovery' WHERE status='running' AND tool NOT IN ('text_to_insight','framework_to_insight')"
            )
            self.db.execute(
                "UPDATE jobs SET status='uncertain',note='Interrupted paid request; review before retry' WHERE status='running'"
            )


def date_start(record, fallback):
    history = str(record.get("history") or "")
    years = [int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", history)]
    return f"{min(years):04d}-01-01" if years else fallback


def choose_entity(metric):
    # Product-brand queries were resolved by Carbon Arc for behavioral data.
    preferences = (
        ("company", "ticker", "product", "website")
        if metric["topic_label"].startswith("Company Reported")
        else ("product", "website", "company", "ticker")
    )
    for rep in preferences:
        for entity in metric["entities"]:
            if entity["rep"] == rep:
                return entity
    return metric["entities"][0]


def add_history(queue, metric, start, end):
    entity = choose_entity(metric)
    label, topic, insight_id = (
        metric["insight_label"],
        metric["topic_label"],
        metric["insight_id"],
    )
    args = {
        "question": (
            f"Return NVIDIA (NVDA), Carbon Arc entity {entity['id']} ({entity['rep']}), "
            f"metric {label}, insight_id {insight_id}, topic {topic}. "
            f"Explicit inclusive date range: {start} through {end}. "
            "Return every observation at the finest native date resolution, sorted earliest first, "
            "with all available dimensions, dates, values, units, currency and publication timestamps. "
            "Do not use a default trailing date window, sample rows, or aggregate over dates. "
            "Report actual first/last observation dates, total row count, any truncation, "
            "and framework IDs or full export URLs if available. Clearly report no data or unsupported requests."
        )
    }
    queue.add(
        "text_to_insight",
        args,
        {"kind": "history", "metric": metric, "start": start, "end": end},
        "1:" + start,
    )


def prepare(queue, seed):
    discovery = json.loads(Path(seed).read_text())
    metrics = discovery["insights"]
    numeric = [
        m
        for m in metrics
        if not NON_NUMERIC.search(m["insight_label"] + " " + m["topic_label"])
    ]
    for metric in numeric:
        queue.add(
            "data_library",
            {"insight_id": metric["insight_id"]},
            {"kind": "provenance", "metric": metric},
            "0:provenance",
        )
    for entity in (
        {"id": 64529, "rep": "company"},
        {"id": 476, "rep": "ticker"},
    ):
        queue.add(
            "get_events_from_entity",
            {
                "entity_id": entity["id"],
                "entity_representation": entity["rep"],
                "limit": 1000,
            },
            {"kind": "event_discovery"},
            "0:events",
        )
    atomic_json(
        queue.output / "plan.json",
        {
            "created_at": stamp(),
            "numeric_insights": len(numeric),
            "non_numeric_insights": len(metrics) - len(numeric),
            "method": "Discover source history; request explicit full history per metric; split oversized date ranges oldest first.",
            "limits": LIMITS,
            "completeness": "MCP analytical responses do not establish complete raw dataset coverage.",
        },
    )
    return numeric


def prepare_catalog(queue, seed):
    """Inventory all accessible datasets; catalog visibility is not a raw-data entitlement."""
    document = json.loads(Path(seed).read_text())
    records = document.get("data")
    if not isinstance(records, list) or any(
        not isinstance(r, dict) or not r.get("tearsheet_name") for r in records
    ):
        raise ValueError("Expected a data_library catalog with dataset names")
    names = set()
    for record in records:
        name = record["tearsheet_name"]
        names.add(name)
        queue.add(
            "data_library",
            {"dataset_id": name},
            {"kind": "catalog_provenance", "dataset_name": name},
            "0:" + name.casefold(),
        )
    atomic_json(
        queue.output / "plan.json",
        {
            "created_at": stamp(),
            "scope": "All accessible Carbon Arc datasets; all entities and all available history",
            "accessible_catalog_datasets": len(names),
            "mode": "free_dataset_discovery_only",
            "raw_download_status": "Requires verified Block/S3/Polaris access; MCP catalog visibility alone does not establish bulk entitlement",
            "numeric_query_jobs_created": 0,
            "company_filter": None,
        },
    )
    return len(names)


def build_children(queue, job, response, fallback, end, start_override=None):
    details = json.loads(job["details"])
    if details["kind"] == "catalog_provenance":
        data = unpack(response)
        name = details["dataset_name"]
        key = hashlib.sha256(name.encode()).hexdigest()[:24]
        atomic_json(
            queue.output / "datasets" / (key + ".json"),
            {
                "requested_dataset_name": name,
                "received_at": stamp(),
                "response": data,
                "raw_download_status": "not_downloaded",
            },
        )
    elif details["kind"] == "provenance":
        data = unpack(response).get("data", {})
        start = start_override or date_start(
            data if isinstance(data, dict) else {}, fallback
        )
        add_history(queue, details["metric"], start, end)
    elif details["kind"] == "history":
        status, _ = classify(response)
        if status in ("oversized", "default_range"):
            start, end_date = map(
                dt.date.fromisoformat, (details["start"], details["end"])
            )
            if start < end_date:
                midpoint = start + (end_date - start) // 2
                add_history(
                    queue,
                    details["metric"],
                    start.isoformat(),
                    midpoint.isoformat(),
                )
                add_history(
                    queue,
                    details["metric"],
                    (midpoint + dt.timedelta(days=1)).isoformat(),
                    end_date.isoformat(),
                )
                return "split"
            return "needs_bulk_export"
    return None


def save_response(queue, job, response):
    base = queue.output / "responses" / job["id"]
    current = queue.db.execute(
        "SELECT attempts FROM jobs WHERE id=?", (job["id"],)
    ).fetchone()[0]
    path = base.with_name(base.name + f".attempt-{current:04d}.json")
    atomic_json(
        path,
        {
            "received_at": stamp(),
            "tool": job["tool"],
            "arguments": json.loads(job["args"]),
            "response": response,
        },
    )
    content = unpack(response).get("data")
    if isinstance(content, str) and content:
        base.with_suffix(".md").write_text(content)
        for n, table in enumerate(markdown_tables(content), 1):
            with base.with_name(base.name + f".table-{n}.csv").open(
                "w", newline=""
            ) as stream:
                csv.writer(stream).writerows(table)
    return path


class FileTokens:
    """Private, project-local OAuth state; never read or alter Codex's login."""

    def __init__(self, path):
        self.path = path

    def read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    async def get_tokens(self):
        data = self.read()
        token = data.get("tokens")
        if token is None:
            return None
        if data.get("expires_at") is not None:
            token["expires_in"] = max(0, int(data["expires_at"] - time.time()))
        return OAuthToken.model_validate(token)

    async def set_tokens(self, tokens):
        data = self.read()
        data["tokens"] = tokens.model_dump(mode="json")
        data["expires_at"] = (
            time.time() + tokens.expires_in
            if tokens.expires_in is not None
            else None
        )
        atomic_json(self.path, data, private=True)

    async def get_client_info(self):
        data = self.read().get("client")
        return OAuthClientInformationFull.model_validate(data) if data else None

    async def set_client_info(self, client_info):
        data = self.read()
        data["client"] = client_info.model_dump(mode="json")
        atomic_json(self.path, data, private=True)


class LoginCallback:
    def __init__(self, port):
        self.event = threading.Event()
        self.values = {}
        self.expected_state = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                values = parse_qs(urlsplit(self.path).query)
                valid = (
                    urlsplit(self.path).path == "/callback"
                    and values.get("state", [None])[0] == owner.expected_state
                    and owner.expected_state
                )
                self.send_response(200 if valid else 400)
                self.end_headers()
                self.wfile.write(
                    b"You can return to the downloader."
                    if valid
                    else b"Invalid OAuth callback."
                )
                if valid:
                    owner.values = values
                    owner.event.set()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    async def redirect(self, url):
        self.expected_state = parse_qs(urlsplit(url).query).get(
            "state", [None]
        )[0]
        print("Authorize Carbon Arc in your browser:\n" + url, flush=True)
        await asyncio.to_thread(webbrowser.open, url)

    async def callback(self):
        if not await asyncio.to_thread(self.event.wait, 600):
            raise RuntimeError("OAuth callback timed out; rerun login.")
        if "error" in self.values:
            raise RuntimeError("Carbon Arc authorization was declined.")
        return self.values["code"][0], self.values.get("state", [None])[0]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@contextlib.asynccontextmanager
async def connect(args, interactive=False):

    config_file = Path(args.config).expanduser()
    config = (
        tomllib.loads(config_file.read_text()) if config_file.exists() else {}
    )
    settings = config.get("mcp_servers", {}).get(args.server, {})
    url = args.url or settings.get("url") or "https://mcp.carbonarc.ai"
    headers = dict(settings.get("http_headers", {}))
    for header, variable in settings.get("env_http_headers", {}).items():
        if variable in os.environ:
            headers[header] = os.environ[variable]
    token = (
        getattr(args, "bearer_token", None)
        or os.environ.get("CARBONARC_TOKEN")
        or os.environ.get(settings.get("bearer_token_env_var", ""))
    )
    callback = None
    auth = None
    if token:
        headers["Authorization"] = "Bearer " + token
    if not any(k.lower() == "authorization" for k in headers):
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        storage = FileTokens(Path(args.auth_dir) / (key + ".json"))
        if interactive:
            callback = LoginCallback(args.callback_port)
        elif not await storage.get_tokens():
            if not args.wait_for_login:
                raise RuntimeError(
                    "Standalone OAuth login required. Set carbonarc_download.command to login in final_v2/config.json, then run final_v2/scripts/carbonarc/download.py."
                )
            while not await storage.get_tokens():
                LOG.info(
                    "Waiting for the standalone Carbon Arc browser login; no analytical calls sent."
                )
                await asyncio.sleep(30)

        async def no_login(*_):
            raise RuntimeError(
                "OAuth expired or unavailable. Rerun the login command; the download queue is preserved."
            )

        auth = OAuthClientProvider(
            server_url=url,
            storage=storage,
            client_metadata=OAuthClientMetadata.model_validate(
                {
                    "client_name": "Carbon Arc History Downloader",
                    "redirect_uris": [
                        f"http://127.0.0.1:{callback.port if callback else args.callback_port}/callback"
                    ],
                    "grant_types": ["authorization_code", "refresh_token"],
                    "response_types": ["code"],
                    "token_endpoint_auth_method": "none",
                }
            ),
            redirect_handler=callback.redirect if callback else no_login,
            callback_handler=callback.callback if callback else no_login,
        )
    try:

        async def observe_response(response):
            if response.status_code != 429:
                return
            # Record only MCP tool identity/arguments, never OAuth request bodies or headers.
            try:
                request = json.loads(response.request.content)
            except (ValueError, RuntimeError):
                request = {}
            params = (
                request.get("params", {})
                if request.get("method") == "tools/call"
                else {}
            )
            args._http_rejection = {
                "at": time.time(),
                "http_status": 429,
                "retry_after": response.headers.get("Retry-After"),
                "tool": params.get("name"),
                "arguments": params.get("arguments"),
            }

        async with httpx.AsyncClient(
            auth=auth,
            headers=headers,
            timeout=httpx.Timeout(60, read=600),
            follow_redirects=False,
            event_hooks={"response": [observe_response]},
        ) as client:
            async with streamable_http_client(
                url, http_client=client
            ) as streams:
                async with ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=dt.timedelta(seconds=660),
                ) as session:
                    await session.initialize()
                    yield session
    finally:
        if callback:
            await asyncio.to_thread(callback.close)


def http_failure(exc):
    children = getattr(exc, "exceptions", [])
    for child in children:
        found = http_failure(child)
        if found:
            return found
    response = getattr(exc, "response", None)
    if response is not None:
        return {
            "http_status": response.status_code,
            "retry_after": response.headers.get("Retry-After"),
        }
    return None


async def inspect_tools(session, queue):
    tools, cursor = [], None
    while True:
        result = await session.list_tools(cursor=cursor)
        tools.extend(
            t.model_dump(mode="json", by_alias=True) for t in result.tools
        )
        cursor = result.nextCursor
        if not cursor:
            break
    atomic_json(
        queue.output / "server_tools.json",
        {"inspected_at": stamp(), "tools": tools},
    )
    return {t["name"]: t for t in tools}


def recover_transport_limit(queue, budget, rejection):
    """Recover an interrupted task only when retry is free or its POST was rejected."""
    wait = max(30, retry_seconds(rejection.get("retry_after")) or 0) + 1
    job = queue.db.execute(
        "SELECT * FROM jobs WHERE id=? AND status='running'",
        (budget.get("active_id"),),
    ).fetchone()
    if job:
        proven_rejection = rejection.get("tool") == job[
            "tool"
        ] and rejection.get("arguments") == json.loads(job["args"])
        if job["tool"] in ANALYTICS and not proven_rejection:
            queue.finish(
                job,
                "uncertain",
                note="Transport rate limit; paid execution outcome unconfirmed",
            )
            return None
        path = save_response(
            queue,
            job,
            {
                "http_status": 429,
                "retry_after": rejection.get("retry_after"),
                "metadata": {
                    "rate_limited": True,
                    "recovered_transport_rejection": True,
                },
            },
        )
        queue.defer(job, wait)
        queue.db.execute(
            "UPDATE jobs SET result_path=? WHERE id=?", (str(path), job["id"])
        )
        queue.db.commit()
        budget["active_id"] = None
    return wait


async def work(session, queue, args, budget):
    available = await inspect_tools(session, queue)
    while True:
        sql = "SELECT * FROM jobs WHERE status='pending'"
        if args.discovery_only:
            sql += " AND tool NOT IN ('text_to_insight','framework_to_insight')"
        jobs = queue.db.execute(sql + " ORDER BY priority,id").fetchall()
        paid_cap = getattr(args, "max_paid_requests", 0)
        if paid_cap and queue.paid_attempts() >= paid_cap:
            jobs = [job for job in jobs if job["tool"] not in ANALYTICS]
        # Finish free provenance first so paid requests use the best available lower bounds.
        discovery = [job for job in jobs if job["tool"] not in ANALYTICS]
        if discovery:
            jobs = discovery
        if not jobs or (
            args.max_requests and budget["sent"] >= args.max_requests
        ):
            return
        now = time.time()
        due = [
            (max(job["not_before"] - now, queue.delay(job["tool"], now)), job)
            for job in jobs
        ]
        ready = [(delay, job) for delay, job in due if delay <= 0]
        if not ready:
            wait = min(delay for delay, _ in due)
            LOG.info(
                "Waiting %.0fs for rate-limit/cooldown window. %s",
                wait,
                queue.status(),
            )
            await asyncio.sleep(min(wait, 60))
            continue
        job = ready[0][1]
        if job["tool"] not in available:
            queue.finish(job, "unsupported_tool")
            continue
        validate(json.loads(job["args"]), available[job["tool"]]["inputSchema"])
        queue.reserve(job)
        budget["active_id"] = job["id"]
        budget["sent"] += 1
        LOG.info("Request %s: %s %s", budget["sent"], job["tool"], job["id"])
        try:
            result = await session.call_tool(
                job["tool"], json.loads(job["args"])
            )
            response = result.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        except Exception as exc:
            failure = http_failure(exc)
            if failure and failure["http_status"] == 429:
                response = failure
            else:
                queue.finish(
                    job,
                    "uncertain",
                    note=type(exc).__name__
                    + ": inspect before resubmitting a possibly billed request",
                )
                raise
        path = save_response(queue, job, response)
        status, retry = classify(response)
        status = validate_scope(job, response, status)
        if status == "rate_limited":
            attempts = queue.db.execute(
                "SELECT attempts FROM jobs WHERE id=?", (job["id"],)
            ).fetchone()[0]
            backoff = min(900, 15 * 2 ** min(attempts - 1, 6))
            wait = max(retry or 0, backoff) + 1
            queue.defer(job, wait)
            budget["active_id"] = None
            LOG.info(
                "Rate limited; retrying the same tool/task after at least %.0fs.",
                wait,
            )
            if response.get("http_status") == 429:
                return "reconnect"  # Transport may be closed; retain the same tool/task and cooldown.
            continue
        if status == "quota_blocked":
            queue.finish(job, status, path)
            budget["active_id"] = None
            LOG.error(
                "Account quota/balance limit; stopping. No automated purchases or tool switching."
            )
            return
        details = json.loads(job["details"])
        if status not in ("error", "scope_mismatch"):
            override = build_children(
                queue, job, response, args.fallback_start, args.end, args.start
            )
            status = override or status
        if details["kind"] != "history" and status not in (
            "error",
            "quota_blocked",
        ):
            status = "discovered"
        queue.finish(job, status, path)
        budget["active_id"] = None
        LOG.info("%s: %s", job["id"], status)


async def online(args, queue):
    if args.command == "login":
        async with connect(args, interactive=True) as session:
            available = await inspect_tools(session, queue)
            print(
                f"Authenticated. Inspected {len(available)} MCP tools. No analytical queries submitted."
            )
        return
    if args.command == "inspect":
        async with connect(args) as session:
            print(
                json.dumps(list(await inspect_tools(session, queue)), indent=2)
            )
        return
    # Exclusive ownership across processes; rate ledger and job claims survive restarts.
    lock = (queue.output / ".runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(
            "Another downloader owns this output directory."
        ) from None
    with lock:
        queue.recover_interrupted()
        budget = {"sent": 0}
        while True:
            connection_started = time.time()
            try:
                async with connect(
                    args, interactive=args.command == "start"
                ) as session:
                    outcome = await work(session, queue, args, budget)
            except Exception as exc:
                failure = http_failure(exc)
                if not failure or failure["http_status"] != 429:
                    raise
                observed = getattr(args, "_http_rejection", {})
                if observed.get("at", 0) < connection_started:
                    observed = {}
                wait = recover_transport_limit(
                    queue, budget, {**failure, **observed}
                )
                if wait is None:
                    LOG.error(
                        "Paid request outcome is uncertain; stopped to avoid duplicate billing."
                    )
                    break
                LOG.info(
                    "HTTP 429: retrying the rejected task after at least %.0fs.",
                    wait,
                )
                until = time.time() + wait
                while time.time() < until:
                    await asyncio.sleep(min(until - time.time(), 60))
                outcome = "reconnect"
            if outcome != "reconnect" or (
                args.max_requests and budget["sent"] >= args.max_requests
            ):
                break
            # Reconnect only after the server's persisted cooldown; no tight handshake loop.
            while True:
                wait = queue.delay("text_to_insight")
                if wait <= 0:
                    break
                LOG.info("Transport rate limited; reconnect in %.0fs.", wait)
                await asyncio.sleep(min(wait, 60))
        print(json.dumps(queue.status(), indent=2))


SCRIPT_SETTINGS = load_config("_utility_schemas")["carbonarc_download"]


def main():

    args = load_script_settings("carbonarc_download", SCRIPT_SETTINGS, None)
    catalog_mode = args.command.startswith("catalog-")
    args.output = args.output or (
        CATALOG_OUTPUT if catalog_mode else DEFAULT_OUTPUT
    )
    args.seed = args.seed or (
        CATALOG_OUTPUT / "catalog.json"
        if catalog_mode
        else DEFAULT_OUTPUT / "discovery.json"
    )
    if catalog_mode:
        args.discovery_only = True  # No numerical jobs or paid tools in the full-catalog inventory.
    for name in ("start", "fallback_start", "end"):
        if getattr(args, name):
            dt.date.fromisoformat(getattr(args, name))
    if (
        (args.start or args.fallback_start) > args.end
        or args.max_requests < 0
        or args.max_paid_requests < 0
    ):
        raise ValueError("Invalid date interval or negative request limit")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    queue = Queue(args.output)
    if args.command == "catalog-prepare":
        count = prepare_catalog(queue, args.seed)
        print(
            f"Prepared free provenance discovery for {count} datasets; no company filter or paid queries. {queue.status()}"
        )
    elif args.command == "prepare":
        metrics = prepare(queue, args.seed)
        print(
            f"Prepared {len(metrics)} numerical insight discovery jobs plus event discovery. {queue.status()}"
        )
    elif args.command in ("status", "catalog-status"):
        print(json.dumps(queue.status(), indent=2))
    else:
        if args.command == "start":
            prepare(queue, args.seed)
        try:
            asyncio.run(online(args, queue))
        except KeyboardInterrupt:
            print(
                "Stopped. Pending work and rate ledger are preserved; any in-flight call requires review."
            )
        except Exception as exc:
            # SDK exceptions may contain OAuth URLs or server content. Never print raw exception text.
            LOG.error(
                "Stopped (%s). Check saved responses; run login if authentication is missing/expired.",
                type(exc).__name__,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
