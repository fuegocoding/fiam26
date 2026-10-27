"""Carbon Arc OAuth transport used directly by the campaign runtime."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import threading
import time
import tomllib
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken


LOG = logging.getLogger("carbonarc")


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
                    "Carbon Arc authentication is required. Configure the existing OAuth credentials or local/carbonarc-api.env before running the campaign."
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
