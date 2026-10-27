"""Construct paced model providers with transient-error retries."""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ..agents.provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    OpenAICompatibleProvider,
    ClaudeCodeProvider,
    ProviderError,
    urllib_transport,
    utc_now,
)
from .configuration import config_path
from ..agents.roles import RoleProviders

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_named_env(path: str | Path | None, names: set[str]) -> dict[str, str]:
    """Read named key variables from dotenv without executing a shell."""
    if path is None or not names:
        return {}
    source = config_path(path)
    if not source.is_file():
        raise ValueError("configured model env_file does not exist")
    values: dict[str, str] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        key, separator, value = text.partition("=")
        key = key.strip()
        if not separator or not _ENV_NAME.fullmatch(key):
            raise ValueError(
                f"invalid model env_file entry on line {line_number}"
            )
        if key not in names:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key] = value
    return values


class _PacedTransport:
    """Apply a minimum start-to-start delay to one provider's HTTP calls."""

    def __init__(self, interval: float):
        if interval < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        self.interval = interval
        self._lock = threading.Lock()
        self._last_start: float | None = None

    def __call__(self, url, headers, body, timeout_s, max_bytes):
        with self._lock:
            now = time.monotonic()
            if self._last_start is not None:
                delay = self.interval - (now - self._last_start)
                if delay > 0:
                    time.sleep(delay)
            self._last_start = time.monotonic()
        return urllib_transport(url, headers, body, timeout_s, max_bytes)


class _UnconfiguredProvider:
    """Keep disabled roles explicit without substituting a mock model."""

    name = "unconfigured"
    is_mock = False

    def __init__(self, role: str):
        self.model = role

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise ProviderError("Online model providers are disabled.")


class _RetryingProvider:
    """Retry transient provider failures without imposing spending quotas."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        retry_attempts: int = 0,
        retry_delay_seconds: float = 30,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.provider = provider
        self.calls = 0
        self.name = provider.name
        self.model = provider.model
        self.is_mock = provider.is_mock
        if not 0 <= retry_attempts <= 3 or not 0 <= retry_delay_seconds <= 60:
            raise ValueError("Bounded provider retry policy required")
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep = sleep

    def complete(self, request: LLMRequest) -> LLMResponse:
        for attempt in range(self.retry_attempts + 1):
            self.calls += 1
            try:
                response = self.provider.complete(request)
            except BaseException as exc:
                if (
                    not isinstance(exc, ProviderError)
                    or not exc.retryable
                    or attempt == self.retry_attempts
                ):
                    raise
                self.sleep(min(60, self.retry_delay_seconds * 2**attempt))
                continue
            return response
        raise AssertionError("Provider retry loop exhausted without a result")


def create_live_models(
    settings: Mapping[str, Any],
    clock: Callable[[], dt.datetime] = utc_now,
) -> RoleProviders:
    """Build explicit providers; credentials remain in private mappings."""
    roles = settings.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("models.roles must be an object")
    for role_name in ("lead", "challenger", "extractor"):
        if not isinstance(roles.get(role_name), Mapping):
            raise ValueError(f"models.roles.{role_name} must be configured")
    key_names = {
        str(row["api_key_env"])
        for row in roles.values()
        if isinstance(row, Mapping) and row.get("api_key_env")
    }
    if any(not _ENV_NAME.fullmatch(name) for name in key_names):
        raise ValueError("api_key_env must be a named environment variable")
    file_values = _load_named_env(settings.get("env_file"), key_names)
    private_environment = {
        name: os.environ.get(name, file_values.get(name, ""))
        for name in key_names
    }
    interval = float(settings.get("min_interval_seconds", 0))
    if not math.isfinite(interval) or not 0 <= interval <= 60:
        raise ValueError("min_interval_seconds must be finite and in [0, 60]")
    online = bool(settings.get("online", False))
    providers: dict[str, Any] = {}
    paced_transport = _PacedTransport(interval)
    for role_name in ("lead", "challenger", "extractor"):
        role = roles.get(role_name)
        assert isinstance(role, Mapping)
        if not online:
            provider = _UnconfiguredProvider(role_name)
        elif role.get("provider") == "claude_code":
            provider = ClaudeCodeProvider(
                model=role.get("model"),
                executable=str(role.get("executable", "claude")),
                timeout_s=float(settings.get("generation", {}).get("timeout_s", 180)),
                clock=clock,
            )
        else:
            if role.get("provider") != "openai":
                raise ValueError(
                    f"unsupported configured provider for {role_name}"
                )
            api_key_env = role.get("api_key_env")
            provider = OpenAICompatibleProvider(
                str(role.get("base_url", "")),
                str(role.get("model", "")),
                str(api_key_env) if api_key_env else None,
                environ=private_environment,
                transport=paced_transport,
                structured_outputs=bool(role.get("structured_outputs", False)),
                reasoning_effort=role.get("reasoning_effort"),
                clock=clock,
            )
        providers[role_name] = _RetryingProvider(
            provider,
            retry_attempts=int(settings.get("retry_attempts", 0)),
            retry_delay_seconds=float(settings.get("retry_delay_seconds", 30)),
        )
    result = RoleProviders(
        lead=providers["lead"],
        challenger=providers["challenger"],
        extractor=providers["extractor"],
    )
    result.identities()
    return result
