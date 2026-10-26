"""Construct paced, globally budgeted model providers from runtime settings."""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from ..agents.budgets import PersistentCallBudget
from ..agents.provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    OpenAICompatibleProvider,
    ProviderError,
    estimate_tokens,
    urllib_transport,
    utc_now,
)
from .configuration import config_path
from .research_loop import RoleProviders

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


class _BudgetedProvider:
    """Charge a shared durable budget before every uncached HTTP completion."""

    def __init__(
        self,
        provider: LLMProvider,
        budget: PersistentCallBudget,
        *,
        retry_attempts: int = 0,
        retry_delay_seconds: float = 30,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.provider = provider
        self.budget = budget
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
            reservation = self.budget.reserve(estimate_tokens(request))
            try:
                response = self.provider.complete(request)
            except BaseException as exc:
                self.budget.settle(reservation, None)
                if (
                    not isinstance(exc, ProviderError)
                    or not exc.retryable
                    or attempt == self.retry_attempts
                ):
                    raise
                self.sleep(min(60, self.retry_delay_seconds * 2**attempt))
                continue
            self.budget.settle(reservation, response.total_tokens)
            return response
        raise AssertionError("Provider retry loop exhausted without a result")


def create_live_models(
    settings: Mapping[str, Any],
    state: str | Path,
    runid: str,
    clock: Callable[[], dt.datetime] = utc_now,
) -> tuple[RoleProviders, PersistentCallBudget]:
    """Build explicit role providers backed by one persistent global budget.

    All roles and pods share one account stored at the configured ``budget_path``
    (or ``state/global_llm_budget.sqlite``). Its cycle defaults to ``runid`` and
    may use a stable configured ``budget_id`` across runs. Environment file
    values stay in private per-provider mappings and are never copied into
    process globals or returned in metadata.
    """
    budget_path = config_path(
        settings.get("budget_path", Path(state) / "global_llm_budget.sqlite")
    )
    budget_cycle = str(settings.get("budget_id", runid))
    budget = PersistentCallBudget(
        budget_path,
        budget_id="global_llm",
        pod="*",
        cycle=budget_cycle,
        max_calls=int(settings.get("global_max_calls", 24)),
        max_tokens=int(settings.get("global_max_tokens", 1_000_000)),
        raise_limits=bool(settings.get("raise_budget_limits", False)),
        clock=clock,
    )
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
        providers[role_name] = _BudgetedProvider(
            provider,
            budget,
            retry_attempts=int(settings.get("retry_attempts", 0)),
            retry_delay_seconds=float(settings.get("retry_delay_seconds", 30)),
        )
    result = RoleProviders(
        lead=providers["lead"],
        challenger=providers["challenger"],
        extractor=providers["extractor"],
    )
    result.identities()
    return result, budget
