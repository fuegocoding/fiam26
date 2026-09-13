"""Minimal transport for any OpenAI-compatible Chat Completions endpoint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 120
    temperature: float = 0.0
    max_output_tokens: int = 1600
    response_format: str = "none"


@dataclass(frozen=True)
class LLMResult:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    request_id: str | None
    raw: Any


class CompatibleLLM:
    """Use only broadly compatible chat arguments; provider features stay optional."""

    def __init__(self, config: LLMConfig, client: Any | None = None):
        self.config = config
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError("install the openai package to call an LLM endpoint") from exc
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_s,
            )

    def complete_json(self, system: str, user: str, request_id: str) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        if self.config.response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if request_id:
            kwargs["extra_headers"] = {"X-Client-Request-Id": request_id}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "response_format" not in kwargs or "response_format" not in str(exc).lower():
                raise
            kwargs.pop("response_format")
            response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = getattr(choice.message, "content", None) or ""
        usage = getattr(response, "usage", None)
        return LLMResult(
            content=content,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            request_id=getattr(response, "_request_id", None)
            or getattr(response, "request_id", None),
            raw=response,
        )
