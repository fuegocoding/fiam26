"""Minimal transport for any OpenAI-compatible Chat Completions endpoint."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


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

    def complete_json_with_tools(
        self,
        system: str,
        user: str,
        request_id: str,
        tools: list[dict[str, Any]],
        dispatch: Callable[[str, dict[str, Any]], dict[str, Any]],
        max_tool_calls: int = 4,
    ) -> LLMResult:
        """Complete a JSON verdict while allowing bounded local tool calls.

        Tools are deliberately supplied by the caller. The provider only sees the
        redacted conversation and tool results; this method has no web or filesystem
        access of its own.
        """
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt_tokens = 0
        completion_tokens = 0
        saw_usage = False
        for _ in range(max_tool_calls + 1):
            kwargs: dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
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
            usage = getattr(response, "usage", None)
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            if prompt is not None:
                prompt_tokens += prompt
                saw_usage = True
            if completion is not None:
                completion_tokens += completion
                saw_usage = True

            message = response.choices[0].message
            calls = getattr(message, "tool_calls", None) or []
            if not calls:
                return LLMResult(
                    content=getattr(message, "content", None) or "",
                    prompt_tokens=prompt_tokens if saw_usage else None,
                    completion_tokens=completion_tokens if saw_usage else None,
                    request_id=getattr(response, "_request_id", None)
                    or getattr(response, "request_id", None),
                    raw=response,
                )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": getattr(message, "content", None),
                "tool_calls": [],
            }
            call_ids: list[str] = []
            for call in calls:
                function = getattr(call, "function", None)
                name = getattr(function, "name", None)
                raw_arguments = getattr(function, "arguments", "{}")
                call_id = getattr(call, "id", None) or f"tool-call-{len(messages)}"
                call_ids.append(call_id)
                if not name:
                    raise ValueError("provider returned a tool call without a function name")
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid arguments for tool {name!r}") from exc
                if not isinstance(arguments, dict):
                    raise ValueError(f"tool arguments for {name!r} must be a JSON object")
                assistant_message["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": raw_arguments},
                    }
                )
            messages.append(assistant_message)
            for call, call_id in zip(calls, call_ids):
                function = getattr(call, "function")
                arguments = json.loads(getattr(function, "arguments", "{}") or "{}")
                try:
                    result = dispatch(getattr(function, "name"), arguments)
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
                    }
                )

        raise RuntimeError("LLM exceeded the maximum number of tool calls without a final verdict")
