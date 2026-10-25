"""Explicit serving profiles for the strategy's proposed LLM roles.

These are contemporary research trials, not historical checkpoint admission.
Providers are tool-less, never substitute a model, and include serving options
in their cache identity. See spec 20.15, 20.18-20.20.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

from pydantic import Field, model_validator

from ..contracts.base import StrictModel
from .provider import (
    DEFAULT_MAX_RESPONSE_BYTES, IncompleteOutputError, LLMRequest, LLMResponse,
    OpenAICompatibleProvider, ProviderError, Transport, parse_usage,
    redact_secrets, sha256_bytes, urllib_transport, utc_now, _NoRedirect,
)

MODEL_IDS = {
    'lead': 'Qwen/Qwen3-30B-A3B-Thinking-2507',
    'challenger': 'openai/gpt-oss-20b',
    'extractor': 'Qwen/Qwen3-8B',
    'lead_upgrade': 'Qwen/Qwen3.6-35B-A3B',
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class ServingProfile(StrictModel):
    role: Literal['lead', 'challenger', 'extractor', 'lead_upgrade']
    repository: str
    protocol: Literal['openai_compatible', 'ollama']
    base_url: str
    served_model: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, pattern=r'^[A-Za-z_][A-Za-z0-9_]*$')
    # Opaque server alias is explicitly mapped by the operator, never inferred.
    alias_attestation: str = Field(min_length=10)
    revision: str | None = Field(default=None, pattern=r'^[0-9a-f]{40,64}$')
    weight_hashes: dict[str, str] = Field(default_factory=dict)
    quantization: str = 'unverified'
    # Required for local Ollama trials: detect mutable tag changes before calls.
    ollama_digest: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    generation: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_profile(self):
        if self.repository != MODEL_IDS[self.role]:
            raise ValueError('profile must use the exact repository specified for its role')
        u = urlparse(self.base_url)
        if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError('base_url must be an explicit endpoint without embedded credentials')
        local = u.hostname in ('127.0.0.1', 'localhost', '::1')
        if not local and (u.scheme != 'https' or not self.api_key_env):
            raise ValueError('remote endpoints require HTTPS and a named key environment variable')
        allowed = {'top_p', 'top_k', 'min_p', 'repetition_penalty', 'num_ctx', 'num_thread', 'num_batch', 'num_gpu'}
        if set(self.generation) - allowed:
            raise ValueError('unreviewed generation option')
        for k, v in self.generation.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError('generation options must be finite numbers')
            if not __import__('math').isfinite(v):
                raise ValueError('generation options must be finite numbers')
            if k in {'num_ctx', 'num_thread', 'num_batch', 'num_gpu', 'top_k'} and (not isinstance(v, int) or v < 0):
                raise ValueError(f'{k} must be a nonnegative integer')
        if not 512 <= self.generation.get('num_ctx', 2048) <= 32768:
            raise ValueError('trial context must be between 512 and 32768 tokens')
        if self.protocol == 'openai_compatible' and any(k.startswith('num_') for k in self.generation):
            raise ValueError('num_* options are Ollama-only; configure compatible server context separately')
        if self.protocol == 'ollama' and not self.ollama_digest:
            raise ValueError('Ollama tags must be pinned to an observed manifest digest')
        if self.role == 'lead_upgrade' and self.protocol != 'openai_compatible':
            raise ValueError('upgrade trial requires an explicitly configured compatible server')
        for value in self.weight_hashes.values():
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('weight hashes must be SHA256 values')
        return self

    @property
    def identity(self):
        return digest(self.model_dump(mode='json'))


def load_profiles(path: str | Path) -> dict[str, ServingProfile]:
    raw = json.loads(Path(path).read_text())
    if raw.get('schema_version') != 'strategy_serving.v1':
        raise ValueError('unknown serving profile schema')
    result = {}
    for row in raw['profiles']:
        profile = ServingProfile.model_validate(row)
        if profile.role in result:
            raise ValueError('duplicate serving role')
        result[profile.role] = profile
    return result


class StrategyModelProvider:
    """One exact explicitly mapped role; no automatic model/provider fallback."""
    is_mock = False

    def __init__(self, profile: ServingProfile, *, transport: Transport | None = None, environ=None, metadata_get=None):
        self.profile = profile
        self.model = profile.served_model
        self.name = f'strategy_{profile.role}_{profile.identity[:20]}'
        self.transport = transport or urllib_transport
        self.metadata_get = metadata_get or self._metadata_get
        # Reuse audited endpoint/key validation and response parser.
        self.http = OpenAICompatibleProvider(profile.base_url, self.model, profile.api_key_env,
            transport=self._compatible_transport, environ=environ, json_mode=False)

    def _compatible_transport(self, url, headers, body, timeout, max_bytes):
        payload = json.loads(body)
        p = self.profile
        generation = {k: v for k, v in p.generation.items() if not k.startswith('num_')}
        payload.update(generation)
        if p.role == 'extractor':
            payload['chat_template_kwargs'] = {'enable_thinking': False}
        elif p.role == 'challenger':
            payload['reasoning_effort'] = 'medium'
        # The July thinking-only checkpoint uses its native reasoning template.
        # Tool calling is deliberately absent even when the server supports it.
        status, body = self.transport(url, headers, json.dumps(payload).encode(), timeout, max_bytes)
        if status == 200:
            try:
                message = json.loads(body)['choices'][0]['message']
            except (ValueError, KeyError, IndexError, TypeError):
                message = {}  # The audited base parser reports malformed completions.
            if message.get('tool_calls'):
                raise ProviderError('tool execution is unavailable in strategy model trials')
            if p.role == 'extractor' and (message.get('reasoning_content') or message.get('reasoning')):
                raise ProviderError('extractor backend returned reasoning despite non-thinking mode')
        return status, body

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self.profile.protocol == 'openai_compatible':
            response = self.http.complete(request)
            if response.served_model != self.model:
                raise ProviderError('served model differs from the explicitly configured alias')
            return dataclasses.replace(response, provider=self.name)
        return self._ollama(request)

    def _post(self, suffix, payload, timeout):
        key = self.http._api_key()
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if key:
            headers['Authorization'] = f'Bearer {key}'
        try:
            status, body = self.transport(self.profile.base_url.rstrip('/') + suffix, headers,
                json.dumps(payload).encode(), timeout, DEFAULT_MAX_RESPONSE_BYTES)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f'provider transport failed: {type(exc).__name__}') from None
        if not isinstance(body, bytes) or len(body) > DEFAULT_MAX_RESPONSE_BYTES:
            raise ProviderError('invalid or oversized Ollama response')
        if status != 200:
            raise ProviderError(f'Ollama HTTP {status}: {redact_secrets(body[:200].decode(errors="replace"), key)}')
        try:
            result = json.loads(body)
        except ValueError:
            raise ProviderError('invalid Ollama JSON response') from None
        if not isinstance(result, dict):
            raise ProviderError('invalid Ollama response object')
        return result, body

    def _metadata_get(self):
        key = self.http._api_key()
        headers = {'Accept': 'application/json'}
        if key:
            headers['Authorization'] = f'Bearer {key}'
        request = urllib.request.Request(self.profile.base_url.rstrip('/')+'/api/tags', headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
        try:
            with opener.open(request, timeout=10) as response:
                raw = response.read(DEFAULT_MAX_RESPONSE_BYTES+1)
            if len(raw) > DEFAULT_MAX_RESPONSE_BYTES:
                raise ProviderError('oversized Ollama tag listing')
            return json.loads(raw)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f'Ollama tag metadata unavailable: {type(exc).__name__}') from None

    def _verify_tag(self):
        tags = self.metadata_get()
        matches = [m for m in tags.get('models', []) if m.get('name') == self.model]
        if len(matches) != 1 or matches[0].get('digest') != self.profile.ollama_digest:
            raise ProviderError('Ollama tag differs from the pinned manifest digest')

    def _ollama(self, request):
        import time
        p = self.profile
        started = time.monotonic()
        self._verify_tag()
        # Show is metadata only. A mutable local tag is checked before inference.
        show, _ = self._post('/api/show', {'model': self.model}, min(15, request.params.timeout_s))
        # Tags are checked again afterward to detect a concurrent tag change.
        if show.get('details', {}).get('family') != ('gptoss' if p.role == 'challenger' else 'qwen3'):
            raise ProviderError('local model architecture differs from the requested role')
        options = {**p.generation, 'temperature': request.params.temperature, 'num_predict': request.params.max_tokens}
        if request.params.seed is not None:
            options['seed'] = request.params.seed
        think = False if p.role == 'extractor' else ('medium' if p.role == 'challenger' else True)
        doc, body = self._post('/api/chat', {'model': self.model,
            'messages': [{'role': 'system', 'content': request.system}, {'role': 'user', 'content': request.user}],
            'think': think, 'stream': False,
            'format': (request.response_schema or 'json') if p.role == 'extractor' else None,
            'options': options, 'keep_alive': 0}, request.params.timeout_s)
        self._verify_tag()
        if doc.get('done') is not True or doc.get('done_reason') != 'stop':
            raise IncompleteOutputError('Ollama did not finish cleanly', response_hash=sha256_bytes(body))
        if doc.get('model') != self.model:
            raise ProviderError('Ollama served a different model alias')
        message = doc.get('message', {})
        if message.get('tool_calls') or (p.role == 'extractor' and message.get('thinking')):
            raise ProviderError('tool calls or unexpected extractor reasoning were returned')
        content = message.get('content')
        if not isinstance(content, str) or not content.strip():
            raise ProviderError('Ollama returned empty final content')
        prompt, completion, total = parse_usage({'prompt_tokens': doc.get('prompt_eval_count'),
            'completion_tokens': doc.get('eval_count')})
        return LLMResponse(text=content, provider=self.name, model=self.model, served_model=doc['model'],
            response_hash=sha256_bytes(body), received_at=utc_now(), latency_s=time.monotonic()-started,
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
