"""Offline tests for providers, structured calls, durable cache, budget and temporal gate.

No test calls a billable model: HTTP goes through an injected transport or a
loopback server on 127.0.0.1, and the Claude CLI through an injected runner.
"""

import datetime as dt
import json
import sqlite3
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from pod_platform.agents.durable_cache import (
    CacheCorruptionError,
    CachedStructuredExtractor,
    DurableExtractionCache,
)
from pod_platform.agents.extractor import ExtractionRecord
from pod_platform.agents.provider import (
    CLAUDE_CODE_BASE_ARGS,
    BudgetExhausted,
    CallBudget,
    ClaudeCodeProvider,
    DeterministicMockProvider,
    GenerationParams,
    IncompleteOutputError,
    LLMProvider,
    LLMRequest,
    MalformedOutputError,
    OpenAICompatibleProvider,
    ProviderError,
    ResearchMode,
    authorize_model_use,
    call_structured,
    parse_strict_json,
)
from pod_platform.agents.registry import ModelCheckpoint, ModelRegistry, TemporalPolicy
from pod_platform.contracts.base import StrictModel
from pod_platform.contracts.errors import ErrorCode, PodPlatformError

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
RELEASED = dt.datetime(2025, 3, 1, tzinfo=UTC)


class Answer(StrictModel):
    label: str
    score: float


def clock():
    return NOW


def checkpoint(model_id="mock-1", released=RELEASED, cid="ckpt1"):
    return ModelCheckpoint(
        owner="test",
        created_at=NOW,
        checkpoint_id=cid,
        repository="repo/x",
        exact_model_id=model_id,
        checkpoint_revision="rev1",
        weight_hash="a" * 64,
        release_evidence_url="https://example.invalid/release",
        cutoff_evidence={"checkpoint_released_at": released, "evidence_note": "test"},
        tokenizer_revision="t1",
        chat_template_revision="c1",
        quantization="none",
        serving_engine="test",
        context_limit_verified=8192,
        reasoning_config="none",
        benchmark_artifact_id="b1",
        admitted=True,
    )


def completion(content, finish="stop", usage=(10, 5), model="served-model"):
    return json.dumps(
        {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        }
    ).encode()


def request(user="{}", **params):
    return LLMRequest(system="sys", user=user, params=GenerationParams(**params))


def mock(text='{"label":"a","score":1.0}', model="mock-1"):
    return DeterministicMockProvider(lambda req: text, model=model, clock=clock)


# ---------------------------------------------------------------- HTTP provider


class FakeTransport:
    def __init__(self, status=200, body=None):
        self.status, self.body = status, body if body is not None else completion('{"label":"a","score":1}')
        self.calls = []

    def __call__(self, url, headers, body, timeout_s, max_bytes):
        self.calls.append((url, dict(headers), json.loads(body), timeout_s, max_bytes))
        return self.status, self.body


def http_provider(transport, **kw):
    return OpenAICompatibleProvider(
        "https://llm.example.invalid/v1/",
        "model-x",
        "MY_KEY_VAR",
        transport=transport,
        environ={"MY_KEY_VAR": "sk-secret"},
        clock=clock,
        **kw,
    )


def test_http_request_shape_and_provenance():
    transport = FakeTransport()
    provider = http_provider(transport)
    assert isinstance(provider, LLMProvider)
    response = provider.complete(request(temperature=0.2, max_tokens=256, seed=7, timeout_s=9))
    url, headers, body, timeout, max_bytes = transport.calls[0]
    assert url == "https://llm.example.invalid/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-secret"
    assert body["model"] == "model-x" and body["temperature"] == 0.2
    assert body["max_tokens"] == 256 and body["seed"] == 7 and body["stream"] is False
    assert timeout == 9 and max_bytes > 0
    assert response.text == '{"label":"a","score":1}'
    assert response.response_hash == __import__("hashlib").sha256(transport.body).hexdigest()
    assert (response.prompt_tokens, response.completion_tokens) == (10, 5)
    assert response.served_model == "served-model" and response.received_at == NOW


def test_http_config_is_explicit_and_fail_closed():
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("", "m", "K")
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("http://llm.example.invalid/v1", "m", "K")  # plain http, not loopback
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("https://u:p@llm.example.invalid/v1", "m", "K")
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("https://llm.example.invalid/v1", "m", None)  # key required remotely
    with pytest.raises(ValueError):
        OpenAICompatibleProvider("https://llm.example.invalid/v1", " ", "K")
    OpenAICompatibleProvider("http://127.0.0.1:1/v1", "m", None)  # loopback may be keyless


def test_http_missing_key_never_calls_transport():
    transport = FakeTransport()
    provider = OpenAICompatibleProvider(
        "https://llm.example.invalid/v1", "m", "ABSENT", transport=transport, environ={}
    )
    with pytest.raises(ProviderError, match="ABSENT"):
        provider.complete(request())
    assert transport.calls == []


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (400, False), (401, False)])
def test_http_error_status_and_key_redaction(status, retryable):
    provider = http_provider(FakeTransport(status, b'{"error":"bad key sk-secret"}'))
    with pytest.raises(ProviderError) as info:
        provider.complete(request())
    assert info.value.retryable is retryable
    assert "sk-secret" not in str(info.value)


def test_http_rejects_truncated_empty_and_non_completion_bodies():
    with pytest.raises(IncompleteOutputError):
        http_provider(FakeTransport(body=completion("{}", finish="length"))).complete(request())
    with pytest.raises(ProviderError):
        http_provider(FakeTransport(body=completion("  "))).complete(request())
    with pytest.raises(ProviderError):
        http_provider(FakeTransport(body=b"not json")).complete(request())
    with pytest.raises(ProviderError):
        http_provider(FakeTransport(body=b'{"choices": []}')).complete(request())


def test_http_oversized_body_rejected():
    provider = http_provider(FakeTransport(body=b"x" * 5000), max_response_bytes=2048)
    with pytest.raises(ProviderError, match="limit"):
        provider.complete(request())


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_http_loopback_server_roundtrip_redirect_and_timeout():
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seen["body"] = json.loads(self.rfile.read(length))
            seen["auth"] = self.headers.get("Authorization")
            if self.path.startswith("/redir"):
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:1/steal")
                self.end_headers()
                return
            if self.path.startswith("/slow"):
                time.sleep(1.5)
            data = completion('{"label":"ok","score":2}')
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except OSError:
                pass

    try:
        server = _serve(Handler)
    except OSError:
        pytest.skip("loopback sockets unavailable")
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        env = {"K": "tok"}
        ok = OpenAICompatibleProvider(base + "/v1", "m", "K", environ=env).complete(request())
        assert json.loads(ok.text)["label"] == "ok" and seen["auth"] == "Bearer tok"
        with pytest.raises(ProviderError):  # redirects are never followed
            OpenAICompatibleProvider(base + "/redir", "m", "K", environ=env).complete(request())
        started = time.monotonic()
        with pytest.raises(ProviderError) as info:
            OpenAICompatibleProvider(base + "/slow", "m", "K", environ=env).complete(request(timeout_s=0.3))
        assert info.value.retryable and time.monotonic() - started < 1.4
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------- Claude Code CLI


def fake_runner(stdout=b'{"label":"a","score":1}', returncode=0, raises=None, store=None):
    def run(cmd, **kwargs):
        if store is not None:
            store.append((cmd, kwargs))
        if raises:
            raise raises
        return SimpleNamespace(stdout=stdout, stderr=b"", returncode=returncode)

    return run


def test_claude_code_command_is_locked_down():
    calls = []
    provider = ClaudeCodeProvider("claude-x", runner=fake_runner(store=calls), clock=clock)
    response = provider.complete(request(user='{"q": "data"}'))
    cmd, kwargs = calls[0]
    assert cmd[0] == "claude"
    assert tuple(cmd[1:7]) == CLAUDE_CODE_BASE_ARGS == ("-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--no-session-persistence")
    assert "--model" in cmd and "claude-x" in cmd
    assert not any("dangerously" in part or "permission" in part for part in cmd)
    assert kwargs["shell"] is False and kwargs["timeout"] > 0
    assert b'{"q": "data"}' in kwargs["input"]  # prompt on stdin, not argv
    assert not any("data" in part for part in cmd)
    assert response.provider == "claude_code_cli" and response.model == "claude-x"
    assert provider.is_mock is False


def test_claude_code_failures_are_errors_not_success():
    with pytest.raises(ProviderError, match="status 3"):
        ClaudeCodeProvider(runner=fake_runner(returncode=3)).complete(request())
    with pytest.raises(ProviderError) as info:
        ClaudeCodeProvider(runner=fake_runner(raises=subprocess.TimeoutExpired("claude", 1))).complete(request())
    assert info.value.retryable
    with pytest.raises(ProviderError):
        ClaudeCodeProvider(runner=fake_runner(raises=FileNotFoundError())).complete(request())
    with pytest.raises(ProviderError):
        ClaudeCodeProvider(runner=fake_runner(stdout=b"  ")).complete(request())
    with pytest.raises(ProviderError, match="limit"):
        ClaudeCodeProvider(max_output_bytes=10, runner=fake_runner(stdout=b"x" * 11)).complete(request())


# ---------------------------------------------------------------- strict JSON / structured calls


@pytest.mark.parametrize(
    "text",
    ["", "sure! {}", '{"a": 1} trailing', "[1]", '{"a": NaN}', '{"a": 1, "a": 2}', "```python\n{}\n```", '{"a":'],
)
def test_parse_strict_json_rejects(text):
    with pytest.raises(ValueError):
        parse_strict_json(text)


def test_parse_strict_json_accepts_object_and_single_fence():
    assert parse_strict_json(' {"a": 1} ') == {"a": 1}
    assert parse_strict_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_call_structured_records_full_provenance():
    provider = mock()
    result = call_structured(provider, request(user="in", temperature=0.0), Answer, clock=clock)
    rec = result.record
    assert result.value == Answer(label="a", score=1.0)
    assert rec.provider == "deterministic_mock" and rec.model == "mock-1" and rec.provider_is_mock
    assert len(rec.prompt_hash) == len(rec.input_hash) == len(rec.response_hash) == len(rec.request_hash) == 64
    assert rec.generation_params == {"temperature": 0.0, "max_tokens": 2048, "seed": None}
    assert rec.requested_at.tzinfo is not None and rec.received_at == NOW and rec.outcome == "valid"


@pytest.mark.parametrize(
    "text", ['{"label": "a"}', '{"label": "a", "score": "high"}', '{"label": "a", "score": 1, "extra": 1}', "I think a", "{}"]
)
def test_call_structured_rejects_malformed_and_attaches_record(text):
    with pytest.raises(MalformedOutputError) as info:
        call_structured(mock(text), request(), Answer, clock=clock)
    assert info.value.record.outcome == "invalid" and len(info.value.record.response_hash) == 64


def test_semantic_validator_failure_is_invalid_output():
    def reject(value):
        raise ValueError("fabricated")

    with pytest.raises(MalformedOutputError, match="fabricated"):
        call_structured(mock(), request(), Answer, clock=clock, validator=reject)


def test_budget_exhaustion_blocks_calls_before_provider():
    provider = mock()
    budget = CallBudget(max_calls=1, max_tokens=10**6)
    call_structured(provider, request(), Answer, budget=budget, clock=clock)
    with pytest.raises(BudgetExhausted) as info:
        call_structured(provider, request(), Answer, budget=budget, clock=clock)
    assert info.value.response.error.code == ErrorCode.PROVIDER_BUDGET_EXCEEDED
    assert provider.calls == 1
    tiny = CallBudget(max_calls=5, max_tokens=100)
    with pytest.raises(BudgetExhausted):
        call_structured(provider, request(max_tokens=2048), Answer, budget=tiny, clock=clock)
    assert provider.calls == 1


def test_failed_provider_call_still_consumes_budget():
    def boom(req):
        raise ProviderError("down", retryable=True)

    provider = DeterministicMockProvider(boom, clock=clock)
    budget = CallBudget(max_calls=1, max_tokens=10**6)
    with pytest.raises(ProviderError):
        call_structured(provider, request(), Answer, budget=budget, clock=clock)
    assert budget.snapshot()["calls_used"] == 1
    with pytest.raises(BudgetExhausted):
        call_structured(provider, request(), Answer, budget=budget, clock=clock)


def test_unexpected_provider_crash_is_a_provider_error():
    def crash(req):
        raise RuntimeError("secret internals")

    with pytest.raises(ProviderError) as info:
        call_structured(DeterministicMockProvider(crash, clock=clock), request(), Answer, clock=clock)
    assert "secret" not in str(info.value) and not info.value.retryable


# ---------------------------------------------------------------- durable cache


def extractor(provider, tmp_path, **kw):
    cache = DurableExtractionCache(tmp_path / "cache.sqlite")
    return CachedStructuredExtractor(provider, cache, Answer, "system prompt", clock=clock, **kw), cache


def run(ex, payload=None, **kw):
    args = dict(role="researcher", namespace="pod_a", research_asof=NOW, mode=ResearchMode.CONTEMPORARY)
    args.update(kw)
    return ex.run(payload or {"q": 1}, **args)


def test_cache_replay_does_not_call_provider_again(tmp_path):
    provider = mock()
    ex, cache = extractor(provider, tmp_path)
    first = run(ex)
    second = run(ex)
    assert provider.calls == 1 and not first.from_cache and second.from_cache
    assert second.value == first.value and second.call.response_hash == first.call.response_hash
    # a brand new process/extractor on the same file replays too
    ex2, _ = extractor(provider, tmp_path)
    assert run(ex2).from_cache and provider.calls == 1 and len(cache) == 1


def test_cache_key_separates_input_params_namespace_asof_and_provider(tmp_path):
    provider = mock()
    ex, cache = extractor(provider, tmp_path)
    run(ex)
    run(ex, {"q": 2})
    run(ex, params=GenerationParams(temperature=0.5))
    run(ex, params=GenerationParams(seed=3))
    run(ex, namespace="pod_b")
    run(ex, research_asof=NOW - dt.timedelta(hours=1))
    assert provider.calls == 6 and len(cache) == 6
    other, _ = extractor(mock(model="mock-2"), tmp_path)
    run(other)
    assert len(cache) == 7
    # timeout is not part of identity: this is a replay of the first entry
    assert run(ex, params=GenerationParams(timeout_s=5)).from_cache
    assert provider.calls == 6


def test_invalid_output_is_not_cached_and_not_replayed(tmp_path):
    provider = mock('{"label": "a"}')
    ex, cache = extractor(provider, tmp_path)
    with pytest.raises(MalformedOutputError):
        run(ex)
    assert len(cache) == 0
    with pytest.raises(MalformedOutputError):
        run(ex)
    assert provider.calls == 2  # no silent caching of the failure, but also no hidden retry loop


def test_semantic_validator_rejection_is_not_cached(tmp_path):
    ex, cache = extractor(mock(), tmp_path)

    def reject(value):
        raise ValueError("unsupported citation")

    with pytest.raises(MalformedOutputError):
        run(ex, validator=reject)
    assert len(cache) == 0


def test_cache_is_append_only_and_tamper_evident(tmp_path):
    ex, cache = extractor(mock(), tmp_path)
    result = run(ex)
    conn = sqlite3.connect(cache.path)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE extraction_cache SET fields_json = '{}'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM extraction_cache")
    conn.execute("DROP TRIGGER extraction_cache_no_update")  # simulate an attacker with file access
    conn.execute("UPDATE extraction_cache SET fields_json = '{\"label\":\"evil\",\"score\":9}'")
    conn.commit()
    conn.close()
    with pytest.raises(CacheCorruptionError):
        run(ex)
    assert result.cache_key


def test_cache_put_keeps_first_response(tmp_path):
    cache = DurableExtractionCache(tmp_path / "c.sqlite")
    rec = ExtractionRecord(key="k", fields={"v": 1}, created_at=NOW)
    assert cache.put("k", rec) is True
    assert cache.put("k", ExtractionRecord(key="k", fields={"v": 2}, created_at=NOW)) is False
    assert cache.get("k").fields == {"v": 1}
    with pytest.raises(ValueError):
        cache.put("n", ExtractionRecord(key="n", fields={}, created_at=dt.datetime(2026, 1, 1)))


# ---------------------------------------------------------------- temporal gate


def test_historical_requires_registered_eligible_checkpoint(tmp_path):
    registry = ModelRegistry()
    registry.register(checkpoint())
    provider = mock()
    ex, _ = extractor(provider, tmp_path, registry=registry)
    asof = dt.datetime(2025, 6, 1, tzinfo=UTC)
    ok = run(ex, mode=ResearchMode.HISTORICAL, research_asof=asof, checkpoint_id="ckpt1")
    assert ok.authorization.label == "historical_release_gated" and not ok.authorization.retrospective_only
    for kwargs in (
        dict(checkpoint_id=None),  # no checkpoint
        dict(checkpoint_id="unknown"),  # unregistered
        dict(checkpoint_id="ckpt1", research_asof=dt.datetime(2024, 6, 1, tzinfo=UTC)),  # before release
    ):
        before = provider.calls
        with pytest.raises(PodPlatformError) as info:
            run(ex, mode=ResearchMode.HISTORICAL, **{"research_asof": asof, **kwargs})
        assert info.value.response.error.code == ErrorCode.ASOF_VIOLATION
        assert provider.calls == before


def test_historical_without_registry_and_model_mismatch_fail_closed(tmp_path):
    no_registry, _ = extractor(mock(), tmp_path)
    with pytest.raises(PodPlatformError):
        run(no_registry, mode=ResearchMode.HISTORICAL, research_asof=NOW - dt.timedelta(days=400), checkpoint_id="ckpt1")
    registry = ModelRegistry()
    registry.register(checkpoint(model_id="some-other-model"))
    with pytest.raises(PodPlatformError, match="differs"):
        authorize_model_use(
            mode=ResearchMode.HISTORICAL,
            research_asof=NOW - dt.timedelta(days=100),
            provider=mock(),
            registry=registry,
            checkpoint_id="ckpt1",
            now=NOW,
        )


def test_policy_b_is_disabled_unless_registry_enables_it():
    registry = ModelRegistry()
    registry.register(checkpoint())
    kwargs = dict(
        mode=ResearchMode.HISTORICAL,
        research_asof=NOW - dt.timedelta(days=100),
        provider=mock(),
        registry=registry,
        checkpoint_id="ckpt1",
        now=NOW,
        policy=TemporalPolicy.CUTOFF_QUALIFIED_BACKCAST,
    )
    with pytest.raises(PodPlatformError):
        authorize_model_use(**kwargs)
    enabled = ModelRegistry(allow_cutoff_qualified_backcast=True)
    enabled.register(checkpoint())
    kwargs["registry"] = enabled
    assert authorize_model_use(**kwargs).label == "historical_cutoff_qualified_backcast"


def test_contemporary_is_labelled_retrospective_and_rejects_stale_or_future_dates():
    auth = authorize_model_use(mode=ResearchMode.CONTEMPORARY, research_asof=NOW, provider=mock(), now=NOW)
    assert auth.label == "contemporary_retrospective_only" and auth.retrospective_only
    with pytest.raises(PodPlatformError):  # a past date must use historical mode
        authorize_model_use(
            mode=ResearchMode.CONTEMPORARY, research_asof=NOW - dt.timedelta(days=30), provider=mock(), now=NOW
        )
    with pytest.raises(PodPlatformError):
        authorize_model_use(
            mode=ResearchMode.CONTEMPORARY, research_asof=NOW + dt.timedelta(days=1), provider=mock(), now=NOW
        )
    with pytest.raises(PodPlatformError):  # naive timestamp
        authorize_model_use(
            mode=ResearchMode.CONTEMPORARY, research_asof=dt.datetime(2026, 10, 25), provider=mock(), now=NOW
        )
