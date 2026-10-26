"""Configuration, global budget and strict structured-output provider tests."""

import datetime as dt
import json
import os

import pytest

from pod_platform.agents.budgets import PersistentCallBudget
from pod_platform.agents.provider import (
    BudgetExhausted,
    GenerationParams,
    LLMRequest,
    LLMResponse,
    OpenAICompatibleProvider,
    ProviderError,
    strict_json_schema,
)
from pod_platform.application.live_models import (
    _BudgetedProvider,
    create_live_models,
)
from pod_platform.papers.components import ComponentRequest
from pod_platform.research.composition import CompositionSpec

NOW = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _settings(env_file, *, max_calls=24):
    return {
        "online": True,
        "global_max_calls": max_calls,
        "global_max_tokens": 100_000,
        "env_file": str(env_file),
        "min_interval_seconds": 0,
        "roles": {
            role: {
                "provider": "openai",
                "model": f"model-{role}",
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key_env": "LIVE_TEST_KEY",
                "structured_outputs": role == "lead",
                "reasoning_effort": "low" if role == "lead" else None,
            }
            for role in ("lead", "challenger", "extractor")
        },
    }


def _request():
    return LLMRequest(
        system="system",
        user="user",
        params=GenerationParams(max_tokens=10),
        role="lead",
    )


class StubProvider:
    name = "stub"
    model = "stub-model"
    is_mock = False

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def complete(self, request):
        if self.error:
            raise self.error
        return self.response


def _response(total=3):
    return LLMResponse(
        text='{"ok":true}',
        provider="stub",
        model="stub-model",
        response_hash="a" * 64,
        received_at=NOW,
        latency_s=0,
        prompt_tokens=1,
        completion_tokens=2,
        total_tokens=total,
    )


def test_transient_retries_wait_and_charge_each_actual_provider_call(tmp_path):
    class Transient:
        name, model, is_mock = "transient", "model", False

        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            if self.calls < 3:
                raise ProviderError("HTTP 429", retryable=True)
            return _response()

    budget = PersistentCallBudget(
        tmp_path / "retries.sqlite", budget_id="retry", pod="*",
        cycle="test", max_calls=3, max_tokens=10_000, clock=lambda: NOW,
    )
    delays = []
    provider = _BudgetedProvider(
        Transient(), budget, retry_attempts=2, retry_delay_seconds=20,
        sleep=delays.append,
    )
    assert provider.complete(_request()).total_tokens == 3
    assert delays == [20, 40]
    assert budget.snapshot()["calls_used"] == 3


def test_nonretryable_provider_failure_does_not_spend_retry_budget(tmp_path):
    budget = PersistentCallBudget(
        tmp_path / "fatal.sqlite", budget_id="fatal", pod="*",
        cycle="test", max_calls=3, max_tokens=10_000, clock=lambda: NOW,
    )
    delays = []
    provider = _BudgetedProvider(
        StubProvider(error=ProviderError("bad credentials")), budget,
        retry_attempts=2, sleep=delays.append,
    )
    with pytest.raises(ProviderError, match="credentials"):
        provider.complete(_request())
    assert not delays
    assert budget.snapshot()["calls_used"] == 1


def test_global_budget_is_shared_across_roles_and_survives_restart(tmp_path):
    env_file = tmp_path / "models.env"
    env_file.write_text("LIVE_TEST_KEY='abc123-secret-value'\n")
    settings = _settings(env_file, max_calls=2)
    roles, budget = create_live_models(settings, tmp_path / "state", "run_a")

    for role in ("lead", "challenger"):
        wrapped = getattr(roles, role)
        wrapped.provider = StubProvider(response=_response())
        wrapped.complete(_request())
    assert budget.snapshot()["calls_used"] == 2
    assert budget.snapshot()["tokens_used"] == 6

    restarted, reopened = create_live_models(
        settings, tmp_path / "state", "run_a"
    )
    wrapped = restarted.extractor
    wrapped.provider = StubProvider(response=_response())
    with pytest.raises(BudgetExhausted):
        wrapped.complete(_request())
    assert reopened.snapshot()["calls_used"] == 2


def test_configured_budget_ledger_is_shared_across_state_paths(tmp_path):
    env_file = tmp_path / "models.env"
    env_file.write_text("LIVE_TEST_KEY='abc123-secret-value'\n")
    settings = {
        **_settings(env_file, max_calls=2),
        "budget_path": str(tmp_path / "shared" / "global.sqlite"),
        "budget_id": "initial_research_v1",
    }
    first_roles, first_budget = create_live_models(
        settings, tmp_path / "state_a", "run_a"
    )
    second_roles, second_budget = create_live_models(
        settings, tmp_path / "state_b", "run_b"
    )
    first_roles.lead.provider = StubProvider(response=_response())
    second_roles.challenger.provider = StubProvider(response=_response())

    first_roles.lead.complete(_request())
    second_roles.challenger.complete(_request())
    assert first_budget.snapshot()["calls_used"] == 2
    assert second_budget.snapshot()["calls_used"] == 2
    assert first_budget.path == second_budget.path

    third_roles, _ = create_live_models(settings, tmp_path / "state_c", "run_c")
    third_roles.extractor.provider = StubProvider(response=_response())
    with pytest.raises(BudgetExhausted):
        third_roles.extractor.complete(_request())


def test_failed_provider_call_keeps_reservation_charged(tmp_path):
    budget = PersistentCallBudget(
        tmp_path / "state" / "global_llm_budget.sqlite",
        budget_id="global_llm",
        pod="*",
        cycle="run_b",
        max_calls=3,
        max_tokens=10_000,
    )
    wrapped = _BudgetedProvider(
        StubProvider(error=ProviderError("simulated transport failure")), budget
    )

    with pytest.raises(ProviderError, match="simulated transport failure"):
        wrapped.complete(_request())
    snapshot = budget.snapshot()
    assert snapshot["calls_used"] == 1
    assert snapshot["tokens_used"] > 0


def test_env_file_loads_only_named_variables_without_mutating_process_env(
    tmp_path, monkeypatch
):
    env_file = tmp_path / "models.env"
    env_file.write_text(
        "LIVE_TEST_KEY='abc123-secret-value'\nUNUSED_SECRET=do-not-load\n"
    )
    monkeypatch.delenv("LIVE_TEST_KEY", raising=False)

    roles, _ = create_live_models(
        _settings(env_file), tmp_path / "state", "run_c"
    )
    lead = roles.lead.provider
    assert lead._environ == {"LIVE_TEST_KEY": "abc123-secret-value"}
    assert "LIVE_TEST_KEY" not in os.environ
    assert "UNUSED_SECRET" not in lead._environ
    transports = [
        getattr(roles, role).provider._transport
        for role in ("lead", "challenger", "extractor")
    ]
    assert transports[0] is transports[1] is transports[2]


def test_strict_schema_payload_and_reasoning_effort_are_transformed():
    sent = {}

    def transport(url, headers, body, timeout_s, max_bytes):
        sent.update(json.loads(body))
        answer = {
            "model": "groq-model",
            "choices": [
                {"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        }
        return 200, json.dumps(answer).encode()

    provider = OpenAICompatibleProvider(
        "http://127.0.0.1:8000/v1",
        "groq-model",
        None,
        transport=transport,
        structured_outputs=True,
        reasoning_effort="low",
    )
    request = LLMRequest(
        system="system",
        user="user",
        params=GenerationParams(max_tokens=10),
        response_schema={
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "meta": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                },
            },
        },
    )
    response = provider.complete(request)

    schema = sent["response_format"]["json_schema"]
    assert sent["reasoning_effort"] == "low"
    assert schema["strict"] is True
    assert schema["schema"]["required"] == ["meta", "ok"]
    assert schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["properties"]["meta"]["required"] == ["note"]
    assert response.total_tokens == 5


def test_config_requires_all_three_explicit_roles(tmp_path):
    settings = _settings(tmp_path / "models.env")
    del settings["roles"]["extractor"]
    with pytest.raises(ValueError, match="extractor"):
        create_live_models(settings, tmp_path / "state", "run_d")


def test_composition_schema_keeps_semantics_and_normalizes_groq_subset():
    original = CompositionSpec.model_json_schema()
    schema = strict_json_schema(original)
    causal_edges = schema["properties"]["causal_edges"]

    assert causal_edges["type"] == "array"
    causal_tuple = causal_edges["items"]
    assert causal_tuple["type"] == "array"
    assert causal_tuple["items"] == {"type": "string"}
    assert "prefixItems" not in causal_tuple
    assert (
        schema["properties"]["reasoning_mode"]["enum"]
        == original["properties"]["reasoning_mode"]["enum"]
    )
    assert schema["properties"]["nodes"]["items"]["$ref"].startswith("#/$defs/")
    assert "anyOf" in schema["properties"]["own_reasoning"]

    unsupported = {
        "format",
        "maxItems",
        "maxLength",
        "minItems",
        "minLength",
        "pattern",
        "prefixItems",
    }

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            assert not unsupported.intersection(node)
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                if node.get("properties"):
                    assert node["required"] == sorted(node["properties"])
            for value in node.values():
                visit(value)

    visit(schema)

    component_schema = strict_json_schema(ComponentRequest.model_json_schema())
    label = component_schema["properties"]["label"]
    assert label["type"] == "string"
    assert label["enum"] == ["component_only"]


def test_dropped_constraints_remain_visible_as_description_text():
    schema = strict_json_schema(CompositionSpec.model_json_schema())
    props = schema["properties"]

    assert "minLength 10" in props["applicability"]["description"]
    assert "minItems 1" in props["competing_explanations"]["description"]
    assert "minimum 1" in props["variant_budget"]["description"]
    assert "maximum 32" in props["variant_budget"]["description"]
    assert "exactly 3 items" in props["causal_edges"]["items"]["description"]
    assert "minItems 1" in props["causal_edges"]["description"]
    lookback = schema["$defs"]["FeatureRecipe"]["properties"]["lookback"]
    assert "minimum 1" in lookback["description"]
    assert "default 1" in lookback["description"]
