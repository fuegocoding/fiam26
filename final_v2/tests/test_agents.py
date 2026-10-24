"""Part 10 acceptance: model registry temporal gating, exception twin,
schema-constrained extraction (spec 20.6-20.10, 20.12-20.14, 20.19)."""

from __future__ import annotations

import datetime as dt
import hashlib

import pytest

from pod_platform.agents import (
    AdoptionRecord,
    AgentResponse,
    AgentTwin,
    CutoffEvidence,
    ExceptionTrigger,
    ExtractorCache,
    ModelCheckpoint,
    ModelRegistry,
    SchemaExtractor,
    TemporalPolicy,
    TriggerKind,
    cache_key,
)
from pod_platform.contracts import ActorKind, ErrorCode, PodPlatformError

UTC = dt.timezone.utc
T0 = dt.datetime(2021, 1, 4, 21, 0, tzinfo=UTC)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _checkpoint(cid: str = "ckpt_1", released=dt.datetime(2020, 6, 1, tzinfo=UTC), admitted=True) -> ModelCheckpoint:
    return ModelCheckpoint(
        owner="integrity",
        created_at=T0,
        checkpoint_id=cid,
        repository="org/model",
        exact_model_id="org/model-v1",
        checkpoint_revision="rev_abc",
        weight_hash=_sha(cid),
        release_evidence_url="https://example.org/release",
        cutoff_evidence=CutoffEvidence(
            pretraining_corpus_boundary="2020-09-01 documented",
            checkpoint_released_at=released,
            evidence_note="model card and release history inspected",
        ),
        tokenizer_revision="tok_v1",
        chat_template_revision="tpl_v1",
        quantization="none",
        serving_engine="vllm_v1",
        context_limit_verified=8192,
        reasoning_config="deliberate",
        benchmark_artifact_id="bench_1",
        admitted=admitted,
    )


def _adoption(cid: str = "ckpt_1", role="lead", **over) -> AdoptionRecord:
    base = dict(
        owner="integrity",
        created_at=T0,
        adoption_id=f"adopt_{cid}_{role}",
        checkpoint_id=cid,
        role=role,
        temporal_policy=TemporalPolicy.RELEASE_GATED,
        effective_start=T0,
    )
    base.update(over)
    return AdoptionRecord(**base)


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(_checkpoint())
    registry.adopt(_adoption())
    return registry


# ------------------------------------------------------- 20.12-20.14, 20.19


def test_release_gated_rejects_future_checkpoint():
    registry = ModelRegistry()
    registry.register(_checkpoint(released=dt.datetime(2022, 1, 1, tzinfo=UTC)))
    assert registry.eligible_at("ckpt_1", dt.datetime(2021, 6, 1, tzinfo=UTC)) is False
    assert registry.eligible_at("ckpt_1", dt.datetime(2022, 6, 1, tzinfo=UTC)) is True


def test_route_fails_without_eligible_checkpoint():
    registry = ModelRegistry()
    registry.register(_checkpoint(released=dt.datetime(2022, 1, 1, tzinfo=UTC)))
    registry.adopt(_adoption())
    with pytest.raises(PodPlatformError) as exc:
        registry.route("lead", dt.datetime(2021, 6, 1, tzinfo=UTC))
    assert exc.value.response.error.code == ErrorCode.ASOF_VIOLATION


def test_route_uses_latest_eligible_adoption():
    registry = _registry()
    registry.register(_checkpoint(cid="ckpt_2"))
    registry.adopt(_adoption(cid="ckpt_2", effective_start=T0 + dt.timedelta(days=30)))
    assert registry.route("lead", dt.datetime(2021, 6, 1, tzinfo=UTC)) == "ckpt_2"


def test_adoption_requires_admitted_checkpoint():
    registry = ModelRegistry()
    registry.register(_checkpoint(admitted=False))
    with pytest.raises(PodPlatformError) as exc:
        registry.adopt(_adoption())
    assert exc.value.response.error.code == ErrorCode.ADMISSION_CERTIFICATE_INVALID


def test_policy_b_disabled_by_default():
    registry = ModelRegistry()
    registry.register(_checkpoint())
    with pytest.raises(PodPlatformError) as exc:
        registry.adopt(
            _adoption(
                temporal_policy=TemporalPolicy.CUTOFF_QUALIFIED_BACKCAST,
                eligibility_certificate_id="cert_b",
            )
        )
    assert exc.value.response.error.code == ErrorCode.ASOF_VIOLATION


def test_policy_b_requires_eligibility_certificate():
    with pytest.raises(ValueError):
        AdoptionRecord(
            owner="integrity",
            created_at=T0,
            adoption_id="adopt_b",
            checkpoint_id="ckpt_1",
            role="lead",
            temporal_policy=TemporalPolicy.CUTOFF_QUALIFIED_BACKCAST,
            effective_start=T0,
        )


def test_policy_b_allowed_when_explicitly_enabled():
    registry = ModelRegistry(allow_cutoff_qualified_backcast=True)
    registry.register(_checkpoint())
    registry.adopt(
        _adoption(
            temporal_policy=TemporalPolicy.CUTOFF_QUALIFIED_BACKCAST,
            eligibility_certificate_id="cert_b",
        )
    )
    assert registry.route("lead", dt.datetime(2021, 6, 1, tzinfo=UTC)) == "ckpt_1"


def test_checkpoint_registry_rejects_duplicates():
    registry = _registry()
    with pytest.raises(ValueError):
        registry.register(_checkpoint())


# ------------------------------------------------------------ 20.7-20.8 twin


def _trigger(kind=TriggerKind.DIFFICULT_TRADE) -> ExceptionTrigger:
    return ExceptionTrigger(kind=kind, context={"security": "sec_1"}, fired_at=T0)


def _twin(agent_fn=None) -> AgentTwin:
    return AgentTwin(
        deterministic_action=lambda trigger: "fixed_rule_resize",
        agent_call=agent_fn or (lambda trigger: None),
    )


def test_agent_receives_exception_triggers_only():
    calls = []

    def agent(trigger):
        calls.append(trigger.kind)
        return AgentResponse(
            ok=True,
            proposed_action="defer_resize",
            rationale="liquidity thin",
            model_ref="model_v1",
            prompt_ref="prompt_v1",
            cache_ref="cache_v1",
        )

    twin = _twin(agent)
    record = twin.handle(_trigger())
    assert record.used_agent is True
    assert record.chosen_action == "defer_resize"
    assert record.twin_action == "fixed_rule_resize"
    assert calls == [TriggerKind.DIFFICULT_TRADE]


def test_malformed_agent_response_falls_back():
    def agent(trigger):
        return AgentResponse(
            ok=False,
            proposed_action="",
            rationale="",
            model_ref="model_v1",
            prompt_ref="prompt_v1",
            cache_ref="cache_v1",
        )

    twin = _twin(agent)
    record = twin.handle(_trigger())
    assert record.used_agent is False
    assert record.chosen_action == "fixed_rule_resize"
    assert record.fallback_reason


def test_agent_exception_never_bypasses_rules():
    def agent(trigger):
        raise RuntimeError("provider exploded")

    twin = _twin(agent)
    record = twin.handle(_trigger())
    assert record.used_agent is False
    assert record.chosen_action == "fixed_rule_resize"


def test_counterfactual_summary_counts_all_periods():
    twin = _twin()
    twin.handle(_trigger())
    twin.handle(_trigger(TriggerKind.NEAR_BINDING_CONSTRAINT))
    summary = twin.counterfactual_summary()
    assert summary["interventions"] == 2
    assert summary["agent_used"] == 0
    assert len(summary["records"]) == 2


def test_agent_decision_event_records_model_refs():
    def agent(trigger):
        return AgentResponse(
            ok=True,
            proposed_action="defer_resize",
            rationale="liquidity thin",
            model_ref="model_v1",
            prompt_ref="prompt_v1",
            cache_ref="cache_v1",
        )

    twin = _twin(agent)
    trigger = _trigger()
    record = twin.handle(trigger)
    response = agent(trigger)
    event = twin.decision_event_from_agent(
        record, response, event_id="ev_1", owner="risk", resulting_artifact_id="art_1"
    )
    assert event.actor_kind == ActorKind.AGENT_ROLE
    assert event.model_ref == "model_v1"
    assert event.prompt_ref == "prompt_v1"
    assert event.cache_ref == "cache_v1"


def test_fixed_rule_decision_event_has_no_agent_refs():
    twin = _twin()
    record = twin.handle(_trigger())
    event = twin.decision_event(record, event_id="ev_2", owner="risk", resulting_artifact_id="art_2")
    assert event.actor_kind == ActorKind.CODE
    assert event.model_ref is None


# --------------------------------------------------------- 20.6, 20.10 cache


def test_extraction_cached_by_full_identity():
    cache = ExtractorCache()
    calls = []

    def model_call(payload, prompt_hash):
        calls.append(payload)
        return {"ticker": "AAA", "event_date": "2021-01-04", "materiality": "high"}

    extractor = SchemaExtractor(
        cache,
        model_call,
        required_fields=("ticker", "event_date", "materiality"),
    )
    kwargs = dict(
        prompt_hash=_sha("prompt"),
        model_id="model_v1",
        checkpoint_revision="rev_abc",
        research_asof=T0,
        generation_params={"temperature": 0.0},
    )
    first = extractor.extract({"text": "filing body"}, **kwargs)
    assert first.from_cache is False
    second = extractor.extract({"text": "filing body"}, **kwargs)
    assert second.from_cache is True
    assert len(calls) == 1
    assert first.fields == second.fields


def test_cache_key_sensitivity():
    base = dict(
        model_id="m",
        checkpoint_revision="r",
        prompt_hash="p",
        input_hashes={"a": "1"},
        research_asof=T0,
        generation_params={"temperature": 0.0},
        role="extractor",
    )
    key1 = cache_key(**base)
    key2 = cache_key(**{**base, "generation_params": {"temperature": 0.7}})
    key3 = cache_key(**{**base, "research_asof": T0 + dt.timedelta(days=1)})
    assert len({key1, key2, key3}) == 3


def test_extraction_schema_validation():
    cache = ExtractorCache()
    extractor = SchemaExtractor(
        cache,
        lambda payload, prompt_hash: {"ticker": "AAA"},
        required_fields=("ticker", "event_date"),
    )
    with pytest.raises(ValueError):
        extractor.extract({"text": "x"}, prompt_hash="p", model_id="m", checkpoint_revision="r", research_asof=T0)


def test_replay_uses_cached_decisions():
    cache = ExtractorCache()
    extractor = SchemaExtractor(
        cache,
        lambda payload, prompt_hash: {"ticker": "AAA", "event_date": "2021-01-04"},
        required_fields=("ticker", "event_date"),
    )
    kwargs = dict(prompt_hash="p", model_id="m", checkpoint_revision="r", research_asof=T0)
    extractor.extract({"text": "x"}, **kwargs)
    replay = extractor.extract({"text": "x"}, **kwargs)
    assert replay.from_cache is True
    assert extractor.replay_uses_cache() is True
