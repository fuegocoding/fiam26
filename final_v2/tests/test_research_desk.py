"""Typed three-role research composes literature and actual native measurements."""

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from pod_platform.agents.provider import DeterministicMockProvider
from pod_platform.application.research_desk import ResearchDesk
from pod_platform.research.composition import observation_digest
from tests.test_research_composition import env as composition_env

NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)


def make_desk(
    tmp_path,
    env,
    *,
    category="testable",
    fail_role=None,
    malformed_first=False,
    repair_attempts=0,
):
    service, spec, grant, rows = env
    seen = []

    def extract(request):
        seen.append(("extractor", json.loads(request.user)))
        if fail_role == "extractor":
            raise RuntimeError("private transport detail")
        payload = json.loads(request.user)
        return json.dumps(
            {
                "evidence_ids": [
                    i["evidence_id"]
                    for i in payload["public_evidence"]
                    + payload["source_metadata"]
                ],
                "grounded_claim": "The supplied literature supports testing conditional demand interactions, without proving profitability.",
                "limitations": [
                    "Synthetic public paper fixtures and descriptive source metadata do not establish historical eligibility."
                ],
            }
        )

    def lead(request):
        seen.append(("lead", json.loads(request.user)))
        if fail_role == "lead":
            raise RuntimeError("private transport detail")
        if malformed_first and "previous_attempt_rejected" not in json.loads(
            request.user
        ):
            return json.dumps({**spec, "composition_name": "Upper_Case"})
        return json.dumps(spec)

    def critique(request):
        seen.append(("challenger", json.loads(request.user)))
        if fail_role == "challenger":
            raise RuntimeError("private transport detail")
        return json.dumps(
            {
                "disposition": "review",
                "reason_category": category,
                "evidence_refs": json.loads(request.user)["evidence_ids"],
                "competing_explanation": "A changing source panel rather than company economics can explain the effect.",
                "falsification_test": "Compare the interaction against both components and the fixed native baseline.",
            }
        )

    providers = SimpleNamespace(
        extractor=DeterministicMockProvider(
            extract, "desk-extractor", clock=lambda: NOW
        ),
        lead=DeterministicMockProvider(lead, "desk-lead", clock=lambda: NOW),
        challenger=DeterministicMockProvider(
            critique, "desk-challenger", clock=lambda: NOW
        ),
    )
    desk = ResearchDesk(
        tmp_path / "desk",
        "pod_fundamentals",
        papers=service.papers,
        composition=service,
        carbon=None,
        providers=providers,
        clock=lambda: NOW,
        run_id="desk_test",
        max_calls=3,
        max_tokens=100000,
        repair_attempts=repair_attempts,
    )
    args = dict(
        question=spec["question"],
        grant_id=grant["grant_id"],
        paper_ids=[c["paper_id"] for c in spec["paper_components"]],
        source_metadata=[
            {
                "source_id": "traffic",
                "metric": "visits",
                "description": "Public foot traffic panel documentation.",
            },
            {
                "source_id": "spend",
                "metric": "spend",
                "description": "Public card spending panel documentation.",
            },
        ],
        reasoning_mode=spec["reasoning_mode"],
    )
    return desk, args, seen, rows


def test_two_papers_original_reasoning_and_alt_metadata_reach_measured_native_graph(
    tmp_path, composition_env
):
    desk, args, seen, rows = make_desk(tmp_path, composition_env)
    result = desk.propose(**args)
    assert result["status"] == "registered", result
    assert result["critique"]["disposition"] == "review"
    assert len(result["evidence_ids"]) == 4
    plan = desk.composition.integrated_plan(result["composition_id"])
    assert len(plan["evidence"]) == 2 and plan["spec"]["own_reasoning"]
    assert [r for r, _ in seen] == ["extractor", "lead", "challenger"]
    # Source cards and actual row/label values stay local; only scalar metadata enters prompts.
    prompt = json.dumps(seen)
    assert (
        "security_ids" not in prompt
        and "source_cards" not in prompt
        and "/tmp/" not in prompt
    )
    assert "observation_id" not in prompt and "source_card_path" not in prompt
    desk.composition.bind_inputs(
        result["composition_id"],
        {
            s: observation_digest([o for o in rows if o.source_id == s])
            for s in ["traffic", "spend", "earnings"]
        },
    )
    measured = desk.tick(result["composition_id"], lambda: rows)
    assert measured["status"] == "measured" and not measured["admitted"]
    assert desk.propose(**args) == result
    assert len(seen) == 3
    assert all(v["calls_used"] == 1 for v in desk.status()["roles"].values())


def test_restarts_and_advancing_wall_clock_reuse_actual_cached_calls(
    tmp_path, composition_env
):
    desk, args, seen, _ = make_desk(tmp_path, composition_env)
    first = desk.propose(**args)
    desk.clock = lambda: NOW + dt.timedelta(seconds=10)
    assert desk.propose(**args) == first and len(seen) == 3
    restarted = ResearchDesk(
        desk.state_dir,
        desk.namespace,
        papers=desk.papers,
        composition=desk.composition,
        carbon=None,
        providers=SimpleNamespace(
            **{role: runner.provider for role, runner in desk.roles.items()}
        ),
        clock=desk.clock,
        run_id="desk_test",
        max_calls=3,
        max_tokens=100000,
    )
    assert restarted.propose(**args) == first and len(seen) == 3


@pytest.mark.parametrize("failed_role", ["extractor", "lead", "challenger"])
def test_failed_role_is_charged_without_fallback_approval_or_repeated_paid_retry(
    tmp_path, composition_env, failed_role
):
    desk, args, seen, _ = make_desk(
        tmp_path, composition_env, fail_role=failed_role
    )
    result = desk.propose(**args)
    assert result["status"] == "model_failed" and not result["admitted"]
    assert "private transport detail" not in json.dumps(result)
    assert desk.status()["roles"][failed_role]["calls_used"] == 1
    assert desk.propose(**args) == result
    assert "composition_id" not in result


def test_non_testable_challenge_holds_without_submitting_strategy(
    tmp_path, composition_env
):
    desk, args, _, _ = make_desk(
        tmp_path, composition_env, category="unsupported_source"
    )
    result = desk.propose(**args)
    assert result["status"] == "held_review" and "composition_id" not in result


def test_original_reasoning_requires_no_paper_or_full_replication(
    tmp_path, composition_env
):
    composition_env[1].update(
        paper_components=[], reasoning_mode="own_reasoning"
    )
    desk, args, _, _ = make_desk(tmp_path, composition_env)
    result = desk.propose(**args)
    assert result["status"] == "registered"
    assert (
        desk.composition.integrated_plan(result["composition_id"])["evidence"]
        == []
    )


@pytest.mark.parametrize(
    "metadata",
    [
        [
            {
                "source_id": "traffic",
                "metric": "visits",
                "data_path": "/tmp/private.csv",
            }
        ],
        [
            {
                "source_id": "traffic",
                "metric": "visits",
                "rows": [{"value": 123}],
            }
        ],
        [{"source_id": "traffic", "metric": "private_return"}],
        [
            {
                "source_id": "traffic",
                "description": "Stored in /home/person/private.csv",
            }
        ],
    ],
)
def test_private_or_ungranted_metadata_cannot_enter_prompts(
    tmp_path, composition_env, metadata
):
    desk, args, seen, _ = make_desk(tmp_path, composition_env)
    with pytest.raises(ValueError):
        desk.propose(**{**args, "source_metadata": metadata})
    assert seen == []


@pytest.mark.parametrize("repair_attempts", [0, 1])
def test_rejected_output_is_returned_for_one_charged_repair(
    tmp_path, composition_env, repair_attempts
):
    desk, args, seen, _ = make_desk(
        tmp_path,
        composition_env,
        malformed_first=True,
        repair_attempts=repair_attempts,
    )
    result = desk.propose(**args)
    leads = [payload for role, payload in seen if role == "lead"]
    if repair_attempts:
        assert result["status"] == "registered", result
        assert len(leads) == 2
        assert "composition_name" in leads[1]["previous_attempt_rejected"]
        assert [r["role"] for r in result["repairs"]] == ["lead"]
        assert desk.status()["roles"]["lead"]["calls_used"] == 2
    else:
        assert result["status"] == "model_failed"
        assert len(leads) == 1 and result["repairs"] == []
