"""One explicitly authorized, cached Groq call covering all four pod mandates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "final_v2"))

from pod_platform.agents.provider import GenerationParams, LLMRequest
from pod_platform.application.common import atomic_write_json
from pod_platform.application.configuration import config_path, load_config
from pod_platform.application.live_models import create_live_models
from pod_platform.contracts.base import StrictModel
from pod_platform.pods.charters import pod_spec
from pod_platform.research.input_permissions import AVAILABLE_RESEARCH_FEATURES
from pod_platform.research.pod_plans import PLAN_PROMPT, PodPlans
from pod_platform.research.training_data import POD_FEATURES


class AllPodPlans(StrictModel):
    fundamentals: PodPlans
    events: PodPlans
    valuation: PodPlans
    dynamics: PodPlans


def main():
    config = load_config()
    state = config_path(config["application"]["state_dir"])
    parent = state / "ac5162037ff2d6d0"
    ledger = json.loads((parent / "candidate_ledger.json").read_text())
    packet = {
        "development_period": "2015-2020",
        "available_research_features": sorted(AVAILABLE_RESEARCH_FEATURES),
        "pods": {},
    }
    for pod, features in POD_FEATURES.items():
        ranked = sorted(
            (x for x in ledger if x["pod"] == pod),
            key=lambda x: x["selection_score"],
            reverse=True,
        )
        packet["pods"][pod] = {
            "mission": pod_spec(pod).mission,
            "primary_features": features,
            "prior_training_summary": [
                {
                    "strategy_id": r["strategy_id"],
                    "net_cagr": r["metrics"]["cagr"],
                    "sharpe": r["metrics"]["sharpe_annualized"],
                }
                for r in ranked[:2]
            ],
        }
    packet["task"] = (
        "Give exactly ONE initial executable economic plan per pod. Diagnose the old fixed-capital, one-control-policy setup. All feature names across pods are available as controls. Select rules on per-unit return edge; capital allocation is a separate continuous optimization, never a capital grid. No forward dates, stock identities or winner anecdotes. These are research proposals, not good-strategy claims."
    )
    params = GenerationParams.model_validate(config["models"]["generation"])
    request = LLMRequest(
        PLAN_PROMPT,
        json.dumps(packet, sort_keys=True),
        params,
        role="lead",
        response_schema=AllPodPlans.model_json_schema(),
    )
    cache = state / "pod_plan_proposals"
    cache.mkdir(exist_ok=True)
    destination = cache / ("all_pods_" + request.request_hash[:16] + ".json")
    if destination.exists():
        print(destination)
        return
    if (cache / "prompt_test_completion.json").exists():
        raise RuntimeError(
            "The approved four-call test is complete; only cached responses may be reused"
        )
    provider_settings = {**config["models"], "retry_attempts": 0}
    providers, _ = create_live_models(provider_settings, cache, "pod_plan_repair")
    response = providers.lead.complete(request)
    # Persist actual output even if schema validation fails; no extra repair
    # call is silently made and invalid proposals cannot enter the evaluator.
    atomic_write_json(
        cache / ("response_" + request.request_hash[:16] + ".json"),
        {
            "text": response.text,
            "response_hash": response.response_hash,
            "request_hash": request.request_hash,
            "model": response.model,
        },
    )
    parsed = AllPodPlans.model_validate_json(response.text)
    atomic_write_json(
        destination,
        {
            "origin": "llm_proposal",
            "proposal": parsed.model_dump(mode="json"),
            "prompt": PLAN_PROMPT,
            "packet": packet,
            "request_hash": request.request_hash,
            "model": response.model,
            "provider": response.provider,
            "received_at": response.received_at.isoformat(),
            "response_hash": response.response_hash,
            "total_tokens": response.total_tokens,
        },
    )
    print(destination)


if __name__ == "__main__":
    main()
