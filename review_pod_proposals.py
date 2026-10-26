"""One cached challenger review of the actual lead model's four pod plans."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "final_v2"))

from pod_platform.agents.provider import GenerationParams, LLMRequest
from pod_platform.application.common import atomic_write_json
from pod_platform.application.configuration import config_path, load_config
from pod_platform.application.live_models import create_live_models
from pod_platform.research.pod_plans import PLAN_PROMPT

from call_pod_proposals import AllPodPlans


def main():
    config = load_config()
    cache = config_path(config["application"]["state_dir"]) / "pod_plan_proposals"
    lead = json.loads((cache / "all_pods_bd3d5e5345c0bf3c.json").read_text())
    task = {
        "original_proposals": lead["proposal"],
        "feature_dictionary": lead["packet"]["pods"],
        "review_requirements": [
            "Diagnose only supported facts; the previous four pod-specific control descriptions were invented. The actual prior battery used sector and beta neutrality in all pods.",
            "ivol_capm is idiosyncratic realized volatility, not options implied volatility.",
            "Centered-rank products are interactions: negative times negative is positive. They cannot stand for high-momentum AND low-volatility or low-issuance AND high-investment without further reasoning. Prefer simple additive signed terms for a first-pass falsifiable mechanism.",
            "Positive f_score denotes financial strength. A negative sign contradicts a quality-long description.",
            "The host uses all 2015-2020 for development. Do not call any chosen subset untouched out-of-sample. Require chronological pre-2021 checks with decisions frozen before the measured returns.",
            "Events are monthly issuance/accrual/investment proxies, with no verified event-day timestamps. Do not propose event-day causal returns.",
            "Sector controls neutralize sector exposure and permit beta; sector_beta neutralizes both. Dollar controls permit both sector and beta exposures. State intentional exposure truthfully.",
            "Test return edge per unit exposure, after spread and modeled borrow, before any capital sizing. Borrow rates and liquidity do not enter the forecast. A separate continuous allocator solves marginal edge versus impact and inventory limits.",
            "Do not invent percent-of-universe turnover metrics, dimensional nonsense about liquidity, performance thresholds chosen after observing results, or capital admission. Falsify with measured chronological return, benchmark exposure, spread/borrow drag, and mechanism-consistent sign stability.",
            "Return exactly one corrected, economically coherent initial plan per pod with concrete supported diagnostic issues. No claim of good performance. Use the given feature dictionary only.",
        ],
    }
    request = LLMRequest(
        PLAN_PROMPT,
        json.dumps(task, sort_keys=True),
        GenerationParams.model_validate(config["models"]["generation"]),
        role="lead_semantic_review",
        response_schema=AllPodPlans.model_json_schema(),
    )
    destination = cache / ("reviewed_" + request.request_hash[:16] + ".json")
    if destination.exists():
        print(destination)
        return
    if (cache / "prompt_test_completion.json").exists():
        raise RuntimeError(
            "The approved four-call test is complete; only cached responses may be reused"
        )
    providers, _ = create_live_models(
        {**config["models"], "retry_attempts": 0}, cache, "pod_plan_repair"
    )
    response = providers.lead.complete(request)
    atomic_write_json(
        cache / ("review_response_" + request.request_hash[:16] + ".json"),
        {"text": response.text, "response_hash": response.response_hash},
    )
    proposal = AllPodPlans.model_validate_json(response.text)
    atomic_write_json(
        destination,
        {
            "origin": "llm_proposal_review",
            "proposal": proposal.model_dump(mode="json"),
            "prompt": PLAN_PROMPT,
            "packet": task,
            "request_hash": request.request_hash,
            "model": response.model,
            "provider": response.provider,
            "total_tokens": response.total_tokens,
            "received_at": response.received_at.isoformat(),
            "response_hash": response.response_hash,
            "lead_request_hash": lead["request_hash"],
        },
    )
    print(destination)


if __name__ == "__main__":
    main()
