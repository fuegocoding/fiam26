"""Deterministic offline demo responder. NOT a language model.

Explicit demo/test use only: it echoes evidence ids from the task input into a
syntactically valid proposal so the queue, cache and validation paths can be
exercised without a billable call. Output is labelled ``[DEMO MOCK]`` and the
provider is flagged ``is_mock`` in every provenance record.
"""

from __future__ import annotations

import json

from ..agents.provider import DeterministicMockProvider, LLMRequest


def demo_responder(request: LLMRequest) -> str:
    task = json.loads(request.user)
    evidence = task["untrusted_evidence"]
    ids = [item["evidence_id"] for item in evidence][:2]
    executor = task["allowed_executors"][0]
    pod = task["task"]["pod_id"]
    proposal = {
        "title": f"[DEMO MOCK] {pod} draft over {len(evidence)} evidence cards",
        "mechanism": "Placeholder mechanism produced by the deterministic demo responder, not by a model.",
        "falsifiable_target": "Demo target: the design-partition rank spread is not above the stated baseline.",
        "baseline": "Simple lagged-characteristic baseline.",
        "features": [
            {
                "name": "demo_feature",
                "definition": "Demo feature built from the cited evidence cards only.",
                "timing": "Inputs known at month end before the signal date.",
                "source_ids": ids,
            }
        ],
        "evidence_refs": ids,
        "experiment_plan": {
            "executor_id": executor,
            "primary_metric": "rank_ic",
            "data_partition": "design",
            "variants": [{"name": "base", "params": {"lookback_months": 12}}],
            "description": "Run the registered executor on the design partition only.",
        },
        "competing_explanation": "The apparent effect may be a known factor under another name.",
        "limitations": [
            "Demo output; no model reasoning and no evaluation have occurred."
        ],
        "missing_information": [
            "Everything: this is not a real research result."
        ],
        "evidence_quality": "weak",
    }
    return json.dumps(proposal)


def demo_provider() -> DeterministicMockProvider:
    return DeterministicMockProvider(demo_responder, model="demo-mock-1")
