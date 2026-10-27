"""Inspect actual research access and distinguish it from disconnected capabilities."""

from __future__ import annotations

import json

from ..prompts import PROMPTS_PATH
from ..research.input_permissions import (
    COMPETITION_CHARACTERISTICS,
    FILING_RESEARCH_FEATURES,
)
from ..research.training_data import POD_FEATURES
from .common import atomic_write_json, file_sha256
from .configuration import config_path


def additional_input_hashes(settings):
    hashes = {"prompts_sha256": file_sha256(PROMPTS_PATH)}
    if settings.get("filings_research_enabled", False):
        hashes["filings_sha256"] = file_sha256(config_path(settings["filings_path"]))
    return hashes


def describe_research_access(config, panel_metadata=None):
    """Describe the active campaign, with successful acquisition reported separately."""
    settings = config.get("training_lab", {})
    metadata = panel_metadata or {}
    filing = bool(settings.get("filings_research_enabled", False))
    return {
        "active_mode": config.get("application", {}).get("mode"),
        "sources": {
            "supplied_characteristics": {
                "status": "available", "count": len(COMPETITION_CHARACTERISTICS),
                "features": list(COMPETITION_CHARACTERISTICS),
                "pod_feature_permissions": "all pods may use all characteristics",
                "default_pod_examples": {p: list(f) for p, f in POD_FEATURES.items()},
            },
            "supplied_8k": {
                "status": "available_for_local_research" if filing else "not_enabled",
                "metadata_and_features_enabled": filing,
                "raw_text_reader": "pod_platform.research.filing_access.FilingResearchStore.query",
                "training_end": settings.get("end"),
                "numeric_features": list(FILING_RESEARCH_FEATURES) + list(metadata.get("text_measurements", {}).get("features", [])),
                "custom_measurements": metadata.get("custom_text_measurements", {}),
                "as_of_query_required": True, "automatic_cloud_transmission": False,
                "corpus_audit": metadata.get("filing_research"), "financial_admission": False,
            },
            "supplied_daily_returns": {"status": "available", "uses": ["daily accounting", "trailing technical hypotheses", "observed-price fills"]},
            "papers": {"status": "connected_search_read_and_prompt_evidence",
                "online": config.get("application", {}).get("online_papers", False),
                "note": "Pods can request additional searches; paper versions must precede the declared training cutoff."},
            "alternative_data": {"status": "connected_discovery_request_receipts_and_dated_feature_intake",
                "remote_access_configured": config.get("carbon", {}).get("allow_remote", False),
                "numeric_intake": metadata.get("alternative_features", []),
                "note": "Pods may discover and request data. Historical numeric use requires dated security binding and verified availability."},
        },
        "remaining_implementation_limits": [],
        "run_health": metadata.get("campaign_run_health", {"status": "run_in_progress_no_success_claim"}),
        "capability_status_scope": "Configured paths; successful acquisitions and actual model execution are reported separately in run_health.",
        "campaign_connections": [
            "dated evidence discovery -> agent code -> protected evaluator -> measured feedback -> revised code",
            "all characteristics and dated filing measurements -> monthly/weekly/daily native reviews",
            "unit edge and concentration curves -> supported pod ownership -> continuous capital allocation",
            "full observed-price fills -> holdings -> daily NAV -> TB3MS and S&P reporting",
            "pod-requested paper and alternative-data research -> revised hypotheses or explicit retention",
        ],
        "implementation_contract": "Executable strategy Python uses a finite IO-free numerical language; forecasts cannot access outcomes or capital.",
        "competition_and_session_boundaries": [
            "Research outcomes and filing access stop at the declared fold training cutoff; forward returns remain labels, never inputs.",
            "Publication timestamps are enforced; orders fill at the first observed quote on or after the decision.",
            "The combined fund must hold 200-500 names when invested and satisfy gross/net/name limits.",
            "Short positions require at least $500m formation market cap.",
            "Filing survivorship and uncertain historical identity maps remain disclosed.",
            "No historical availability is inferred from a current alternative-data receipt.",
        ],
    }


def write_access_audit(config, output, panel_metadata=None):
    output = config_path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = describe_research_access(config, panel_metadata)
    atomic_write_json(output / "research_access.json", audit)
    lines = [
        "# Research access audit",
        "",
        f"Active mode: `{audit['active_mode']}`.",
        "",
        "All 147 supplied characteristics are available to all pods. Pod feature examples guide research; they do not restrict data access.",
        "",
        "The supplied 8-K corpus is enabled for dated local research, including paginated full-text queries. It is not automatically transmitted to a remote model, and its known biases remain recorded.",
        "",
        "| Capability | Current status |",
        "|---|---|",
    ]
    lines += [
        f"| {row['item']} | {row['detail']} |"
        for row in audit["remaining_implementation_limits"]
    ]
    if audit.get("campaign_connections"):
        lines += ["", "Implemented connections:", ""]
        lines += ["- " + item for item in audit["campaign_connections"]]
        lines += [
            "",
            "Actual run health (implementation availability does not establish successful execution):",
            "",
            "```json",
            json.dumps(audit["run_health"], indent=2),
            "```",
        ]
    lines += ["", "The session and competition boundaries remain:", ""]
    lines += [
        "- " + item for item in audit["competition_and_session_boundaries"]
    ]
    (output / "RESEARCH_ACCESS.md").write_text("\n".join(lines) + "\n")
    return audit
