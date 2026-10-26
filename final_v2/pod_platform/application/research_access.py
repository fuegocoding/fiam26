"""Inspect actual research access and distinguish it from disconnected capabilities."""

from __future__ import annotations

import json

from ..research.input_permissions import (
    COMPETITION_CHARACTERISTICS,
    FILING_RESEARCH_FEATURES,
)
from ..research.training_data import POD_FEATURES
from .common import atomic_write_json, file_sha256
from .configuration import config_path


def additional_input_hashes(settings):
    if settings.get("filings_research_enabled", False):
        return {
            "filings_sha256": file_sha256(config_path(settings["filings_path"]))
        }
    return {}


def describe_research_access(config, panel_metadata=None):
    settings = config.get("training_lab", {})
    mode = config.get("application", {}).get("mode")
    filing = bool(settings.get("filings_research_enabled", False))
    result = {
        "active_mode": mode,
        "sources": {
            "supplied_characteristics": {
                "status": "available",
                "count": len(COMPETITION_CHARACTERISTICS),
                "features": list(COMPETITION_CHARACTERISTICS),
                "pod_feature_permissions": "all pods may use all characteristics",
                "default_pod_examples": {
                    p: list(f) for p, f in POD_FEATURES.items()
                },
            },
            "supplied_8k": {
                "status": "available_for_local_research"
                if filing
                else "not_enabled",
                "metadata_and_features_enabled": filing,
                "raw_text_reader": "pod_platform.research.filing_access.FilingResearchStore.query",
                "training_end": settings.get("end"),
                "numeric_features": list(FILING_RESEARCH_FEATURES),
                "as_of_query_required": True,
                "automatic_cloud_transmission": False,
                "corpus_audit": (panel_metadata or {}).get("filing_research"),
                "financial_admission": False,
            },
            "supplied_daily_returns": {
                "status": "available",
                "uses": [
                    "daily accounting",
                    "trailing technical hypotheses",
                    "execution timing",
                ],
            },
            "papers": {
                "status": "implemented_tools_not_connected_to_active_diagnostics",
                "research_papers_utility_online_default": config.get(
                    "research_papers", {}
                ).get("online", False),
                "note": "Current diagnostics consume cached plans; they do not automatically search and read papers.",
            },
            "alternative_data": {
                "status": "connected_services_not_automatically_explored_by_active_diagnostics",
                "remote_access_configured": config.get("carbon", {}).get(
                    "allow_remote", False
                ),
                "analytical_call_limit": config.get("carbon", {}).get(
                    "max_analytical_calls"
                ),
                "discovery_utility_online_default": config.get(
                    "discover_alt_sources", {}
                ).get("online", False),
            },
        },
        "remaining_implementation_limits": [
            {
                "item": "Autonomous strategy coding",
                "status": "not_implemented_in_active_loop",
                "detail": "Codex writes Python; runtime models currently return structured formulas.",
            },
            {
                "item": "Formula language",
                "status": "restricted",
                "detail": "Up to six terms using identity/product/difference; fixed smoothing and review options. This is an implementation limit, not a competition rule.",
            },
            {
                "item": "Text interpretation",
                "status": "reader_available_extractor_not_connected",
                "detail": "Raw dated 8-K text is locally accessible. Automatic model-driven semantic feature generation is not yet part of run.py.",
            },
            {
                "item": "Experiments",
                "status": "cached_plan_batch",
                "detail": "Current diagnostics test recorded proposals and host controls, rather than generating new experiments autonomously.",
            },
            {
                "item": "Horizon and trading rules",
                "status": "one_month_signal_screen",
                "detail": "Active prefix diagnostics precede entry/exit and full holding-period execution tests.",
            },
            {
                "item": "Theo's full signal stack",
                "status": "not_wired_to_active_mode",
                "detail": "Existing specialized modules and saved artifacts are not invoked by edge_diagnostics.",
            },
            {
                "item": "Legacy composition pilot",
                "status": "separate_bounded_interface",
                "detail": "The older composition grant still caps 500 securities and 300,000 observations; the active columnar training universe has no stock-count cap.",
            },
            {
                "item": "Final benchmark reporting",
                "status": "not_complete_in_active_research",
                "detail": "Research uses zero cash yield and an equal-weight diagnostic reference; official TB3MS+4% and S&P reporting remain separate.",
            },
            {
                "item": "External model test",
                "status": "approved_four_call_test_complete",
                "detail": "The recorded authorization is exhausted. This access expansion performs no new model calls.",
            },
        ],
        "competition_and_session_boundaries": [
            "Development observations end by 2020-12-31; forward returns remain evaluation labels, never input features.",
            "Publication timestamps and the next-session-close execution convention remain enforced.",
            "The combined fund, not each signal diagnostic, must hold 200-500 names and satisfy gross/net/name limits.",
            "Short positions require at least $500m formation market cap.",
            "8-K survivorship and uncertain historical identity maps remain disclosed; access does not certify an unbiased source.",
            "Legacy external caches are not silently treated as approved finals inputs.",
        ],
    }

    if mode == "research_campaign":
        result["sources"]["papers"]["status"] = (
            "connected_search_read_and_prompt_evidence"
        )
        result["sources"]["papers"]["note"] = (
            "The campaign searches dated paper versions and supplies actual cached text and references to code generation."
        )
        result["sources"]["alternative_data"]["status"] = (
            "connected_discovery_request_receipts_and_dated_feature_intake"
        )
        result["sources"]["supplied_8k"]["numeric_features"] += list(
            (panel_metadata or {})
            .get("text_measurements", {})
            .get("features", [])
        )
        result["remaining_implementation_limits"] = []
        result["run_health"] = (panel_metadata or {}).get(
            "campaign_run_health",
            {"status": "run_in_progress_no_success_claim"},
        )
        result["capability_status_scope"] = (
            "Configured implementation paths; actual successful acquisition and agent execution are reported separately in run_health."
        )
        result["campaign_connections"] = [
            "evidence discovery -> code proposal -> protected evaluator -> measured feedback -> revised code",
            "all characteristic and dated local filing measurements -> monthly/weekly/daily native reviews",
            "capital-free edge and concentration curves -> persistent pod rosters -> continuous capacity allocation",
            "staged fills -> actual holdings -> daily NAV -> official TB3MS+4% and S&P reporting",
            "original economic controls -> identical execution and allocation counterfactual",
        ]
        result["implementation_contract"] = (
            "Executable strategy Python uses a finite IO-free numerical language; forecasts cannot access outcomes or capital."
        )
    return result


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
        "All 147 supplied characteristics are available to all pods. Default pod feature lists guide baseline experiments; they do not restrict data access.",
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
