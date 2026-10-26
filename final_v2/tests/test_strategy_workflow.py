"""Configured research uses actual graph execution and shared service budgets."""

import datetime as dt
import json
import hashlib

import pytest

from pod_platform.application.configuration import load_config
from pod_platform.application.research_loop import build_fixture_loop
from pod_platform.application.strategy_workflow import StrategyWorkflow
from pod_platform.services.app import build_server
from pod_platform.research.composition import observation_digest
from tests.test_research_composition import env as composition_env


def test_root_configuration_runs_composed_research_and_resumes(
    tmp_path, composition_env
):
    service, spec, grant, observations = composition_env
    original = build_fixture_loop(tmp_path / "host")
    settings = load_config()
    settings["application"].update(online_papers=False, paper_queries=[])
    settings["research"].update(
        prepare_inputs=False, desk_evaluation={"enabled": False}
    )
    settings["portfolio"]["cash_run"]["enabled"] = False
    settings["application"].update(
        state_dir=str(tmp_path / "runtime"),
        output_dir=str(tmp_path / "output"),
        manifest=str(original.builtin.manifest.path),
        papers_db=service.papers.store.path,
        run_id="configured_test",
    )
    settings["research"].update(
        measurement_grants=[grant],
        compositions=[{"namespace": "pod_fundamentals", "spec": spec}],
        financial_reviews=[],
        proposals=[],
    )
    bindings = []
    for source in grant["source_metrics"]:
        rows = [o for o in observations if o.source_id == source]
        path = tmp_path / (source + ".json")
        path.write_text(json.dumps([o.model_dump(mode="json") for o in rows]))
        import_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        bindings.append(
            {
                "source_id": source,
                "path": str(path),
                "sha256": import_hash,
                "observation_digest": observation_digest(rows),
            }
        )
    settings["research"]["observation_sources"] = bindings
    at = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)
    workflow = StrategyWorkflow(
        settings, providers=original.providers, clock=lambda: at
    )
    result = workflow.run()
    assert result["research"][0]["status"] == "measured"
    assert result["orders"] == 0
    assert (
        result["portfolio"]["status"]
        == "waiting_for_admitted_forecasts_and_market_inputs"
    )
    assert workflow.run() == result
    assert all(b.calls_used == 0 for b in workflow.loop.budgets.values())
    workflow.carbon.transport = lambda *_args: {
        "rows": [{"date": "2015-01-31", "value": 12}]
    }
    workflow.carbon.register_capability(
        dict(
            capability_id="requested_demand",
            namespace="pod_fundamentals",
            source_id="demand",
            dataset_id="demand",
            tearsheet_name="demand",
            entity_id=111,
            entity_rep="company",
            metric_id=55,
            metric_label="Demand",
            topic_name="Demand",
            unit="provider_native_unverified",
            reviewed_by="test_provider_policy",
        )
    )
    receipt = workflow.carbon.request(
        "pod_fundamentals",
        {
            "capability_id": "requested_demand",
            "start": "2015-01-01",
            "end": "2015-01-31",
        },
    )
    attached = workflow.attach_source(
        "pod_fundamentals",
        grant["grant_id"],
        receipt["request_id"],
        "0",
        "demand",
    )
    assert (
        not attached["formation_eligible"]
        and attached["join_status"] == "unverified_research_mapping"
    )
    binding = workflow.loop._events("requested_source", "pod_fundamentals")[0]
    normalized = workflow._carbon_observations(binding)
    assert (
        normalized[0].available_at == at
        and normalized[0].version_known_at == at
    )
    assert (
        normalized[0].period_end.year == 2015
    )  # receipt time was not backdated
    server = build_server(
        workflow.loop.state_dir,
        "fundamentals",
        clock=lambda: at,
        research_runtime=workflow.loop,
        discovery=workflow.loop.discovery,
    )
    assert {
        "research.investigate",
        "research.attach_source",
        "sources.discover_stock",
        "sources.request_stock",
        "research.run_composition",
    } <= server.tools.keys()
    server.close()
    bindings[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file changed"):
        workflow._load_observations(bindings)
