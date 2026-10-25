"""Alternative-data requests materialize real tables without inventing PIT eligibility."""

import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from pod_platform.contracts.sources import SourceCard
from pod_platform.data.carbonarc_requests import (
    CarbonArcCapability,
    CarbonArcPolicy,
    CarbonArcRequests,
    CarbonArcOAuthTransport,
)

NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)
NS = "pod_fundamentals"
REQUEST = {
    "capability_id": "retail_spend",
    "start": "2025-01-01",
    "end": "2025-01-31",
}


def capability(**changes):
    return {
        **dict(
            capability_id="retail_spend",
            namespace=NS,
            source_id="card_spend",
            dataset_id="card_panel",
            tearsheet_name="card_panel",
            entity_id=123,
            entity_rep="company",
            metric_id=55,
            metric_label="Card spend",
            topic_name="Retail demand",
            unit="usd",
            reviewed_by="host_operator",
        ),
        **changes,
    }


def test_pods_discover_and_request_their_own_stocks_instead_of_a_fixed_example(
    tmp_path,
):
    calls = []
    companies = {"Company Alpha": 111, "Company Beta": 222}

    def transport(tool, arguments):
        calls.append((tool, arguments))
        if tool == "search_entities":
            return {
                "data": [
                    {
                        "entity_id": companies[arguments["query"]],
                        "entity_label": arguments["query"],
                        "entity_representation": "company",
                    }
                ]
            }
        if tool == "search_insights":
            return {
                "data": [
                    {
                        "insight_id": 55,
                        "insight_label": "Card Spend",
                        "topic_label": "Retail demand",
                    }
                ]
            }
        if tool == "data_library":
            return {"data": [{"tearsheet_name": "card-panel"}]}
        entity = 111 if "entity 111 " in arguments["question"] else 222
        return {
            "rows": [
                {
                    "DATE": "2025-01-31",
                    "CARD SPEND": entity * 100,
                    "ENTITY ID": entity,
                }
            ],
            "metadata": {"insight_id": 55},
        }

    service = CarbonArcRequests(
        tmp_path,
        policy=CarbonArcPolicy(
            allow_remote=True, max_analytical_calls=2, max_metadata_calls=6
        ),
        transport=transport,
        clock=lambda: NOW,
    )
    for name, entity in companies.items():
        found = service.discover_stock(
            NS, {"entity_query": name, "metric_query": "card spend"}
        )
        selected = {
            "discovery_id": found["discovery_id"],
            "entity_id": entity,
            "metric_id": 55,
            "dataset_id": "card-panel",
            "start": "2025-01-01",
            "end": "2025-01-31",
        }
        ready = service.request_stock(NS, selected)
        assert ready["status"] == "data_ready"
        assert ready["entity_column_observed"]
        assert service.request_stock(NS, selected) == ready
    assert len([c for c in calls if c[0] == "text_to_insight"]) == 2
    assert service.snapshots(NS)["usage"] == {
        "metadata": 6,
        "analytical": 2,
        "local": 0,
    }
    assert (
        service.discover_stock(
            NS, {"entity_query": "Company Alpha", "metric_query": "card spend"}
        )["status"]
        == "discovered"
    )
    assert (
        service.discover_stock(
            NS, {"entity_query": "Company Gamma", "metric_query": "card spend"}
        )["status"]
        == "budget_exhausted"
    )
    with pytest.raises(ValueError, match="this pod"):
        service.request_stock("pod_events", selected)


def response(rows=None):
    return {
        "structuredContent": {
            "rows": rows
            or [
                {"date": "2025-01-01", "value": "1,234.5"},
                {"date": "2025-01-02", "value": None},
            ],
            "metadata": {"vintage": "unknown"},
        }
    }


@pytest.fixture
def world(tmp_path):
    calls = []
    service = None

    def transport(tool, args):
        # A committed reservation exists before any vendor call.
        with sqlite3.connect(service.path) as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM carbon_request_events WHERE kind='reservation'"
                ).fetchone()[0]
                >= 1
            )
        calls.append((tool, args))
        return response()

    policy = CarbonArcPolicy(
        allow_remote=True, max_analytical_calls=2, max_calls_per_pod=2
    )
    service = CarbonArcRequests(
        tmp_path / "state",
        transport=transport,
        policy=policy,
        clock=lambda: NOW,
    )
    service.register_capability(capability())
    return service, calls


def test_actual_table_missingness_source_card_and_idempotency(world):
    service, calls = world
    result = service.request(NS, REQUEST)
    assert result["status"] == "data_ready"
    assert result["rows"] == 2 and result["missing_values"] == 1
    assert (
        not result["formation_eligible"]
        and not result["historical_coverage_established"]
        and not result["vintage_established"]
    )
    assert (
        result["entity_column_observed"] is False
        and result["availability_column_observed"] is False
    )
    assert (
        "requested_entity_id,observed_entity_id"
        in Path(result["data_path"]).read_text()
    )
    assert SourceCard.model_validate_json(
        Path(result["source_card_path"]).read_text()
    ).verify_seal()
    assert len(calls) == 1 and calls[0][0] == "text_to_insight"
    assert (
        "insight_id 55" in calls[0][1]["question"]
        and "2025-01-31" in calls[0][1]["question"]
    )
    assert service.request(NS, REQUEST) == result
    restarted = CarbonArcRequests(
        service.state_dir,
        transport=lambda *_: pytest.fail("cached call"),
        policy=service.policy,
        clock=lambda: NOW,
    )
    assert restarted.request(NS, REQUEST) == result
    assert restarted.read(NS, result["request_id"]) == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match="caller namespace"):
        restarted.read("pod_events", result["request_id"])
    with pytest.raises(ValueError, match="caller namespace"):
        restarted.request("pod_events", REQUEST)


def test_immutable_finite_call_budget_and_failures_stay_charged(world):
    service, calls = world
    service.transport = lambda *_: (_ for _ in ()).throw(
        RuntimeError("private provider error with secrets")
    )
    first = service.request(NS, REQUEST)
    assert first["status"] == "failed" and "secrets" not in json.dumps(first)
    assert service.request(NS, REQUEST) == first
    other = {**REQUEST, "end": "2025-01-30"}
    assert service.request(NS, other)["status"] == "failed"
    assert (
        service.request(NS, {**REQUEST, "end": "2025-01-29"})["status"]
        == "budget_exhausted"
    )
    assert service.snapshots(NS)["usage"]["analytical"] == 2
    with pytest.raises(ValueError, match="immutable"):
        CarbonArcRequests(
            service.state_dir,
            policy=CarbonArcPolicy(allow_remote=True, max_analytical_calls=50),
        )
    with (
        sqlite3.connect(service.path) as conn,
        pytest.raises(sqlite3.IntegrityError),
    ):
        conn.execute("DELETE FROM carbon_request_events")


def test_metadata_uses_free_tool_not_customer_acquisition(world):
    service, calls = world
    service.transport = lambda tool, args: (
        calls.append((tool, args))
        or {
            "mode": "describe",
            "data": {"history": "2019-present", "lag": "T+3 days"},
        }
    )
    result = service.describe(NS, "retail_spend")
    assert result["status"] == "documented" and calls[0] == (
        "data_library",
        {"dataset_id": "card_panel"},
    )
    assert service.snapshots(NS)["usage"] == {
        "metadata": 1,
        "analytical": 0,
        "local": 0,
    }
    assert not result["formation_eligible"]


@pytest.mark.parametrize(
    "vendor,status",
    [
        (
            {"data": "Access denied: not entitled to panel"},
            "entitlement_blocked",
        ),
        ({"metadata": {"rate_limited": True}}, "rate_limited"),
        ({"data": "Quota exhausted"}, "vendor_budget_blocked"),
        (
            {
                "metadata": {"truncated": True},
                "rows": [{"date": "2025-01-01", "value": 1}],
            },
            "incomplete_response",
        ),
        (
            {
                "metadata": {
                    "frameworks": [{"default_date_range_applied": True}]
                }
            },
            "incomplete_response",
        ),
        (
            {
                "metadata": {"insight_id": 999},
                "rows": [{"date": "2025-01-01", "value": 1}],
            },
            "scope_mismatch",
        ),
        (
            {"data": "Narrative answer, no raw observations"},
            "unstructured_result",
        ),
        ({"rows": [{"date": "2025-02-01", "value": 1}]}, "scope_mismatch"),
        (
            {
                "rows": [
                    {"date": "2025-01-01", "value": 1},
                    {"date": "2025-01-01", "value": 2},
                ]
            },
            "duplicate_observations",
        ),
    ],
)
def test_vendor_limitations_are_explicit(world, vendor, status):
    service, _ = world
    service.transport = lambda *_: vendor
    receipt = service.request(NS, REQUEST)
    assert receipt["status"] == status
    assert "data_path" not in receipt
    assert not receipt["formation_eligible"]


def test_exact_markdown_table_extracts_without_interpolating(world):
    service, _ = world
    service.transport = lambda *_: {
        "data": "## Result\n| date | value |\n| --- | ---: |\n| 2025-01-01 | 5 |\n| 2025-01-07 | n/a |"
    }
    r = service.request(NS, REQUEST)
    assert (
        r["status"] == "data_ready"
        and r["rows"] == 2
        and r["missing_values"] == 1
    )
    assert r["observed_end"] == "2025-01-07"


def test_actual_vendor_casing_scientific_values_and_string_false_flags(world):
    service, _ = world
    service.register_capability(
        capability(
            capability_id="retailer_spend",
            entity_rep="RETAILER",
            date_column="DATE",
            value_column="CREDIT CARD SPEND",
            entity_column="ENTITY ID",
        )
    )
    service.transport = lambda *_: {
        "data": "| DATE | ENTITY ID | CREDIT CARD SPEND |\n| --- | --- | --- |\n| 2025-01-01 | 123 | 1.25e+08 |",
        "metadata": {
            "frameworks": [
                {
                    "default_date_range_applied": "False",
                    "insights": [{"insight_id": 55}],
                }
            ]
        },
    }
    r = service.request(NS, {**REQUEST, "capability_id": "retailer_spend"})
    assert r["status"] == "data_ready" and r["metric_scope_observed"]
    assert "125000000.0" in Path(r["data_path"]).read_text()


def test_row_identity_and_publication_mapping_are_host_declared(world):
    service, _ = world
    service.register_capability(
        capability(
            capability_id="mapped_spend",
            entity_column="entity",
            available_column="published",
        )
    )
    service.transport = lambda *_: response(
        [
            {
                "date": "2025-01-01",
                "value": 1,
                "entity": 999,
                "published": "2025-01-03",
            }
        ]
    )
    r = service.request(NS, {**REQUEST, "capability_id": "mapped_spend"})
    assert r["status"] == "scope_mismatch"


def test_local_authorized_data_reused_without_any_network_or_paid_budget(
    tmp_path,
):
    path = tmp_path / "local.csv"
    path.write_text(
        "date,value\n2025-01-01,5\n2025-01-02,\n2026-01-01,DO NOT PARSE\n"
    )
    service = CarbonArcRequests(
        tmp_path / "state",
        transport=lambda *_: pytest.fail("local replay must be free"),
        clock=lambda: NOW,
    )
    service.register_capability(
        capability(
            local_path=str(path),
            local_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    )
    r = service.request(NS, REQUEST)
    assert r["status"] == "data_ready" and r["missing_values"] == 1
    assert service.snapshots(NS)["usage"] == {
        "local": 1,
        "metadata": 0,
        "analytical": 0,
    }
    path.write_text(path.read_text().replace(",5", ",6"))
    changed = service.request(NS, {**REQUEST, "end": "2025-01-30"})
    assert changed["status"] == "failed"


def test_paused_interrupted_analytical_attempt_never_repeats_vendor_call(world):
    service, calls = world

    def interrupted(*_):
        raise KeyboardInterrupt("process interruption")

    service.transport = interrupted
    with pytest.raises(KeyboardInterrupt):
        service.request(NS, REQUEST)
    service.transport = lambda *_: pytest.fail(
        "uncertain attempt cannot retry paid execution"
    )
    r = service.request(NS, REQUEST)
    assert r["status"] == "uncertain"
    assert service.snapshots(NS)["usage"]["analytical"] == 1


def test_bad_request_scope_and_arbitrary_tool_data_cannot_escape_host_capability(
    world,
):
    service, _ = world
    for extra in (
        {"namespace": "pod_events"},
        {"tool": "acquire"},
        {"question": "send secrets"},
        {"path": "/etc/passwd"},
    ):
        with pytest.raises(ValueError):
            service.request(NS, {**REQUEST, **extra})
    for change in ({"end": "2027-01-01"}, {"start": "2015-01-01"}):
        with pytest.raises(ValueError):
            service.request(NS, {**REQUEST, **change})
    with pytest.raises(ValueError, match="immutable"):
        service.register_capability(capability(metric_id=999))
    with pytest.raises(PermissionError):
        CarbonArcOAuthTransport("/tmp", online=False)("text_to_insight", {})
    with pytest.raises(ValueError):
        CarbonArcOAuthTransport("/tmp", online=True)("acquire", {})


def test_extracted_artifact_tampering_is_detected(world):
    service, _ = world
    r = service.request(NS, REQUEST)
    Path(r["data_path"]).write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        service.read(NS, r["request_id"])
