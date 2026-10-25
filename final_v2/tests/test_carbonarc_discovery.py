import datetime as dt
import pytest

from pod_platform.data.carbonarc_discovery import CarbonArcDiscoveryConnector, provider_reference
from pod_platform.data.discovery import DiscoveryService
from pod_platform.data.catalogue import CatalogueManifest
from pod_platform.application.namespaces import RoleContext


def test_real_catalogue_shape_is_bounded_budgeted_and_not_approved(tmp_path):
    calls = []
    def call(args):
        calls.append(args)
        return {"structuredContent": {"mode": "search", "data": [
            {"tearsheet_name": "card-us-complete-panel", "overview": "Card transactions", "history": "2019-Present", "ecosystem": "demand"},
            {"tearsheet_name": "foot-traffic", "overview": "Store visits"}]}}
    service = DiscoveryService(tmp_path / "discovery.sqlite", catalogue=CatalogueManifest())
    service.register_connector(CarbonArcDiscoveryConnector(call))
    service.set_provider_budget("pod_a", "carbonarc", 1)
    ctx = RoleContext(caller_role="strategy", namespace="pod_a", run_id="r", research_asof=dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc))
    result = service.discover(ctx, "pod_a", "carbonarc", "card spending", limit=1)
    assert calls == [{"query": "card spending"}]
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["provider_ref"] == provider_reference("card-us-complete-panel")
    assert candidate["availability"] == "unresolved_history"
    assert not candidate["formation_eligible"] and not candidate["historical_coverage_established"]
    sample = service.sample(ctx, "pod_a", result["fingerprint"], candidate["provider_ref"])
    assert sample["status"] == "unresolved_history" and len(calls) == 1
    assert service.discover(ctx, "pod_a", "carbonarc", "card spending", limit=1)["cached"]
    assert len(calls) == 1


def test_vendor_errors_and_sampling_cannot_call_analytics():
    connector = CarbonArcDiscoveryConnector(lambda _: {"isError": True})
    with pytest.raises(ValueError):
        connector.discover("retail", as_of=dt.datetime.now(dt.timezone.utc), limit=3)
    with pytest.raises(PermissionError):
        connector.sample("foot_traffic", as_of=dt.datetime.now(dt.timezone.utc), max_rows=1)
