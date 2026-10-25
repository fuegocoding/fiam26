#!/usr/bin/env python3
"""Finite, budgeted Carbon Arc catalogue discovery. No paid analytical calls."""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "final_v2"))

from pod_platform.application.namespaces import RoleContext
from pod_platform.data.carbonarc_discovery import CarbonArcDiscoveryConnector
from pod_platform.data.catalogue import CatalogueManifest
from pod_platform.data.discovery import DiscoveryService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--namespace", choices=["pod_fundamentals", "pod_events", "pod_valuation", "pod_dynamics"], default="pod_fundamentals")
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".pod_state" / "source_discovery")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if not args.online:
        print(json.dumps({"status": "BLOCKED_ONLINE_OPT_IN", "tool": "data_library", "analytical_calls": 0}))
        return 2
    spec = importlib.util.spec_from_file_location("carbonarc_host_transport", ROOT / "scripts/carbonarc/download.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    connection = SimpleNamespace(config=str(Path.home() / ".codex/config.toml"), server="carbonarc", url=None,
                                 auth_dir=ROOT / "local/carbonarc-auth", callback_port=8765, wait_for_login=False)

    async def remote(arguments):
        async with transport.connect(connection, interactive=False) as session:
            return await session.call_tool("data_library", arguments)

    connector = CarbonArcDiscoveryConnector(lambda arguments: asyncio.run(remote(arguments)))
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    service = DiscoveryService(args.state_dir / "discovery.sqlite", catalogue=CatalogueManifest())
    service.register_connector(connector)
    service.set_provider_budget(args.namespace, connector.provider_id, 10)
    context = RoleContext(caller_role="strategy", namespace=args.namespace, run_id="source_discovery",
                          research_asof=dt.datetime.now(dt.timezone.utc))
    result = service.discover(context, args.namespace, connector.provider_id, args.query, limit=args.limit)
    print(json.dumps(result, default=str, indent=2))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
