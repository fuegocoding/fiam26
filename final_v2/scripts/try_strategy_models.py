#!/usr/bin/env python3
"""Prepare or explicitly run the strategy's exact proposed model roles.

Default command `plan` performs no network or model calls. No subcommand admits
a model, fabricates checkpoint evidence, or performs historical investing.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from urllib.parse import urlparse

from pod_platform.agents.model_trials import ModelTrial, load_tasks
from pod_platform.agents.strategy_models import load_profiles
from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings

ROOT = Path(__file__).resolve().parents[1]


def plan(profiles, provenance=None):
    provenance = provenance or {}
    commands = {}
    for role, p in profiles.items():
        if p.protocol != "openai_compatible":
            continue
        endpoint = urlparse(p.base_url)
        if endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
            commands[role] = {
                "status": "HOSTED_API_NO_SERVER_LAUNCH",
                "repository": p.repository,
            }
            continue
        revision = p.revision or provenance.get(p.repository, {}).get(
            "revision"
        )
        if not revision:
            commands[role] = {
                "status": "BLOCKED_PENDING_REVISION",
                "repository": p.repository,
            }
            continue
        command = [
            "vllm",
            "serve",
            p.repository,
            "--revision",
            revision,
            "--tokenizer-revision",
            revision,
            "--served-model-name",
            p.served_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(endpoint.port or 8000),
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            "1",
        ]
        if role == "lead":
            command.extend(["--reasoning-parser", "deepseek_r1"])
        elif role in ("extractor", "lead_upgrade"):
            command.extend(["--reasoning-parser", "qwen3"])
        commands[role] = {
            "status": "PREPARED_NOT_RUNTIME_VERIFIED",
            "command": shlex.join(command),
        }
    return {
        "status": "PREPARED_NO_MODEL_CALLS",
        "configured_roles": sorted(profiles),
        "all_three_roles_configured": {"lead", "challenger", "extractor"}
        <= set(profiles),
        "adoption_allowed": False,
        "historical_routing_enabled": False,
        "serving_commands": commands,
        "compute_note": "Choose compute first. Benchmark sequentially; use workflow --stage to release each model before loading the next.",
        "serving_note": "Commands target self-hosted vLLM, not arbitrary hosted APIs. Pin and verify the engine build on selected hardware.",
        "checkpoint_note": "Repository revisions/LFS hashes are metadata, not proof of which weights a remote endpoint serves.",
    }


SCRIPT_SETTINGS = load_config("_utility_schemas")["try_strategy_models"]


def main(argv=None):

    args = load_script_settings("try_strategy_models", SCRIPT_SETTINGS, argv)
    profiles = load_profiles(args.endpoints)
    if args.command == "plan":
        provenance = (
            json.loads(args.provenance.read_text()).get("models", {})
            if args.provenance.exists()
            else {}
        )
        report = plan(profiles, provenance)
        report["benchmark_tasks"] = len(load_tasks(args.tasks))
    else:
        if not args.online:
            raise ValueError(
                "inference requires --online; use plan to prepare without model calls"
            )
        runner = ModelTrial(
            args.state_dir,
            args.trial_id,
            profiles,
            max_calls=args.max_calls,
            max_tokens=args.max_tokens,
            cycle=args.cycle,
        )
        if args.command == "workflow":
            report = runner.workflow(
                json.loads(args.packet.read_text()), stop_after=args.stage
            )
        else:
            report = runner.benchmark(
                load_tasks(args.tasks),
                args.role,
                limit=args.limit,
                repeats=args.repeats,
            )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return (
        2
        if report.get("status") in ("FAILED_CLOSED", "BENCHMARK_STOPPED")
        else 0
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
