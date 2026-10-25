#!/usr/bin/env python3
"""Plan a minimal registered DESIGN diagnostic; --run consumes a bounded trial.

Settings are read from the matching section of final_v2/config.json.
"""

import json

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.research.component_probe import (
    ComponentProbeService,
    ComponentProbeSpec,
)

SCRIPT_SETTINGS = load_config("_utility_schemas")["run_component_probe"]


def main():

    args = load_script_settings("run_component_probe", SCRIPT_SETTINGS, None)
    spec = ComponentProbeSpec.model_validate_json(args.spec.read_text())
    service = ComponentProbeService(args.manifest, args.state_dir)
    result = (
        service.run(service.register(spec)["probe_id"])
        if args.run
        else service.plan(spec)
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
