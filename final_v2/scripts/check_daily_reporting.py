#!/usr/bin/env python3
"""Validate operator supplied daily CSV coverage; optionally reconcile explicit inputs."""

import json
import sys
from pathlib import Path

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.reporting.daily import reconcile_daily
from pod_platform.reporting.supplied_daily import (
    inputs_from_json,
    read_supplied_daily,
    write_json,
)

SCRIPT_SETTINGS = load_config("_utility_schemas")["check_daily_reporting"]


def main():
    a = load_script_settings("check_daily_reporting", SCRIPT_SETTINGS, None)
    source = read_supplied_daily(
        a.file, a.security_id, a.start, a.end, descriptive=a.descriptive
    )
    result = source.metadata()
    if a.inputs_json:
        inputs, policy = inputs_from_json(
            json.loads(Path(a.inputs_json).read_text()), source
        )
        result.update(
            status="daily_reconciled",
            performance_metrics=reconcile_daily(inputs, policy).as_dict(),
        )
    write_json(a.out, result, overwrite=a.overwrite)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "observations",
                    "trading_days",
                    "missing_return_count",
                    "sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"daily reporting check failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
