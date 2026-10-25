#!/usr/bin/env python3
"""Create a local Arcana positions review bundle from explicit canonical JSON."""

import json
import sys
from pathlib import Path

from pod_platform.application.configuration import load_config
from pod_platform.application.script_settings import load_script_settings
from pod_platform.reporting.arcana import make_bundle, write_bundle

SCRIPT_SETTINGS = load_config("_utility_schemas")["export_arcana"]


def main():
    a = load_script_settings("export_arcana", SCRIPT_SETTINGS, None)
    records = json.loads(Path(a.input).read_text(encoding="utf-8"))
    mapping = json.loads(Path(a.mapping).read_text()) if a.mapping else None
    schema = json.loads(Path(a.schema).read_text()) if a.schema else None
    b = make_bundle(
        records,
        mapping=mapping,
        vendor_schema=schema,
        weight_unit=a.weight_unit,
        source_files=[a.input, *a.source_file],
        descriptive=a.descriptive,
    )
    cp, jp = write_bundle(b, a.out, overwrite=a.overwrite)
    print(
        json.dumps(
            {
                "status": b["status"],
                "csv": str(cp),
                "metadata": str(jp),
                "payload_sha256": b["payload_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Arcana export failed: {e}", file=sys.stderr)
        raise SystemExit(2)
