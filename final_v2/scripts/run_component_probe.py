#!/usr/bin/env python3
"""Plan a minimal registered DESIGN diagnostic; --run consumes a bounded trial."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pod_platform.research.component_probe import ComponentProbeService, ComponentProbeSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec = ComponentProbeSpec.model_validate_json(args.spec.read_text())
    service = ComponentProbeService(args.manifest, args.state_dir)
    result = service.run(service.register(spec)['probe_id']) if args.run else service.plan(spec)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
