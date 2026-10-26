"""Build an auditable current-quote fee model without reading stock returns."""

import argparse
import json
from pathlib import Path

from pod_platform.application.common import atomic_write_json
from pod_platform.research.borrow_calibration import calibrate_borrow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--size-asof", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_borrow(args.quotes, args.panel, args.size_asof)
    atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
