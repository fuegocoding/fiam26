"""Run the strategy platform using settings in config.json."""

from __future__ import annotations

import json

from pod_platform.application.strategy_workflow import run_configured


def main():
    """Execute the configured workflow and print its stage results."""
    print(json.dumps(run_configured(), sort_keys=True, default=str))


if __name__ == "__main__":
    main()
